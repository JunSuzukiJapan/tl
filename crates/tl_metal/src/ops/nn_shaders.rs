//! NN操作の Metal GPU シェーダディスパッチ

use crate::device::get_device;
use crate::shaders::{self, SHADER_CONV1D_F32, SHADER_CONV_TRANSPOSE2D_F32,
    SHADER_INTERPOLATE_NEAREST_F32, SHADER_INTERPOLATE_BILINEAR_F32,
    SHADER_ADAPTIVE_AVG_POOL2D_F32, SHADER_PAD_F32, SHADER_CUMSUM_F32,
    SHADER_GROUP_NORM_F32};
use crate::tensor::MetalTensor;
use crate::DType;
use metal::MTLResourceOptions;
use tl_backend::{BackendResult, BackendError};

/// MSL struct Conv1dParams と同じレイアウト (u32 x 8)
#[repr(C)]
struct Conv1dParams {
    batch: u32,
    in_ch: u32,
    in_len: u32,
    out_ch: u32,
    k_len: u32,
    stride: u32,
    padding: u32,
    out_len: u32,
}

/// MSL struct ConvTranspose2dParams と同じレイアウト (u32 x 11)
#[repr(C)]
struct ConvTranspose2dParams {
    batch: u32,
    in_ch: u32,
    ih: u32,
    iw: u32,
    out_ch: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    padding: u32,
    oh: u32,
    ow: u32,
}

/// MSL struct InterpolateParams と同じレイアウト (u32 x 6)
#[repr(C)]
struct InterpolateParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
}

/// MSL struct PadParams と同じレイアウト (u32 x 4)
#[repr(C)]
struct PadParams {
    outer: u32,
    old_last: u32,
    new_last: u32,
    pad_left: u32,
}

/// MSL struct CumsumParams と同じレイアウト (u32 x 2)
#[repr(C)]
struct CumsumParams {
    outer: u32,
    inner: u32,
}

/// MSL struct GroupNormParams と同じレイアウト (u32 x 5 + f32 x 1)
#[repr(C)]
struct GroupNormParams {
    batch: u32,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    group_size: u32,
    eps: f32,
}

fn make_params_buf<T>(params: &T) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        params as *const T as *const _,
        std::mem::size_of::<T>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn make_scalar_buf(val: u32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &val as *const u32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    )
}

fn make_f32_buf(val: f32) -> metal::Buffer {
    let device = get_device();
    device.device().new_buffer_with_data(
        &val as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    )
}

/// ダミーバッファ (bias が null な場合用)
fn dummy_buf() -> metal::Buffer {
    let device = get_device();
    let zero: f32 = 0.0;
    device.device().new_buffer_with_data(
        &zero as *const f32 as *const _,
        4,
        MTLResourceOptions::StorageModeShared,
    )
}

impl MetalTensor {
    /// Conv1d GPU 実装
    pub fn conv1d_impl(
        &self,
        weight: &MetalTensor,
        bias: Option<&MetalTensor>,
        stride: usize,
        padding: usize,
    ) -> BackendResult<MetalTensor> {
        let ishape = self.shape();
        let wshape = weight.shape();
        let (batch, in_ch, in_len) = (ishape[0], ishape[1], ishape[2]);
        let (out_ch, _wch, k_len) = (wshape[0], wshape[1], wshape[2]);
        let out_len = (in_len + 2 * padding - k_len) / stride + 1;

        let out_shape = [batch, out_ch, out_len];
        let result = MetalTensor::uninit(&out_shape, DType::F32);
        let elem_count = batch * out_ch * out_len;

        let params = Conv1dParams {
            batch: batch as u32, in_ch: in_ch as u32, in_len: in_len as u32,
            out_ch: out_ch as u32, k_len: k_len as u32,
            stride: stride as u32, padding: padding as u32, out_len: out_len as u32,
        };
        let params_buf = make_params_buf(&params);
        let has_bias_val: u32 = if bias.is_some() { 1 } else { 0 };
        let has_bias_buf = make_scalar_buf(has_bias_val);
        let bias_buf_holder = dummy_buf();

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_CONV1D_F32)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let weight_buf = weight.buffer() as *const metal::Buffer;
        let bias_ptr = bias.map(|b| b.buffer() as *const metal::Buffer);
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;
        let has_bias_ptr = &has_bias_buf as *const metal::Buffer;
        let bias_holder_ptr = &bias_buf_holder as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                encoder.set_buffer(1, Some(&*weight_buf), 0);
                if let Some(bp) = bias_ptr {
                    encoder.set_buffer(2, Some(&*bp), 0);
                } else {
                    encoder.set_buffer(2, Some(&*bias_holder_ptr), 0);
                }
                encoder.set_buffer(3, Some(&*result_buf), 0);
                encoder.set_buffer(4, Some(&*params_ptr), 0);
                encoder.set_buffer(5, Some(&*has_bias_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }

    /// ConvTranspose2d GPU 実装
    pub fn conv_transpose2d_impl(
        &self,
        weight: &MetalTensor,
        bias: Option<&MetalTensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> BackendResult<MetalTensor> {
        let ishape = self.shape();
        let wshape = weight.shape();
        let (batch, in_ch, ih, iw) = (ishape[0], ishape[1], ishape[2], ishape[3]);
        let (_, out_ch, kh, kw) = (wshape[0], wshape[1], wshape[2], wshape[3]);
        let oh = (ih - 1) * stride - 2 * padding + kh + output_padding;
        let ow = (iw - 1) * stride - 2 * padding + kw + output_padding;

        let out_shape = [batch, out_ch, oh, ow];
        let result = MetalTensor::uninit(&out_shape, DType::F32);
        let elem_count = batch * out_ch * oh * ow;

        let params = ConvTranspose2dParams {
            batch: batch as u32, in_ch: in_ch as u32, ih: ih as u32, iw: iw as u32,
            out_ch: out_ch as u32, kh: kh as u32, kw: kw as u32,
            stride: stride as u32, padding: padding as u32, oh: oh as u32, ow: ow as u32,
        };
        let params_buf = make_params_buf(&params);
        let has_bias_val: u32 = if bias.is_some() { 1 } else { 0 };
        let has_bias_buf = make_scalar_buf(has_bias_val);
        let bias_buf_holder = dummy_buf();

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_CONV_TRANSPOSE2D_F32)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let weight_buf = weight.buffer() as *const metal::Buffer;
        let bias_ptr = bias.map(|b| b.buffer() as *const metal::Buffer);
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;
        let has_bias_ptr = &has_bias_buf as *const metal::Buffer;
        let bias_holder_ptr = &bias_buf_holder as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                encoder.set_buffer(1, Some(&*weight_buf), 0);
                if let Some(bp) = bias_ptr {
                    encoder.set_buffer(2, Some(&*bp), 0);
                } else {
                    encoder.set_buffer(2, Some(&*bias_holder_ptr), 0);
                }
                encoder.set_buffer(3, Some(&*result_buf), 0);
                encoder.set_buffer(4, Some(&*params_ptr), 0);
                encoder.set_buffer(5, Some(&*has_bias_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }

    /// Interpolate GPU 実装 (mode=0: nearest, mode=1: bilinear)
    pub fn interpolate_impl(
        &self,
        out_h: usize,
        out_w: usize,
        mode: i64,
    ) -> BackendResult<MetalTensor> {
        let shape = self.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_shape = [n, c, out_h, out_w];
        let result = MetalTensor::uninit(&out_shape, DType::F32);
        let elem_count = n * c * out_h * out_w;

        let params = InterpolateParams {
            batch: n as u32, channels: c as u32, in_h: h as u32, in_w: w as u32,
            out_h: out_h as u32, out_w: out_w as u32,
        };
        let params_buf = make_params_buf(&params);

        let shader_name = if mode == 0 { SHADER_INTERPOLATE_NEAREST_F32 } else { SHADER_INTERPOLATE_BILINEAR_F32 };

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), shader_name)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
                encoder.set_buffer(2, Some(&*params_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }

    /// Adaptive Average Pool 2D GPU 実装
    pub fn adaptive_avg_pool2d_impl(
        &self,
        out_h: usize,
        out_w: usize,
    ) -> BackendResult<MetalTensor> {
        let shape = self.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let out_shape = [n, c, out_h, out_w];
        let result = MetalTensor::uninit(&out_shape, DType::F32);
        let elem_count = n * c * out_h * out_w;

        let params = InterpolateParams {
            batch: n as u32, channels: c as u32, in_h: h as u32, in_w: w as u32,
            out_h: out_h as u32, out_w: out_w as u32,
        };
        let params_buf = make_params_buf(&params);

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_ADAPTIVE_AVG_POOL2D_F32)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
                encoder.set_buffer(2, Some(&*params_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }

    /// Pad GPU 実装 (最後の次元にパディング)
    pub fn pad_impl(
        &self,
        pad_left: usize,
        pad_right: usize,
        value: f32,
    ) -> BackendResult<MetalTensor> {
        let shape = self.shape();
        let last = *shape.last().unwrap();
        let new_last = last + pad_left + pad_right;
        let outer: usize = shape[..shape.len() - 1].iter().product::<usize>().max(1);

        let mut new_shape = shape.to_vec();
        *new_shape.last_mut().unwrap() = new_last;
        let result = MetalTensor::uninit(&new_shape, DType::F32);
        let elem_count = outer * new_last;

        let params = PadParams {
            outer: outer as u32,
            old_last: last as u32,
            new_last: new_last as u32,
            pad_left: pad_left as u32,
        };
        let params_buf = make_params_buf(&params);
        let value_buf = make_f32_buf(value);

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_PAD_F32)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;
        let value_ptr = &value_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
                encoder.set_buffer(2, Some(&*params_ptr), 0);
                encoder.set_buffer(3, Some(&*value_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(elem_count, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }

    /// Cumsum GPU 実装 (外側次元を並列化)
    pub fn cumsum_impl(&self, dim: i32) -> BackendResult<MetalTensor> {
        let shape = self.shape();
        let ndim = shape.len();
        let d = if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };
        let _inner = shape[d..].iter().product::<usize>();
        let _outer = shape[..d].iter().product::<usize>().max(1);
        // cumsum の inner はdim以降だが、cumsumは1次元分だけ
        // 簡易実装: flatten して outer x inner で分ける
        let inner_size = shape[d];
        let outer_size = self.elem_count() / inner_size;

        let result = MetalTensor::uninit(shape, DType::F32);

        let params = CumsumParams {
            outer: outer_size as u32,
            inner: inner_size as u32,
        };
        let params_buf = make_params_buf(&params);

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_CUMSUM_F32)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                encoder.set_buffer(1, Some(&*result_buf), 0);
                encoder.set_buffer(2, Some(&*params_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(outer_size, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }

    /// Group Norm GPU 実装
    pub fn group_norm_impl(
        &self,
        num_groups: usize,
        weight: Option<&MetalTensor>,
        bias: Option<&MetalTensor>,
        eps: f32,
    ) -> BackendResult<MetalTensor> {
        let shape = self.shape();
        let channels = if shape.len() >= 2 { shape[1] } else { shape[0] };
        let spatial: usize = if shape.len() > 2 { shape[2..].iter().product() } else { 1 };
        let batch = if shape.len() >= 2 { shape[0] } else { 1 };
        let group_size = channels / num_groups;

        let result = MetalTensor::uninit(shape, DType::F32);
        let thread_count = batch * num_groups;

        let params = GroupNormParams {
            batch: batch as u32,
            channels: channels as u32,
            spatial: spatial as u32,
            num_groups: num_groups as u32,
            group_size: group_size as u32,
            eps,
        };
        let params_buf = make_params_buf(&params);
        let has_weight_val: u32 = if weight.is_some() { 1 } else { 0 };
        let has_bias_val: u32 = if bias.is_some() { 1 } else { 0 };
        let has_weight_buf = make_scalar_buf(has_weight_val);
        let has_bias_buf = make_scalar_buf(has_bias_val);
        let dummy = dummy_buf();

        let device = get_device();
        let mut shaders = shaders::get_shaders().lock().unwrap();
        let pipeline = shaders.get_pipeline(device.device(), SHADER_GROUP_NORM_F32)
            .map_err(|e| BackendError::InternalError(e))?;

        let input_buf = self.buffer() as *const metal::Buffer;
        let weight_ptr = weight.map(|w| w.buffer() as *const metal::Buffer);
        let bias_ptr = bias.map(|b| b.buffer() as *const metal::Buffer);
        let result_buf = result.buffer() as *const metal::Buffer;
        let params_ptr = &params_buf as *const metal::Buffer;
        let hw_ptr = &has_weight_buf as *const metal::Buffer;
        let hb_ptr = &has_bias_buf as *const metal::Buffer;
        let dummy_ptr = &dummy as *const metal::Buffer;

        crate::command_stream::stream_encode(|encoder| {
            encoder.set_compute_pipeline_state(pipeline);
            unsafe {
                encoder.set_buffer(0, Some(&*input_buf), 0);
                if let Some(wp) = weight_ptr {
                    encoder.set_buffer(1, Some(&*wp), 0);
                } else {
                    encoder.set_buffer(1, Some(&*dummy_ptr), 0);
                }
                if let Some(bp) = bias_ptr {
                    encoder.set_buffer(2, Some(&*bp), 0);
                } else {
                    encoder.set_buffer(2, Some(&*dummy_ptr), 0);
                }
                encoder.set_buffer(3, Some(&*result_buf), 0);
                encoder.set_buffer(4, Some(&*params_ptr), 0);
                encoder.set_buffer(5, Some(&*hw_ptr), 0);
                encoder.set_buffer(6, Some(&*hb_ptr), 0);
            }
            let (grid, tpg) = shaders::compute_thread_groups(thread_count, pipeline);
            encoder.dispatch_thread_groups(grid, tpg);
        });

        Ok(result)
    }
}
