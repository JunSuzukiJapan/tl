use crate::{device::get_device, make_tensor, memory_manager::tl_tensor_release, OpaqueTensor};
use candle_core::{CustomOp1, Layout, Result, Shape, Tensor};
use std::ffi::c_void;
use std::ops::Deref;

#[derive(Clone)]
pub(crate) struct ContextPtr(pub *mut c_void);
unsafe impl Send for ContextPtr {}
unsafe impl Sync for ContextPtr {}

#[derive(Clone)]
pub(crate) struct FunctionPtr(
    pub extern "C" fn(*mut c_void, *mut OpaqueTensor) -> *mut OpaqueTensor,
);
unsafe impl Send for FunctionPtr {}
unsafe impl Sync for FunctionPtr {}

#[derive(Clone)]
pub struct TlCheckpointOp {
    pub(crate) ctx: ContextPtr,
    pub(crate) func: FunctionPtr,
}

impl CustomOp1 for TlCheckpointOp {
    fn name(&self) -> &'static str {
        "checkpoint"
    }

    fn cpu_fwd(
        &self,
        s1: &candle_core::CpuStorage,
        l: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        let data = s1.as_slice::<f32>()?;
        let shape = l.shape();
        let device = get_device();
        
        let t_cpu = Tensor::from_slice(data, shape, &candle_core::Device::Cpu)?;
        let t = if device.is_cpu() {
            t_cpu
        } else {
            t_cpu.to_device(&device)?
        };

        let t_ptr = make_tensor(t);
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in cpu_fwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = out_opaque.as_tensor().map_err(|e| candle_core::Error::Msg(e.into()))?;
        let out_cpu = out_tensor.to_device(&candle_core::Device::Cpu)?;
        let (storage, out_layout) = out_cpu.storage_and_layout();

        match storage.deref() {
            candle_core::Storage::Cpu(cpu_storage) => {
                let res_storage = cpu_storage.clone();
                let res_shape = out_layout.shape().clone();

                // Cleanup
                tl_tensor_release(t_ptr);
                tl_tensor_release(out_ptr);

                Ok((res_storage, res_shape))
            }
            _ => Err(candle_core::Error::Msg(
                "Checkpoint output must be on CPU for cpu_fwd".into(),
            )),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        _s1: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        Err(candle_core::Error::Msg("cuda_fwd in checkpoint requires manual implementation for official candle branch".into()))
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        _storage: &candle_core::MetalStorage,
        _layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        Err(candle_core::Error::Msg("metal_fwd in checkpoint requires manual implementation for official candle branch".into()))
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, out_grad: &Tensor) -> Result<Option<Tensor>> {
        let arg_var = candle_core::Var::from_tensor(arg)?;
        let t_ptr = make_tensor(arg_var.as_tensor().clone());

        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in bwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = out_opaque.as_tensor().map_err(|e| candle_core::Error::Msg(e.into()))?;

        let out_grad = out_grad.to_device(out_tensor.device())?;

        let loss = (out_tensor * out_grad)?;
        let loss = loss.sum_all()?;

        let mut new_grads = loss.backward()?;

        let arg_grad = new_grads.remove(&arg_var);

        tl_tensor_release(t_ptr);
        tl_tensor_release(out_ptr);

        Ok(arg_grad)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn tl_checkpoint(
    ctx: *mut c_void,
    func: extern "C" fn(*mut c_void, *mut OpaqueTensor) -> *mut OpaqueTensor,
    input: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    unsafe {
        let input_tensor = match (*input).as_tensor() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Checkpoint input error: {}", e);
                return std::ptr::null_mut();
            }
        };
        let op = TlCheckpointOp {
            ctx: ContextPtr(ctx),
            func: FunctionPtr(func),
        };

        match input_tensor.apply_op1(op) {
            Ok(ret_tensor) => make_tensor(ret_tensor),
            Err(e) => {
                eprintln!("Checkpoint error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}
