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
        eprintln!("DEBUG: cpu_fwd start");
        // Recover Tensor from Storage
        let data = s1.as_slice::<f32>()?;
        let shape = l.shape();
        // Create tensor on current device
        let device = get_device();
        let t_cpu = Tensor::from_slice(data, shape, &candle_core::Device::Cpu)?;
        let t = if device.is_metal() || device.is_cuda() {
            t_cpu.to_device(&device)?
        } else {
            t_cpu
        };

        let t_ptr = make_tensor(t);
        eprintln!("DEBUG: cpu_fwd calling func with input {:p}", t_ptr);
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);
        eprintln!("DEBUG: cpu_fwd func returned {:p}", out_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in cpu_fwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;
        let out_cpu = out_tensor.to_device(&candle_core::Device::Cpu)?;
        let (storage, out_layout) = out_cpu.storage_and_layout();

        // We need to return CpuStorage.
        // Assuming Output is on CPU.
        match storage.deref() {
            candle_core::Storage::Cpu(cpu_storage) => {
                let res_storage = cpu_storage.clone();
                let res_shape = out_layout.shape().clone();

                // Cleanup with proper release
                tl_tensor_release(t_ptr);
                tl_tensor_release(out_ptr);

                Ok((res_storage, res_shape))
            }
            _ => Err(candle_core::Error::Msg(
                "Checkpoint output must be on CPU for cpu_fwd".into(),
            )),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        // use candle_core::backend::BackendStorage; // Unused
        eprintln!("DEBUG: metal_fwd start");

        // Create tensor from MetalStorage
        let shape = layout.shape();
        let device = get_device();

        #[cfg(feature = "metal")]
        let cpu_tensor = {
            // Manual copy from Private Metal Buffer to CPU using metal crate directly
            use metal::MTLResourceOptions; // MTLStorageMode unused

            let elem_count = shape.elem_count();
            let size = (elem_count * std::mem::size_of::<f32>()) as u64;

            // storage.buffer() returns &metal::Buffer directly
            let buffer = storage.buffer(); // &metal::Buffer
            let mtl_device = buffer.device(); // &metal::Device (raw from metal crate)

            // Create a Shared buffer for reading
            let options = MTLResourceOptions::StorageModeShared;
            let read_buffer = mtl_device.new_buffer(size, options);

            // Create Command Queue & Buffer
            let command_queue = mtl_device.new_command_queue();
            let command_buffer = command_queue.new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();

            // Encode Copy
            blit_encoder.copy_from_buffer(buffer, 0, &read_buffer, 0, size);
            blit_encoder.end_encoding();

            // Commit and Wait
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // Read data
            let ptr = read_buffer.contents() as *const f32;
            if ptr.is_null() {
                return Err(candle_core::Error::Msg(
                    "Failed to read from shared buffer (null ptr)".into(),
                ));
            }
            let cpu_data = unsafe { std::slice::from_raw_parts(ptr, elem_count) }.to_vec();

            Tensor::from_vec(cpu_data, shape, &candle_core::Device::Cpu)?
        };

        #[cfg(not(feature = "metal"))]
        let cpu_tensor = {
            return Err(candle_core::Error::Msg(
                "metal feature not enabled in tl_runtime".into(),
            ));
        };

        // Move to Metal device
        let metal_tensor = cpu_tensor.to_device(&device)?;

        // Call the user's function
        let t_ptr = make_tensor(metal_tensor);
        eprintln!("DEBUG: metal_fwd calling func");
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);
        eprintln!("DEBUG: metal_fwd func returned {:p}", out_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in metal_fwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;

        // Get the output storage
        let (out_storage, out_layout) = out_tensor.storage_and_layout();

        match out_storage.deref() {
            candle_core::Storage::Metal(metal_storage) => {
                let res_storage = metal_storage.clone();
                let res_shape = out_layout.shape().clone();

                // Cleanup with proper release
                tl_tensor_release(t_ptr);
                tl_tensor_release(out_ptr);

                Ok((res_storage, res_shape))
            }
            candle_core::Storage::Cpu(cpu_storage) => {
                // Output is on CPU, need to copy to Metal
                // Get data from CpuStorage and create new tensor
                let cpu_slice = cpu_storage.as_slice::<f32>()?;
                let cpu_tensor =
                    Tensor::from_slice(cpu_slice, out_layout.shape(), &candle_core::Device::Cpu)?;
                let metal_tensor = cpu_tensor.to_device(&device)?;
                let (metal_storage, metal_layout) = metal_tensor.storage_and_layout();

                match metal_storage.deref() {
                    candle_core::Storage::Metal(ms) => {
                        // Cleanup
                        tl_tensor_release(t_ptr);
                        tl_tensor_release(out_ptr);
                        Ok((ms.clone(), metal_layout.shape().clone()))
                    }
                    _ => Err(candle_core::Error::Msg(
                        "Failed to convert to Metal storage".into(),
                    )),
                }
            }
            _ => Err(candle_core::Error::Msg(
                "Checkpoint output has unsupported storage type".into(),
            )),
        }
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, out_grad: &Tensor) -> Result<Option<Tensor>> {
        eprintln!("DEBUG: bwd start");
        // Checkpointing logic:
        // Rebuild graph: Block(arg) -> out_tensor
        // Important: arg needs to be a Var to track gradients during recomputation
        let arg_var = candle_core::Var::from_tensor(arg)?;
        let t_ptr = make_tensor(arg_var.as_tensor().clone());

        eprintln!("DEBUG: bwd calling func");
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);
        eprintln!("DEBUG: bwd func returned {:p}", out_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in bwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;

        // Ensure devices match (out_grad comes from outside)
        let out_grad = out_grad.to_device(out_tensor.device())?;

        // Backprop through: (out_tensor * out_grad).sum()
        let loss = (out_tensor * out_grad)?;
        let loss = loss.sum_all()?;

        let mut new_grads = loss.backward()?;

        let arg_grad = new_grads.remove(&arg_var);

        // Cleanup with proper release
        tl_tensor_release(t_ptr);
        tl_tensor_release(out_ptr);

        Ok(arg_grad)
    }
}

// Runtime API to start checkpoint
#[no_mangle]
pub extern "C" fn tl_checkpoint(
    ctx: *mut c_void,
    func: extern "C" fn(*mut c_void, *mut OpaqueTensor) -> *mut OpaqueTensor,
    input: *mut OpaqueTensor,
) -> *mut OpaqueTensor {
    eprintln!("DEBUG: tl_checkpoint called");
    unsafe {
        let input_tensor = &(*input).0;
        let op = TlCheckpointOp {
            ctx: ContextPtr(ctx),
            func: FunctionPtr(func),
        };

        // Apply the custom op
        match input_tensor.apply_op1(op) {
            Ok(ret_tensor) => make_tensor(ret_tensor),
            Err(e) => {
                eprintln!("Checkpoint error: {}", e);
                std::ptr::null_mut()
            }
        }
    }
}
