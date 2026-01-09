use candle_core::{Tensor, CustomOp1, Layout, Shape, Result};
use crate::runtime::{device::get_device, make_tensor, OpaqueTensor};
use std::ffi::c_void;
use std::ops::Deref;

#[derive(Clone)]
pub(crate) struct ContextPtr(pub *mut c_void);
unsafe impl Send for ContextPtr {}
unsafe impl Sync for ContextPtr {}

#[derive(Clone)]
pub(crate) struct FunctionPtr(pub extern "C" fn(*mut c_void, *mut OpaqueTensor) -> *mut OpaqueTensor);
unsafe impl Send for FunctionPtr {}
unsafe impl Sync for FunctionPtr {}

#[derive(Clone)]
pub struct TlCheckpointOp {
    pub(crate) ctx: ContextPtr,
    pub(crate) func: FunctionPtr,
}

impl CustomOp1 for TlCheckpointOp {
    fn name(&self) -> &'static str { "checkpoint" }

    fn cpu_fwd(&self, s1: &candle_core::CpuStorage, l: &Layout) -> Result<(candle_core::CpuStorage, Shape)> {
        // Recover Tensor from Storage (copying data for now as specific API is unknown)
        // Assuming F32 for now
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
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);
        
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
                 
                 // Cleanup
                 unsafe {
                     let _ = Box::from_raw(t_ptr);
                     let _ = Box::from_raw(out_ptr);
                 }
                 
                 Ok((res_storage, res_shape))
             }
             _ => Err(candle_core::Error::Msg("Checkpoint output must be on CPU for cpu_fwd".into()))
        }
    }

    fn bwd(&self, arg: &Tensor, _res: &Tensor, out_grad: &Tensor) -> Result<Option<Tensor>> {
        // Checkpointing logic:
        // Rebuild graph: Block(arg) -> out_tensor
        // Important: arg needs to be a Var to track gradients during recomputation
        let arg_var = candle_core::Var::from_tensor(arg)?;
        let t_ptr = make_tensor(arg_var.as_tensor().clone());
        
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);
        
        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;
        
        // Ensure devices match (out_grad comes from outside)
        let out_grad = out_grad.to_device(out_tensor.device())?;
        
        // Backprop through: (out_tensor * out_grad).sum()
        let loss = (out_tensor * out_grad)?;
        let loss = loss.sum_all()?;
        
        let mut new_grads = loss.backward()?;
        
        let arg_grad = new_grads.remove(&arg_var);
        
        unsafe {
            // make_tensor boxed the OpaqueTensor, need to free it.
            let _ = Box::from_raw(t_ptr);
            let _ = Box::from_raw(out_ptr);
        }
        
        Ok(arg_grad)
    }
}

// Runtime API to start checkpoint
#[no_mangle]
pub extern "C" fn tl_checkpoint(
    ctx: *mut c_void,
    func: extern "C" fn(*mut c_void, *mut OpaqueTensor) -> *mut OpaqueTensor,
    input: *mut OpaqueTensor
) -> *mut OpaqueTensor {
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
