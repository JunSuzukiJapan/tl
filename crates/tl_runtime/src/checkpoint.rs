use crate::{make_tensor, memory_manager::tl_tensor_release, OpaqueTensor};
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
        eprintln!("DEBUG: cpu_fwd start (zero-copy)");
        let shape = l.shape();
        // Zero-copy: wrap existing storage in a Tensor
        // NOTE: CpuStorage clone() is expensive because it's a Vec in this Candle version.
        // But we still use from_storage to avoid the previous triple-copy logic.
        let storage = candle_core::Storage::Cpu(s1.clone());
        let t = candle_core::from_storage(storage, shape.clone(), candle_core::BackpropOp::none(), false);

        let t_ptr = make_tensor(t);
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in cpu_fwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;
        let (storage, out_layout) = out_tensor.storage_and_layout();

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
        s1: &candle_core::CudaStorage,
        l: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        eprintln!("DEBUG: cuda_fwd start (zero-copy)");
        let shape = l.shape();
        // Use try_clone for zero-copy (shares the slice internally in real backend)
        let storage = candle_core::Storage::Cuda(s1.try_clone(l)?);
        let t = candle_core::from_storage(storage, shape.clone(), candle_core::BackpropOp::none(), false);

        let t_ptr = make_tensor(t);
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in cuda_fwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;
        let (storage, out_layout) = out_tensor.storage_and_layout();

        match storage.deref() {
            candle_core::Storage::Cuda(cuda_storage) => {
                let res_storage = cuda_storage.try_clone(out_layout)?;
                let res_shape = out_layout.shape().clone();

                // Cleanup
                tl_tensor_release(t_ptr);
                tl_tensor_release(out_ptr);

                Ok((res_storage, res_shape))
            }
            _ => Err(candle_core::Error::Msg(
                "Checkpoint output must be on CUDA for cuda_fwd".into(),
            )),
        }
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        storage: &candle_core::MetalStorage,
        layout: &Layout,
    ) -> Result<(candle_core::MetalStorage, Shape)> {
        eprintln!("DEBUG: metal_fwd start (zero-copy)");
        let shape = layout.shape();
        
        // Zero-copy: wrap existing storage in a Tensor
        let s = candle_core::Storage::Metal(storage.clone());
        let metal_tensor = candle_core::from_storage(s, shape.clone(), candle_core::BackpropOp::none(), false);

        // Call the user's function
        let t_ptr = make_tensor(metal_tensor);
        let out_ptr = (self.func.0)(self.ctx.0, t_ptr);

        if out_ptr.is_null() {
            tl_tensor_release(t_ptr);
            return Err(candle_core::Error::Msg(
                "Checkpoint function returned null in metal_fwd".into(),
            ));
        }

        let out_opaque = unsafe { &*out_ptr };
        let out_tensor = &out_opaque.0;
        let (out_storage, out_layout) = out_tensor.storage_and_layout();

        match out_storage.deref() {
            candle_core::Storage::Metal(metal_storage) => {
                let res_storage = metal_storage.clone();
                let res_shape = out_layout.shape().clone();

                // Cleanup
                tl_tensor_release(t_ptr);
                tl_tensor_release(out_ptr);

                Ok((res_storage, res_shape))
            }
            _ => Err(candle_core::Error::Msg(
                "Checkpoint output must be on Metal for metal_fwd".into(),
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
