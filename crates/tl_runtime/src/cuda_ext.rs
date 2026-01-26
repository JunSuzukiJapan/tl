use crate::OpaqueTensor;

#[cfg(feature = "cuda")]
use crate::make_tensor;
#[cfg(feature = "cuda")]
use candle_core::Tensor;

// FFI declaration for the C wrapper functions
#[cfg(feature = "cuda")]
extern "C" {
    fn launch_sigmoid_kernel(x: *const f32, y: *mut f32, n: i32, stream: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::sys::CUstream;

/// Applies sigmoid using custom CUDA kernel
/// Returns new OpaqueTensor
#[unsafe(no_mangle)]
#[allow(unused_variables)]
pub extern "C" fn tl_cuda_sigmoid(t: *const OpaqueTensor) -> *mut OpaqueTensor {
    #[cfg(feature = "cuda")]
    unsafe {
        if t.is_null() {
            return std::ptr::null_mut();
        }
        let tensor = &(*t).0;

        if !tensor.device().is_cuda() {
            eprintln!("Error: tl_cuda_sigmoid requires CUDA tensor");
            return std::ptr::null_mut();
        }

        // Ensure contiguous standard layout for raw pointer access
        let t_storage = match tensor.to_dtype(candle_core::DType::F32) {
            Ok(t) => t,
            Err(_) => return std::ptr::null_mut(),
        };
        let t_cont = match t_storage.contiguous() {
            Ok(t) => t,
            Err(_) => return std::ptr::null_mut(),
        };

        let num_elements = t_cont.elem_count();
        let dims = t_cont.dims();

        // Allocate result tensor on same device
        let res_t = match Tensor::zeros(dims, candle_core::DType::F32, tensor.device()) {
            Ok(t) => t,
            Err(_) => return std::ptr::null_mut(),
        };

        // Get Raw Pointers (Requires internal candle access or using cudarc slice)
        // Candle's `as_cuda_slice` returns a device slice.
        // We need the pointer from it.

        // NOTE: High-level access to raw pointers in Candle is tricky without direct cudarc access.
        // But assuming we can get the storage:

        let (storage, _) = t_cont.storage_and_layout();
        let (res_storage, _) = res_t.storage_and_layout();

        match (storage, res_storage) {
            (candle_core::Storage::Cuda(lhs), candle_core::Storage::Cuda(rhs)) => {
                let lhs_ptr = *lhs.as_cuda_slice::<f32>().unwrap().device_ptr() as *const f32;
                let rhs_ptr = *rhs.as_cuda_slice::<f32>().unwrap().device_ptr() as *mut f32;

                // For stream, we ideally use the device's current stream.
                // Here passing null (0) for default stream for simplicity.
                let stream = std::ptr::null_mut();

                launch_sigmoid_kernel(lhs_ptr, rhs_ptr, num_elements as i32, stream);

                return make_tensor(res_t);
            }
            _ => {
                eprintln!("Error: Storage mismatch in cuda_sigmoid");
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("Error: CUDA feature not enabled");
    }

    std::ptr::null_mut()
}
