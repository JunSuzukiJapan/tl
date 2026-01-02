use candle_core::Device;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum DeviceType {
    Cpu,
    Metal,
    Cuda,
}

pub struct DeviceManager {
    current_device: Device,
    #[allow(dead_code)]
    device_type: DeviceType,
}

impl DeviceManager {
    pub fn new() -> Self {
        let requested_device = std::env::var("TL_DEVICE").unwrap_or_else(|_| "auto".to_string());

        // Priority: CUDA -> Metal -> CPU if auto
        if requested_device == "cuda"
            || (requested_device == "auto" && candle_core::utils::cuda_is_available())
        {
            #[cfg(feature = "cuda")]
            {
                println!("Initializing Runtime: CUDA backend selected.");
                match Device::new_cuda(0) {
                    Ok(device) => {
                        return DeviceManager {
                            current_device: device,
                            device_type: DeviceType::Cuda,
                        }
                    }
                    Err(e) => eprintln!("Failed to initialize CUDA: {}. Falling back.", e),
                }
            }
            #[cfg(not(feature = "cuda"))]
            if requested_device == "cuda" {
                eprintln!("CUDA requested but 'cuda' feature not enabled.");
            }
        }

        if requested_device == "metal"
            || (requested_device == "auto" && candle_core::utils::metal_is_available())
        {
            #[cfg(feature = "metal")]
            {
                println!("Initializing Runtime: Metal backend selected.");
                match Device::new_metal(0) {
                    Ok(device) => {
                        if check_metal_health(&device) {
                            return DeviceManager {
                                current_device: device,
                                device_type: DeviceType::Metal,
                            };
                        } else {
                            eprintln!("WARNING: Metal backend failed self-test (returned incorrect results). Falling back to CPU.");
                        }
                    }
                    Err(e) => eprintln!("Failed to initialize Metal: {}. Falling back.", e),
                }
            }
            #[cfg(not(feature = "metal"))]
            if requested_device == "metal" {
                eprintln!("Metal requested but 'metal' feature not enabled.");
            }
        }

        println!("Initializing Runtime: CPU backend selected.");
        DeviceManager {
            current_device: Device::Cpu,
            device_type: DeviceType::Cpu,
        }
    }

    pub fn device(&self) -> &Device {
        &self.current_device
    }
}

#[cfg(feature = "metal")]
fn check_metal_health(device: &Device) -> bool {
    let t_cpu = match candle_core::Tensor::new(&[1.0f32], &candle_core::Device::Cpu) {
        Ok(t) => t,
        Err(_) => return false,
    };

    let t_metal = match t_cpu.to_device(device) {
        Ok(t) => t,
        Err(_) => return false,
    };

    let t_res = match t_metal.add(&t_metal) {
        Ok(t) => t,
        Err(_) => return false,
    };

    let t_back = match t_res.to_device(&candle_core::Device::Cpu) {
        Ok(t) => t,
        Err(_) => return false,
    };

    let vals = match t_back.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(_) => return false,
    };

    // 1.0 + 1.0 = 2.0
    if vals.len() == 1 && (vals[0] - 2.0).abs() < 1e-5 {
        true
    } else {
        false
    }
}

// Global Singleton for Device Manager
lazy_static! {
    pub static ref DEVICE_MANAGER: Arc<Mutex<DeviceManager>> =
        Arc::new(Mutex::new(DeviceManager::new()));
}

// Thread-local cached device to avoid repeated Mutex locks
thread_local! {
    static CACHED_DEVICE: std::cell::RefCell<Option<Device>> = const { std::cell::RefCell::new(None) };
}

pub fn get_device() -> Device {
    CACHED_DEVICE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.is_none() {
            *cache = Some(DEVICE_MANAGER.lock().unwrap().device().clone());
        }
        cache.as_ref().unwrap().clone()
    })
}
