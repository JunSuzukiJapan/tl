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
                        return DeviceManager {
                            current_device: device,
                            device_type: DeviceType::Metal,
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

// Global Singleton for Device Manager
lazy_static! {
    pub static ref DEVICE_MANAGER: Arc<Mutex<DeviceManager>> =
        Arc::new(Mutex::new(DeviceManager::new()));
}

pub fn get_device() -> Device {
    DEVICE_MANAGER.lock().unwrap().device().clone()
}
