use candle_core::Device;
use lazy_static::lazy_static;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub enum DeviceType {
    Cpu,
    Metal,
    Cuda,
}

pub struct DeviceManager {
    current_device: Device,
    device_type: DeviceType,
}

impl DeviceManager {
    pub fn new() -> Self {
        // Priority: CPU -> Metal (Metal disabled for now due to JIT issues)
        println!("Initializing Runtime: CPU backend selected.");
        DeviceManager {
            current_device: Device::Cpu,
            device_type: DeviceType::Cpu,
        }
        /*
        // Check for Metal
        if candle_core::utils::metal_is_available() {
            println!("Initializing Runtime: Metal backend selected.");
            match Device::new_metal(0) {
                Ok(device) => {
                    return DeviceManager {
                        current_device: device,
                        device_type: DeviceType::Metal,
                    }
                }
                Err(e) => eprintln!("Failed to initialize Metal: {}. Falling back to CPU.", e),
            }
        }
        */
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
