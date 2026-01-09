use candle_core::Device;
use lazy_static::lazy_static;
use log::{error, info, warn};
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
    generation: usize,
}

impl DeviceManager {
    pub fn new() -> Self {
        let requested_device = std::env::var("TL_DEVICE").unwrap_or_else(|_| "auto".to_string());
        let (device, device_type) = Self::init_device(&requested_device);

        DeviceManager {
            current_device: device,
            device_type,
            generation: 1,
        }
    }

    fn init_device(requested_device: &str) -> (Device, DeviceType) {
        // Explicit requests should never silently fall back.
        if requested_device == "cuda" {
            #[cfg(feature = "cuda")]
            {
                info!("Initializing Runtime: CUDA backend selected.");
                match Device::new_cuda(0) {
                    Ok(device) => return (device, DeviceType::Cuda),
                    Err(e) => panic!("CUDA requested but initialization failed: {}", e),
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                panic!("CUDA requested but 'cuda' feature not enabled.");
            }
        }

        if requested_device == "metal" {
            #[cfg(feature = "metal")]
            {
                info!("Initializing Runtime: Metal backend selected.");
                match Device::new_metal(0) {
                    Ok(device) => {
                        if check_metal_health(&device) {
                            return (device, DeviceType::Metal);
                        }
                        panic!("Metal backend failed self-test.");
                    }
                    Err(e) => panic!("Metal requested but initialization failed: {}", e),
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                panic!("Metal requested but 'metal' feature not enabled.");
            }
        }

        // Auto: Priority CUDA -> Metal -> CPU.
        if requested_device == "auto" && candle_core::utils::cuda_is_available() {
            #[cfg(feature = "cuda")]
            {
                info!("Initializing Runtime: CUDA backend selected.");
                if let Ok(device) = Device::new_cuda(0) {
                    return (device, DeviceType::Cuda);
                }
            }
        }

        if requested_device == "auto" && candle_core::utils::metal_is_available() {
            #[cfg(feature = "metal")]
            {
                info!("Initializing Runtime: Metal backend selected.");
                if let Ok(device) = Device::new_metal(0) {
                    if check_metal_health(&device) {
                        return (device, DeviceType::Metal);
                    }
                }
            }
        }

        info!("Initializing Runtime: CPU backend selected.");
        (Device::Cpu, DeviceType::Cpu)
    }

    pub fn set_device(&mut self, name: &str) {
        let (device, device_type) = Self::init_device(name);
        self.current_device = device;
        self.device_type = device_type;
        self.generation += 1;
        GLOBAL_DEVICE_GENERATION.store(self.generation, std::sync::atomic::Ordering::Relaxed);
        info!("Device switched to: {:?}", self.device_type);
    }

    pub fn device(&self) -> &Device {
        &self.current_device
    }

    pub fn generation(&self) -> usize {
        self.generation
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

// Atomic generation counter exposed for fast checks
pub static GLOBAL_DEVICE_GENERATION: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(1);

// Thread-local cached device to avoid repeated Mutex locks
thread_local! {
    // Cache stores (Device, Generation)
    static CACHED_DEVICE: std::cell::RefCell<Option<(Device, usize)>> = const { std::cell::RefCell::new(None) };
}

pub fn get_device() -> Device {
    CACHED_DEVICE.with(|cache| {
        let mut cache_ref = cache.borrow_mut();
        let global_gen = GLOBAL_DEVICE_GENERATION.load(std::sync::atomic::Ordering::Relaxed);

        let needs_update = if let Some((_, gen)) = *cache_ref {
            gen != global_gen
        } else {
            true
        };

        if needs_update {
            let manager = DEVICE_MANAGER.lock().unwrap();
            *cache_ref = Some((manager.device().clone(), manager.generation));
        }

        cache_ref.as_ref().unwrap().0.clone()
    })
}
