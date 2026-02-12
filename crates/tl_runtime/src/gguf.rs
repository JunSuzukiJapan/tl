//! GGUF Loader Module
//! Loads GGUF files and creates OpaqueTensorMap populated with QTensor.

use crate::string_ffi::StringStruct;
use crate::tensor_map::OpaqueTensorMap;
use crate::quantized::{QTensor, GGMLType};
use std::collections::HashMap;
use std::sync::Arc;
use std::ffi::CStr;
use std::fs::File;
use memmap2::Mmap;

// Import gguf-rs types
use gguf_rs::{ByteOrder, GGUFContainer, GGMLType as RsGgmlType};
use std::cell::RefCell;
use std::rc::Rc;
use std::io::Read;

struct ByteCounter<R> {
    inner: R,
    count: Rc<RefCell<u64>>,
}

impl<R: Read> Read for ByteCounter<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        *self.count.borrow_mut() += n as u64;
        Ok(n)
    }
}

/// GGUF Load Function
#[unsafe(no_mangle)]
pub extern "C" fn tl_gguf_load(path: *mut StringStruct) -> *mut OpaqueTensorMap {
    unsafe {
        if path.is_null() || (*path).ptr.is_null() {
             return std::ptr::null_mut();
        }
        let path_str = CStr::from_ptr((*path).ptr).to_string_lossy();
        let expanded = crate::file_io::expand_path(&path_str);

        // Open file and memory map it
        let file = match File::open(&expanded) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Error: Failed to open GGUF file {:?}: {}", expanded, e);
                return std::ptr::null_mut();
            }
        };
        let mmap = match Mmap::map(&file) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error: Failed to mmap GGUF file: {}", e);
                return std::ptr::null_mut();
            }
        };

        // Determine ByteOrder from magic
        if mmap.len() < 4 {
             eprintln!("Error: File too small");
             return std::ptr::null_mut();
        }
        let magic = &mmap[0..4];
        let byte_order = if magic == b"GGUF" {
             ByteOrder::LE
        } else if magic == b"FUGG" {
             ByteOrder::BE
        } else {
             eprintln!("Error: Invalid GGUF magic");
             return std::ptr::null_mut();
        };

        // Prepare reader wrapper to track offset
        // UNSAFE: Extend lifetime of slice to 'static to satisfy GGUFContainer trait object requirement.
        // This is safe because `container` (which holds the reader) is dropped before `mmap`.
        let static_slice: &'static [u8] = std::mem::transmute(&mmap[..]);
        let mut cursor = std::io::Cursor::new(static_slice);
        // Skip magic
        cursor.set_position(4);
        
        let read_count = Rc::new(RefCell::new(4u64));
        let counter_reader = ByteCounter {
            inner: cursor,
            count: read_count.clone(),
        };

        // Parse GGUF
        // max_array_size is arbitrary, setting to a large value or default
        let mut container = GGUFContainer::new(byte_order, Box::new(counter_reader), u64::MAX);
        
        let model = match container.decode() {
            Ok(m) => m,
            Err(e) => {
                eprintln!("Error: Failed to decode GGUF: {:?}", e);
                return std::ptr::null_mut();
            }
        };
        
        // Calculate data offset
        // GGUF spec: header + metadata + padding(alignment) -> data
        // current read_count is at the end of metadata (tensors info)
        let metadata_end = *read_count.borrow();
        
        // Get alignment
        let alignment = model.metadata().get("general.alignment")
            .and_then(|v| v.as_u64()) // gguf-rs uses serde_json::Value which has as_u64
            .unwrap_or(32); // Default alignment 32

        // Calculate padding
        // Data offset must be multiple of alignment
        let padding = (alignment - (metadata_end % alignment)) % alignment;
        let data_offset = metadata_end + padding;

        eprintln!("[tl_gguf_load] Meta end: {}, Align: {}, Data offset: {}", metadata_end, alignment, data_offset);

        let entries = HashMap::new();
        let mut qtensors = HashMap::new();

        for tensor in model.tensors() {
             let name = tensor.name.clone();
             let mut shape: Vec<usize> = tensor.shape.iter().map(|&d| d as usize).collect();
             // GGUFフォーマットは末尾を1でパディングするため、trim
             while shape.len() > 1 && shape.last() == Some(&1) {
                 shape.pop();
             }
             // GGML は列優先 (column-major) のshapeを使用するため、
             // 行優先 (row-major) のテンソルシステムに合わせて反転
             shape.reverse();
             // Tensor offset is relative to data_offset
             let offset = data_offset + tensor.offset; 
             let offset_usize = offset as usize;
             
             // Convert GGMLType
             // gguf-rs u32 kind -> GGMLType conversion is done internally by gguf-rs into enum?
             // No, tensor.kind is u32 in struct definition.
             // We need to convert u32 to our GGMLType.
             // Wait, struct Tensor definition had pub kind: u32.
             // But we saw RsGgmlType enum.
             // We can try RsGgmlType::try_from(tensor.kind).
             
             let rs_type = match RsGgmlType::try_from(tensor.kind) {
                 Ok(t) => t,
                 Err(_) => {
                     eprintln!("Warning: Unknown tensor type {} for {}", tensor.kind, name);
                     continue;
                 }
             };

             let ggml_type = match rs_type {
                 RsGgmlType::F32 => GGMLType::F32,
                 RsGgmlType::F16 => GGMLType::F16,
                 RsGgmlType::Q4_0 => GGMLType::Q4_0,
                 RsGgmlType::Q4_1 => GGMLType::Q4_1,
                 RsGgmlType::Q5_0 => GGMLType::Q5_0,
                 RsGgmlType::Q5_1 => GGMLType::Q5_1,
                 RsGgmlType::Q8_0 => GGMLType::Q8_0,
                 RsGgmlType::Q8_1 => GGMLType::Q8_1,
                 RsGgmlType::Q4_K => GGMLType::Q4_K,
                 RsGgmlType::Q6_K => GGMLType::Q6_K,
                 _ => GGMLType::Unknown,
             };
             
             if ggml_type == GGMLType::Unknown {
                 eprintln!("[tl_gguf_load] Skipping tensor '{}': unsupported type kind={} ({:?})", name, tensor.kind, rs_type);
                 continue;
             }
             
             // Calculate size for bounds check
             // This is rough estimation or use tensor.size if trusted
             // tensor.size is available.
             
             if offset_usize + (tensor.size as usize) > mmap.len() {
                 eprintln!("Error: Tensor {} data out of bounds", name);
                 continue;
             }
             
             let data = mmap[offset_usize..offset_usize + (tensor.size as usize)].to_vec();
             qtensors.insert(name, Arc::new(QTensor::new(data, shape, ggml_type)));
        }

        eprintln!("[tl_gguf_load] Loaded {} quantized tensors", qtensors.len());

        Box::into_raw(Box::new(OpaqueTensorMap {
            entries,
            qtensors, 
        }))
    }
}
