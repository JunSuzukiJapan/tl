use tl::runtime::cpu_device;  // runtime側で、cpuとmetalの実装を区別できるようにエクスポートする。
use tl::runtime::metal_device;

trait IDevice {
    fn tensor_add(self, left: OpaqueTensor, right: OpaqueTensor);
}

trait ITensor<T> {
    fn tensor_add(self, left: T, right: T) -> T;
}

struct CpuDevice<T: ITensor>;
struct MetalDevice<T: ITensor>;

impl<T: ITensor> IDevice for CpuDevice<T> {
    #[inline]
    fn tensor_add(self, left: T, right: T) -> T {
        cpu_device::tensor_add(left, right);
    }
}

impl<T: ITensor> IDevice for MetalDevice<T> {
    #[inline]
    fn tensor_add(self, left: T, right: T) -> T {
        metal_device::tensor_add(left, right);
    }
}
