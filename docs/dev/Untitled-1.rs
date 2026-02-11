
trait IDevice<T>;
struct CpuDevice;

trait IDevice<T> {
    fn tensor_add(self, left: OpaqueTensor, right: OpaqueTensor);
}


impl IDevice<f32> for CpuDevice {
    fn tensor_add(self, left: OpaqueTensor, right: OpaqueTensor) {
        todo!()
    }
}


struct MetalDevice;

impl IDevice<f32> for MetalDevice {
    fn tensor_add(self, left: OpaqueTensor, right: OpaqueTensor) {
        todo!()
    }
}
