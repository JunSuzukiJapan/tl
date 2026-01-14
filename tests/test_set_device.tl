fn main() {
    print("Testing Device enum...");
    let d: Device = Device::Cpu;
    Param::set_device(d);
    Param::set_device(Device::Auto);
    print("set_device(Device) completed successfully!");
}
