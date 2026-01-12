fn main() {
    print("Testing Device enum...");
    let d: Device = Device::Cpu;
    set_device(d);
    set_device(Device::Auto);
    print("set_device(Device) completed successfully!");
}
