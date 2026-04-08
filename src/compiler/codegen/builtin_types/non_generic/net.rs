use crate::compiler::builtin_loader::BuiltinLoader;

pub const SOURCE: &str = include_str!("net.tl");

pub fn load_net_data() -> (crate::compiler::builtin_loader::BuiltinTypeData, crate::compiler::builtin_loader::BuiltinTypeData) {
    let listener = BuiltinLoader::load_module_data(SOURCE, "TcpListener")
        .expect("Failed to load non-generic builtin TcpListener (net)");
    let stream = BuiltinLoader::load_module_data(SOURCE, "TcpStream")
        .expect("Failed to load non-generic builtin TcpStream (net)");
    (listener, stream)
}
