use crate::compiler::builtin_loader::BuiltinLoader;

pub fn load_net_data() -> (crate::compiler::builtin_loader::BuiltinTypeData, crate::compiler::builtin_loader::BuiltinTypeData) {
    let source = crate::compiler::codegen::builtin_types::assets::BuiltinAssets::get_source("non_generic/net.tl");
    let listener = BuiltinLoader::load_module_data(&source, "TcpListener")
        .expect("Failed to load non-generic builtin TcpListener (net)");
    let stream = BuiltinLoader::load_module_data(&source, "TcpStream")
        .expect("Failed to load non-generic builtin TcpStream (net)");
    (listener, stream)
}
