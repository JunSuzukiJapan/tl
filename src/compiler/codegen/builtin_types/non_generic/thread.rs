use crate::compiler::codegen::type_manager::{CodeGenType, TypeManager};
use crate::compiler::codegen::CodeGenerator;
use crate::compiler::ast::Type;
use inkwell::values::{BasicValueEnum, ValueKind};

pub fn register_thread_types(manager: &mut TypeManager) {
    let mut thread_type = CodeGenType::new("Thread");
    
    // Thread::spawn(closure: Fn() -> i64) -> Thread
    // Since closures are typed as Fn(args) -> ret in AST, here we just use Type::Fn as a placeholder.
    let fn_type = Type::Fn(vec![], Box::new(Type::I64));
    
    thread_type.register_evaluated_static_method(
        "spawn", 
        compile_thread_spawn, 
        vec![fn_type],
        Type::Struct("Thread".to_string(), vec![])
    );
    
    // Thread.join() -> i64
    thread_type.register_evaluated_instance_method(
        "join", 
        compile_thread_join, 
        vec![],
        Type::I64
    );
    
    manager.register_type(thread_type);
}

// ---------------------------------------------------------
// Thread Methods
// ---------------------------------------------------------

pub fn compile_thread_spawn<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    args: Vec<(BasicValueEnum<'ctx>, Type)>,
    _hint: Option<&Type>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let closure_val = args[0].0;
    
    // The closure in TL is compiled as a Fat Pointer: a struct { fn_ptr, env_ptr }
    // which is returned directly as a StructValue by compile_closure.
    let closure_struct = closure_val.into_struct_value();
    
    let fn_ptr = codegen.builder.build_extract_value(closure_struct, 0, "fn_ptr")
        .unwrap().into_pointer_value();
        
    let env_ptr = codegen.builder.build_extract_value(closure_struct, 1, "env_ptr")
        .unwrap().into_pointer_value();

    // Call tl_thread_spawn(fn_ptr, env_ptr)
    let b_fn = codegen.module.get_function("tl_thread_spawn")
        .ok_or("tl_thread_spawn not found")?;
        
    let call = codegen.builder.build_call(b_fn, &[fn_ptr.into(), env_ptr.into()], "thread_id")
        .map_err(|e| e.to_string())?;
        
    let id_val = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from tl_thread_spawn".into()),
    };
    
    // Allocate a Thread struct { i64 } to return
    let t_struct_ty = codegen.context.struct_type(&[codegen.context.i64_type().into()], false);
    let ret_ptr = codegen.builder.build_alloca(t_struct_ty, "thread_struct")
        .map_err(|e| e.to_string())?;
        
    let id_gep = codegen.builder.build_struct_gep(t_struct_ty, ret_ptr, 0, "id_field")
        .map_err(|_| "Failed to GEP")?;
    codegen.builder.build_store(id_gep, id_val)
        .map_err(|e| e.to_string())?;
        
    Ok((ret_ptr.into(), Type::Struct("Thread".to_string(), vec![])))
}

pub fn compile_thread_join<'ctx>(
    codegen: &mut CodeGenerator<'ctx>,
    obj: BasicValueEnum<'ctx>,
    _obj_ty: Type,
    _args: Vec<(BasicValueEnum<'ctx>, Type)>,
) -> Result<(BasicValueEnum<'ctx>, Type), String> {
    let t_struct_ty = codegen.context.struct_type(&[codegen.context.i64_type().into()], false);
    let ptr = try_extract_pointer(obj)?;
    
    let id_gep = codegen.builder.build_struct_gep(t_struct_ty, ptr, 0, "id_field")
        .map_err(|_| "Failed to GEP")?;
    let id_val = codegen.builder.build_load(codegen.context.i64_type(), id_gep, "id_val")
        .map_err(|e| e.to_string())?;
        
    let b_fn = codegen.module.get_function("tl_thread_join")
        .ok_or("tl_thread_join not found")?;
        
    let call = codegen.builder.build_call(b_fn, &[id_val.into()], "join_res")
        .map_err(|e| e.to_string())?;
        
    let res = match call.try_as_basic_value() {
        ValueKind::Basic(v) => v,
        _ => return Err("Invalid return from tl_thread_join".into()),
    };
    
    Ok((res, Type::I64))
}

fn try_extract_pointer<'ctx>(val: BasicValueEnum<'ctx>) -> Result<inkwell::values::PointerValue<'ctx>, String> {
    if val.is_pointer_value() {
        Ok(val.into_pointer_value())
    } else {
        Err(format!("Value is not a pointer: {:?}", val))
    }
}
