# Extern Function Migration Checklist

Goal: remove `extern fn` usage from `.tl` scripts and expose runtime APIs as class/instance methods (except global `print/println/read_line`).

Last updated: 2026-01-21

## Rules / decisions
- [x] Global functions stay global: `print`, `println`, `read_line`.
- [x] `System::memory_mb()` is the canonical API for memory reporting.

## Compiler/runtime work
- [x] Add global builtin `read_line` (no `extern fn` in scripts).
- [x] Allow `tl_*` runtime calls in semantic checking (no extern declarations needed).
- [x] Confirm `fn_return_types` coverage for `tl_*` runtime functions.
- [x] Add method-level semantic fallbacks (static/instance) for built-in types below (optional if wrappers stay).
- [x] Add built-in method mappings (codegen) where needed to avoid wrapper impls.

### Built-in types & methods to support
- [x] Tokenizer
  - [x] `Tokenizer::new(path: String) -> Tokenizer`
  - [x] `tokenizer.encode(prompt: String) -> Tensor<i64, 1>`
  - [x] `tokenizer.decode(ids: Tensor<i64, 1>) -> String`
- [x] Map (GGUF)
  - [x] `Map::load(path: String) -> Map`
  - [x] `map.get(key: String) -> Tensor<f32, 2>`
  - [x] `map.get_1d(key: String) -> Tensor<f32, 1>`
  - [x] `map.get_quantized(key: String) -> i64`
- [x] KVCache
  - [x] `KVCache::new(layers: i64) -> KVCache`
  - [x] `cache.free()`
  - [x] `cache.get_k(layer: i64) -> Tensor<f32, 4>`
  - [x] `cache.get_v(layer: i64) -> Tensor<f32, 4>`
  - [x] `cache.update(layer: i64, k: Tensor<f32, 4>, v: Tensor<f32, 4>)`
- [ ] File / Path / Http
  - [x] `File::exists(path: String) -> bool`
  - [x] `File::read(path: String) -> String`
  - [x] `File::write(path: String, content: String) -> bool`
  - [x] `File::download(url: String, path: String) -> bool`
  - [ ] `File::open(path: String, mode: String) -> File`
  - [ ] `file.read_string() -> String`
  - [ ] `file.write_string(content: String)`
  - [ ] `file.close()`
  - [x] `Path::exists(path: String) -> bool`
- [x] String
  - [x] `String::from_int(i: i64) -> String`
  - [x] `string.concat(other: String) -> String`
  - [x] `string.contains(needle: String) -> bool`
- [x] Tensor ops used by apps/tests
  - [x] `Tensor::embedding`, `Tensor::rms_norm`, `Tensor::matmul`, `Tensor::add`, `Tensor::silu`, `Tensor::mul`
  - [x] `Tensor::scale`, `Tensor::transpose`, `Tensor::transpose_2d`
  - [x] `Tensor::apply_rope`, `Tensor::repeat_interleave`, `Tensor::new_causal_mask`
  - [x] `Tensor::narrow`, `Tensor::cat_4d`, `Tensor::rope_new_cos`, `Tensor::rope_new_sin`
  - [x] `Tensor::argmax`, `Tensor::len`, `Tensor::item_i64`, `Tensor::cat_i64`, `Tensor::sample`
  - [x] `Tensor::matmul_4d`, `Tensor::add_4d`, `Tensor::softmax`
  - [x] `Tensor::matmul_quantized` (QTensor)
  - [x] `Tensor::print_1`, `Tensor::print_2`, `Tensor::clear_grads` (debug)

## Script migration (remove extern declarations)
- [x] `examples/apps/llama3/chatbot_llama3.tl`
- [x] `examples/apps/tinyllama/chatbot.tl`
- [x] `examples/debug/debug_structure.tl`
- [x] `examples/debug/debug_tokenizer.tl`
- [x] `examples/tasks/addition/train_verify_2digit.tl`
- [x] `examples/tasks/heavy/train_heavy.tl`
- [x] `examples/apps/llama3/debug/*` (parser_limit_test_50/100, parser_mixed_*, bug_repro, parser_order_test, chatbot_reordered)
- [x] `examples/apps/llama3/parser_error_repro.tl`
- [x] `tests/debug/*` (test_externs, test_chatbot_*, test_simple_loop, test_crash_bisect, test_leak, manual_test_readline, debug_vocab_size, debug_tokenizer, repro_ensure_tensor, train_checkpoint_verify, test_struct_crash)

## Progress notes
- [x] Initial plan captured
