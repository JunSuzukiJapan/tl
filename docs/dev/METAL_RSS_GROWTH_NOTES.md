# Metal RSS Growth Investigation (N-Queens)

## Summary
When running `examples/readme_n_queens.tl` on Metal, the process RSS grows steadily (multi‑GB),
even though:
- TL runtime refcounts/scopes are stable.
- TL tensor pool stays small.
- Candle Metal buffer pool stays small (bytes ~100KB).
- Forcing `wait_until_completed()` and dropping unused buffers does not stop growth.

CPU runs remain stable (tens of MB).

Conclusion: RSS growth appears to be driven by Metal driver/OS behavior rather than TL memory
management or Candle buffer pooling.

## Repro

### Baseline (Metal)
```
TL_DEVICE=metal cargo run -- examples/readme_n_queens.tl
```
RSS increases over time (multi‑GB).

### With mem trace (per‑statement)
```
TL_DEVICE=metal TL_MEM_TRACE=1 cargo run -- examples/readme_n_queens.tl > /tmp/mem_trace.log 2>&1
```
RSS still increases. The largest deltas are correlated with the training loop, but disabling
`total_loss.item()` and the `current_queens` check did **not** stop growth.

### CPU (stable)
```
TL_DEVICE=cpu cargo run -- examples/readme_n_queens.tl
```
RSS stays around ~40–50MB.

## Instrumentation Added

### Runtime / Metal
- `System::metal_pool_bytes()`, `System::metal_pool_mb()`, `System::metal_pool_count()`
  - Uses Candle Metal `buffer_pool_stats` to read internal pool sizes.
- `System::metal_sync()`
  - Calls `wait_until_completed()` and `drop_unused_buffers()` on Metal device.
- `TL_MEM_TRACE=1`
  - Emits one line per statement:
    ```
    [TL_MEM_TRACE] file:line:col tag=StmtKind rss_mb=... metal_pool_bytes=... metal_pool_count=... pool_count=... refcount_count=... scope_depth=...
    ```

### Candle Metal (vendor)
- `MetalDevice::buffer_pool_stats()` added.
- `MetalDevice::drop_unused_buffers()` made public.
- Logging hooks (optional) already present for `TL_MEM_LOG`.

## Observations

### Metal RSS growth
Example from `/tmp/mem_trace.log` (Metal + mem trace):
```
Memory: 319 MB (metal_pool_mb=0, metal_pool_count=70, metal_pool_bytes=55104)
Memory: 5812 MB (metal_pool_mb=0, metal_pool_count=312, metal_pool_bytes=98776)
Memory: 7600 MB (metal_pool_mb=0, metal_pool_count=356, metal_pool_bytes=106472)
```
Metal pool bytes remain roughly constant while RSS grows.

### TL memory manager health
From TL internal logs:
- `register_tensor` count == `free_tensor` count
- `refcount_count` stable
- `pool_count` stable

### Statement-level deltas
The per‑statement trace shows increases during the training loop, but **disabling**
`total_loss.item()` and the `current_queens` check still resulted in RSS growth.

## Hypothesis
Metal driver/OS internal caches or GPU resource tracking inflate RSS over time. This is not
visible via Candle’s Metal buffer pool or TL’s memory manager.

## Mitigations
- Prefer CPU for long‑running loops: `TL_DEVICE=cpu`
- If Metal is required, consider process restarts for long sessions.

## Related external reports (summarized)
There are reports of Metal allocations not returning to the OS and RSS growing even when
app‑level allocations are stable. This aligns with our observations.

## Files Modified in Investigation
- `crates/tl_runtime/src/lib.rs`
- `crates/tl_runtime/src/device.rs`
- `vendor/candle-core-0.9.1/src/metal_backend/device.rs`
- `src/compiler/codegen/builtins.rs`
- `src/compiler/codegen/expr.rs`
- `src/compiler/codegen/mod.rs`
- `src/compiler/codegen/stmt.rs`
- `src/compiler/semantics.rs`
- `examples/readme_n_queens.tl`

