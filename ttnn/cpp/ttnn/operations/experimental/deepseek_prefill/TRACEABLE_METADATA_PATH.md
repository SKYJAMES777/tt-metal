# Trace-safe metadata path for the chunked-prefill MLA ops

## Why

The Kimi/DeepSeek chunked-prefill `transformer.forward()` is captured as a ttnn trace to collapse the
per-op host-dispatch (op2op) gaps. A trace records the device command stream **once** and replays it; any
op argument baked into the program at capture is frozen, so an op that takes a **per-chunk scalar**
(e.g. the running KV length) cannot be replayed across chunks — the captured value would be wrong for
every chunk but the one captured.

Fix: move each per-chunk scalar out of the host dispatch path and into a small **metadata DRAM tensor**
that the op reads **on-device**. The program no longer depends on the per-chunk value, so one captured
trace replays across all chunks. The metadata tensor is the runner's `h2d_socket_sync` payload:

```
metadata: uint32 DRAM tensor, replicated across the mesh, canonical layout
          [slot_id, actual_start, actual_end]      (3 words, 12 bytes)
            index 0 = slot_id        (= cache_user_id)
            index 1 = actual_start   (= kv_actual_isl, the prior valid KV length, tokens)
            index 2 = actual_end     (pad-zero boundary)
```

Produced by `ttnn.H2DStreamService` + `ttnn.experimental.deepseek_prefill.inbound_socket_service_sync`
(see `models/demos/deepseek_v3_d_p/tt/runners/prefill_runner.py`,
`tt/runners/runner_utils.py::build_h2d_service`).

## Dual signature (per op)

Every converted op keeps its **original scalar signature** and gains a second **metadata** overload,
selected by whether a `metadata` tensor is passed. Both produce identical device results. The scalar
form stays the default for existing callers (e.g. `tt/mla/mla.py`); the metadata form is opt-in and is
the trace-safe one.

## Per-op status

| Op | Per-chunk value(s) → metadata index | Consumer kernel | Status |
|----|--------------------------------------|-----------------|--------|
| `update_padded_kv_cache` | `slot_idx`→0, `kv_actual_global`→1 | writer (dataflow) | **done** |
| `rotary_embedding_indexed` (Q + KV rope) | `kv_actual_global`→1 | reader (dataflow) | **done** |
| `zero_padded_kv_cache` | `slot_idx`→0, `valid_global`(=actual_end)→2 | reader + writer (dataflow) | **done** |
| `ring_mla` | `kv_actual_isl`, `logical_n`, `kv_cache_batch_idx` | reader + **compute** | **TODO** (separate plan) |

`ring_mla` is harder: `kv_actual_isl` drives host-side ring-iteration masks / Q-mapping / valid-page
counts baked into runtime args, and its **compute** kernel needs derived values — needs its own design.

## Implementation pattern (shared by all three done ops)

Host (`device/<op>_device_operation.{hpp,cpp}`):
- `tensor_args_t.metadata` is `std::optional<Tensor>`; `operation_attributes_t` keeps the scalar(s),
  used only on the scalar path (0 on the metadata path).
- `compute_program_hash` includes `metadata.has_value()` so the scalar and metadata programs never
  collide. The per-chunk **values** are never hashed on either path (one cached program per layer).
- `create_descriptor` adds a `has_metadata` compile-time flag to the consumer kernel(s), appends a
  metadata `TensorAccessorArgs` (only when present), allocates a small L1 metadata-scratch CB, and puts
  the metadata tensor's raw DRAM address (else the scalar) in a common runtime arg.
- `MeshWorkloadFactory::override_runtime_arguments` patches that common arg on cache hits — the metadata
  address (metadata path) or the scalar(s) (scalar path) — since the buffer-binding fast path leaves
  raw-address/scalar common args stale otherwise.
- Two public overloads (top-level `.cpp`) + two nanobind `ttnn::overload_t` overloads, disambiguated by
  the differing positional arg type (`int` scalar vs `Tensor` metadata).

Consumer kernel (dataflow):
- Body is `template <bool HasMeta>` called from `kernel_main()` as `run_x<get_compile_time_arg_val(flag)>()`.
  This is required: `if constexpr` inside the non-template `kernel_main` would still **instantiate** the
  discarded branch's non-dependent templates and fail to compile. Inside the template, the metadata
  branch is genuinely discarded for the scalar program.
- The metadata `TensorAccessorArgs<offset>` offset is made **dependent on `HasMeta`**
  (`HasMeta ? cache_args.next_compile_time_args_offset() : 0`) so the scalar program never names an
  out-of-range compile-time-arg index (a fixed offset there static-asserts "Index out of range").
- Metadata read: `noc.async_read(meta_accessor, meta_cb, N, {.page_id=0})` (N = 8 or 12 B — only the
  needed indices), then `CoreLocalMem<volatile uint32_t>` to extract the fields.

### `zero_padded_kv_cache` — note the compute kernel

This op has a **compute** kernel (masks the partial boundary tile), and compute kernels cannot NoC-read
DRAM. Rather than a reader→compute control handoff (the `unified_routed_expert_ffn` UNPACK→mailbox
pattern), the kernels use an **unconditional CB protocol**: the reader always pushes `src`+`mask`, the
compute always multiplies exactly `Wt` tiles (`Wt` is a structural common arg readable by all three
compute threads), and the writer always pops the `out` tiles but only writes them back on the chip that
owns the partial (discarding them otherwise). The compute kernel is therefore path-agnostic and needs no
per-chunk value at all — no control CB, no mailbox, no multi-thread coordination. Only the dataflow
reader/writer read the metadata.

## Tests

Each op has an H2D-service equivalence test that proves **metadata path == scalar path, bit-exact**,
driving the metadata from a **real** `H2DStreamService` + `inbound_socket_service_sync` (not a hand-built
tensor), on an 8×4 mesh with `fabric_config=FABRIC_2D`:
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_update_padded_kv_cache.py::test_update_padded_kv_cache_metadata_matches_scalar`
- `models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_deepseek_prefill_rotary_embedding_indexed.py::test_rotary_embedding_indexed_metadata_matches_scalar`
- `models/demos/deepseek_v3_d_p/tests/test_zero_padded_kv_cache.py::test_zero_padded_kv_cache_metadata_matches_scalar`

The existing scalar-path op tests are the regression that the scalar path is unchanged.

## Verified (8×4 Blackhole, Kimi K2.6)

- update_padded_kv_cache: equivalence 3/3 dtypes bit-exact; op regression 6 passed.
- rotary_embedding_indexed: equivalence 2/2 offsets bit-exact; scalar regression 4 passed.
- zero_padded_kv_cache: equivalence 4/4 (740 / 2600 / 4512 windows, slot 0/1) bit-exact; scalar
  regression 10 passed.
- End-to-end `test_kimi_prefill_transformer_chunked_trace_kv_pcc` (L10, model's scalar callers): passed,
  min KV-cache PCC = 0.993906.

## Gotcha

A kernel JIT-compile failure segfaults the test process and can wedge an active ethernet core
("Timed out while waiting for active ethernet core … become active again"). Recover with
`tt-smi -glx_reset` before retrying.
