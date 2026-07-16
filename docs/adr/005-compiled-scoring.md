# ADR 005: compiled scoring path, chosen by profile, pinned by parity tests

Status: accepted

## Context

Single-row scoring through the fitted `ColumnTransformer` measured ~4-10 ms
p99 on the ULB artifact - hovering at the repo's stated 10 ms p99 budget.
Profiling (cProfile, 300 rows) put roughly two thirds of that inside
pandas/ColumnTransformer machinery building and indexing a 1-row frame, and
only ~0.3 ms in the gradient-boosted model itself. The slow part was
bookkeeping, not math.

## Decision

Compile the fitted preprocessor once at load time: extract the one-hot
category-to-column maps and the numeric column order from `transformers_`,
then build the model's input matrix directly in numpy per request
(`scoring.CompiledScorer`). The model call itself is untouched sklearn.

Three rules made this safe to ship:

1. **Profile first, optimize the top hotspot only.** No speculative caching,
   no reimplemented tree traversal, no second serialization format.
2. **The reference path stays.** `PandasScorer` remains in the repo as the
   implementation of record; tests pin the compiled transform to it exactly
   (`np.array_equal`, NaN-aware, decisions included) across full rows, missing
   fields, and unknown categories.
3. **The gain is guarded.** A latency-regression test fails CI if compiled
   single-row p95 exceeds a loose bound (5 ms; measured ~1 ms; the pandas
   path measured ~4-12 ms), so the optimization cannot silently rot.

Measured result (committed benchmark, real ULB artifact): single-row mean
4.13 -> 1.02 ms, p99 5.50 -> 1.73 ms. Batch throughput barely moved
(605k -> 663k rows/s) because ColumnTransformer amortizes at volume - stated
plainly instead of implying the compiled path speeds everything up.

## Rejected

- **"Just batch the requests":** authorization-time scoring is one transaction
  with a deadline; batching is a throughput tool and exists separately
  (`/predict/batch`).
- **ONNX/treelite compilation:** see ADR 004; the model was never the hotspot.
- **Dropping the pandas path once compiled won:** would delete both the parity
  oracle and the benchmark baseline that justify the compiled path's existence.
