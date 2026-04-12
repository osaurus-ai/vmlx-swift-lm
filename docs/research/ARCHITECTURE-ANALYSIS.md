# Achieving Execution Parity: MLX-Swift vs. Python Operator Fusion

Source: Detailed architectural analysis provided 2026-04-11.
Core thesis: 2.25x performance degradation is fusion interference from Swift state capture mechanics.

## The Math (confirmed by our benchmarks)

| Metric | Python | Swift |
|---|---|---|
| Graph build | 1.73ms | 1.66ms |
| CPU dispatch overhead | 0.80ms | 13.30ms |
| Raw GPU compute | 7.47ms | 7.54ms |
| **Total step** | **10.0ms** | **22.5ms** |

2600 unfused dispatches x 4us = 10.4ms = the measured gap.

## Four Required Fixes (priority order)

### 1. Refactor Swift State Capture for Graph Purity
Audit Transforms+Compile.swift — the copyContext/updateInternal/tracer cycle
injects identity/slice/reshape nodes that act as fusion barriers. Must use raw
C++ pointers or opaque handles to bypass Swift array abstractions during tracing.

### 2. Audit Shapeless Dimension Constraints
Swift Int offsets captured in compile closures bake constants into MSL code.
KV cache offsets, RoPE offsets, attention masks MUST be MLXArray (symbolic)
not Swift Int (concrete) inside compiled closures.

### 3. 1:1 CustomKernel FFI Mapping
Verify all 30 Python CustomKernel ops have Swift equivalents being called.
Especially: gated_delta kernel, RoPE, quantized matmul variants.

### 4. Consolidate Evaluation Barriers
Ensure zero implicit eval/sync between token embed and final logit sampling.
No shape checks, no .item() calls, no print statements in the hot path.

See full analysis text in conversation history (too large for file).
