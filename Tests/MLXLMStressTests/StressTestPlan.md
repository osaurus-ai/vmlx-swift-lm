# MLXLMStressTests — exhaustive stability matrix

This test target exercises the full cache + modality + workload matrix
that has been escaping us in production. Every cell must pass before a
release pin is bumped.

The matrix is defined by these axes:

| Axis              | Values |
|-------------------|--------|
| Cache mode        | paged, non-paged |
| Disk tier         | enabled, disabled |
| Prefix cache (L1) | hit, miss |
| L2 disk-restore   | cold, warm-restore, warm-restore + Memory.clearCache between |
| KV mode           | fp16, turboQuant(3,3), turboQuant(4,4), turboQuant(8,8) |
| Architecture      | pure-attn (Qwen3), hybrid SSM (Nemotron-3, Qwen3.5/3.6), pure SSM (Mamba2) |
| defaultMaxKVSize  | 1024, 8192, 32768 |
| Prompt length     | short (32), mid (2k), long (16k), over-cap (64k) |

Workloads (each runs across the cell matrix):

1. Single request, baseline (sanity)
2. Two identical back-to-back requests (Bug 1 repro — warm disk hit)
3. Two different prompts sharing a long prefix (prefix-cache hit)
4. 10-request burst, same model, mixed prompts
5. Multi-turn: 8-turn conversation, growing context, KV reuse each turn
6. Cancellation-mid-prefill: kick a request, cancel before first token
7. Cancellation-mid-decode: cancel halfway through generation
8. Concurrent requests on same BatchEngine (slot pressure)
9. Memory.clearCache() mid-request (force the eviction race)
10. Over-cap prompt (Bug 2 repro): prompt > defaultMaxKVSize * longPromptMultiplier

Modality:
- Text-only LLM
- VLM image (single + multi-image)
- VLM video (Nemotron-Omni RADIO)
- Audio STT (Nemotron-Omni Parakeet)
- Multimodal mixed (image + text turn, then audio + text turn)

Pass criteria per cell:
- No notifyExternalReferencesNonZeroOnDealloc
- No metal::malloc fatals
- No fatalError
- Output token-distribution within tolerance of cold-start reference
- Memory delta after teardown <= 5% of peak (no leaks)

Implementation approach:
- Use real models from local HF cache (`~/.cache/huggingface/hub/`)
  with env-var gating so CI can skip when models are absent.
- Drive via `BatchEngine` directly (no HTTP layer involved). The
  existing `Tests/MLXLMTests/BatchEngineTests.swift` shows the
  setup.
- For each cell, capture: pass/fail, peak RSS, peak Metal alloc,
  duration, and a list of any caught crashes.
- Output: `STRESS_REPORT.md` with the pass/fail grid + diagnostics.
