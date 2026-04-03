# mlx-swift-lm Fork -- Complete Audit

> Comprehensive audit of all changes: osaurus-ai/mlx-swift-lm vs upstream ml-explore/mlx-swift-lm.
> 22 files changed, +2,639 lines across 16 commits.
> Final audit date: 2026-04-03

---

## Commit History

| Commit | Description |
|--------|-------------|
| 5d5b2dc | feat: Gemma 4 text + JANG loader (squashed) |
| 6c5b8fd | feat: Gemma 4 VLM (vision encoder + processor) |
| e8223fd | fix: vision .linear. key strip + pooler mask |
| 496fa2f | fix: per-layer quant key remapping for VLM |
| f841d5d | fix: Qwen3.5 VLM sanitize for JANG |
| 7089a0d | feat: Mistral Small 4 (MLA + MoE) |
| 00ac224 | fix: JANG per-layer quant + Nemotron key remap |
| 6bdf041 | fix: dequantize MoE gate weights |
| c50052e | perf: bfloat16 conversion for MoE |
| c6e81f9 | feat: Mistral4 VLM + rope_interleave fix |
| 802187e | perf: compiled forward (reverted in dd45df3) |
| 2c8d66a | perf: compiled GLU activations |
| dd45df3 | perf: revert compiled forward, keep GLU |
| 7c82d21 | Add speculative decoding (upstream cherry-pick) |
| 50f1341 | fix: per-tensor group_size inference |
| ce9ca10 | feat: expose isVLM on ModelContext/ModelContainer |
| 0f8361d | fix: Gemma 4 tool call format |

---

## Net File Changes

| File | Change | Purpose |
|------|--------|---------|
| MLXLLM/Models/Gemma4Text.swift | NEW 640L | Gemma 4 text: MoE + Dense, dual attention, v_norm, K=V |
| MLXVLM/Models/Gemma4.swift | NEW 784L | Gemma 4 VLM: 2D RoPE vision, pooler, processor |
| MLXLLM/Models/Mistral4.swift | NEW 483L | Mistral 4: MLA attention, 128-expert MoE + shared, YaRN |
| MLXVLM/Models/Mistral4VLM.swift | NEW 509L | Mistral 4 VLM: MLA text + Pixtral vision |
| MLXLMCommon/JangLoader.swift | EXTENDED +132L | Gate dequant, per-tensor group_size inference |
| MLXLMCommon/Load.swift | MODIFIED +60L | JANG pipeline, VLM key remap, bfloat16 conversion |
| MLXLMCommon/SwitchLayers.swift | MODIFIED +32L | Compiled SwiGLU/GeGLU kernels |
| MLXLMCommon/Evaluate.swift | MODIFIED +391L | Speculative decoding (upstream cherry-pick) |
| MLXLMCommon/KVCache.swift | MODIFIED +1L | trimPromptCache fix (all layers) |
| MLXLMCommon/LanguageModel.swift | MODIFIED +6L | VisionLanguageModelProtocol marker |
| MLXLMCommon/ModelFactory.swift | MODIFIED +3L | ModelContext.isVLM |
| MLXLMCommon/ModelContainer.swift | MODIFIED +7L | ModelContainer.isVLM |
| MLXLMCommon/Tool/ToolCallFormat.swift | MODIFIED +13L | .gemma4 tool call format |
| MLXLMCommon/Tool/Parsers/GemmaFunctionParser.swift | MODIFIED +16L | Configurable tags for Gemma 3/4 |
| MLXLLM/LLMModelFactory.swift | MODIFIED +63L | gemma4, mistral4 registrations |
| MLXLLM/Models/NemotronH.swift | MODIFIED +20L | JANG key remap |
| MLXVLM/Models/Qwen35.swift | MODIFIED +24L | JANG VLM sanitize |
| MLXVLM/Models/Mistral3.swift | MODIFIED +6L | Internal visibility for Mistral4 reuse |
| MLXVLM/VLMModel.swift | MODIFIED +1L | Extends VisionLanguageModelProtocol |
| MLXVLM/VLMModelFactory.swift | MODIFIED +2L | gemma4 VLM registration |
| Tests/SpeculativeDecodingTests.swift | NEW 84L | Speculative decoding tests |

---

## Weight Loading Pipeline (Load.swift)

Verified execution order for all model types:

```
1. Load safetensors files
2. model.sanitize(weights, metadata)      -- per-model key remapping
3. JangLoader.dequantizeMoEGates()        -- JANG only: gate bf16 passthrough
4. Determine per-layer quantization       -- JANG inferred OR config.json
5. VLM key prefix remapping               -- non-JANG VLM only
6. quantize(model)                        -- Linear -> QuantizedLinear
7. model.update(parameters, verify: .all) -- strict weight loading
8. convertToBFloat16(model)               -- MoE only: prevents Metal promotion
9. finalize
```

**Gate dequant before quantize is load-bearing**: dequant removes .scales keys so quantize() keeps gates as float Linear instead of QuantizedLinear.

**bfloat16 after update is safe**: only affects float16/float32 params, skips uint32 quantized weights and .scales/.biases.

---

## Model Routing

### Dispatch Order

`ModelFactoryRegistry` tries VLM factory first, then LLM. For dual-registered types (gemma4, qwen3_5, qwen3_5_moe), VLM wins via the generic loadModel() path.

### Complete Table

| model_type | LLM Class | VLM Class | Processor | Notes |
|------------|-----------|-----------|-----------|-------|
| gemma4 | Gemma4TextModel | Gemma4 | Gemma4Processor | Dual-registered; VLM first |
| gemma4_text | Gemma4TextModel | -- | -- | LLM only |
| mistral3 (text=mistral4) | Mistral4Model | Mistral3VLM | Mistral3Processor | Custom closure sniffs text_config |
| mistral4 | Mistral4Model | -- | -- | LLM only |
| nemotron_h | NemotronHModel | -- | -- | LLM only |
| minimax / minimax_m2 | MiniMaxModel | -- | -- | LLM only |
| qwen3_5 | Qwen35Model | Qwen35 | Qwen3VLProcessor | Dual-registered |
| qwen3_5_moe | Qwen35MoEModel | Qwen35MoE | Qwen3VLProcessor | Dual-registered |
| qwen3_5_text | Qwen35TextModel | -- | -- | LLM only |

### isVLM API

- `VisionLanguageModelProtocol` marker in MLXLMCommon
- `VLMModel` extends it (in MLXVLM)
- `ModelContext.isVLM` / `ModelContainer.isVLM` checks `model is VisionLanguageModelProtocol`
- All VLM models return true, all LLM models return false

---

## Performance Optimizations

| Optimization | Status | Impact |
|-------------|--------|--------|
| bfloat16 MoE conversion | ACTIVE | +38% Gemma4, prevents Metal float16->float32 promotion |
| Compiled SwiGLU/GeGLU | ACTIVE | +5%, fuses gate activation into single Metal dispatch |
| Compiled forward pass | REVERTED | Crashed on dynamic-shape MoE ops (argPartition, Split) |
| Stream-per-call | REVERTED | Overhead destroyed performance |
| Speculative decoding | ACTIVE (upstream) | +29-79% with draft model |

### Results

| Model | Before | After | Gain |
|-------|-------:|------:|-----:|
| Qwen3.5-35B MoE | 42.4 tok/s | 58.7 tok/s | +38% |
| Gemma4 26B MoE | 25.0 tok/s | 43.8 tok/s | +75% |
| Qwen3.5-4B Dense | 123 tok/s | 145 tok/s | +18% |

---

## Tool Call Formats

| Format | Start Tag | End Tag | Escape | Inferred From |
|--------|-----------|---------|--------|---------------|
| .gemma (Gemma 3) | `<start_function_call>` | `<end_function_call>` | `<escape>` | model_type == "gemma" |
| .gemma4 (Gemma 4) | `<\|tool_call>` | `<tool_call\|>` | `<\|"\|>` | model_type starts with "gemma4" |

Both use the same `GemmaFunctionParser` with configurable tags.

---

## Speculative Decoding (Upstream Cherry-Pick)

- `SpeculativeTokenIterator`: draft model proposes tokens, main model verifies in batch
- `TokenIteratorProtocol`: shared interface for normal and speculative iterators
- `trimPromptCache` bugfix: now trims ALL cache layers (was only first)
- Works with KVCacheSimple; NOT compatible with RotatingKVCache after wrap
- Tests in SpeculativeDecodingTests.swift

---

## Known Issues (Ranked)

### MEDIUM

1. **Mistral4VLM config extraction is dead code** (Mistral4VLM.swift:120-131)
   - AnyCodable is empty, decode always fails; all MLA fields default to nil
   - Works only because defaults match Mistral Small 4 dimensions

2. **Mistral4VLM not registered in VLM factory**
   - "mistral3" VLM routes to Mistral3VLM (no MLA text decoder)
   - Text-only via LLM factory correctly uses Mistral4Model

3. **Dual-registration VLM-first dispatch**
   - gemma4, qwen3_5, qwen3_5_moe are in both factories
   - Generic loadModel() always tries VLM first
   - Text-only models (no processor_config.json) may fail VLM load before falling through to LLM

### LOW

4. **JangLoader misleading comment** (JangLoader.swift:357)
   - Says Gemma4 router.proj is "already handled separately" but there is no separate handling
   - Router stays quantized (QuantizedLinear), which is intentional but comment is misleading

5. **SwitchGLU activation detection forces GPU sync at init** (SwitchLayers.swift:69-72)
   - .item(Bool.self) synchronizes CPU<-GPU once per MoE layer at model load

6. **Speculative decoding: no asyncEval pipelining between rounds**
   - Main model verify pass has no async prefetch, slightly lower throughput vs TokenIterator

7. **Mixed group_size ambiguity in JANG inference**
   - If actual group_size differs from config blockSize AND wrong (bits, gs) pair passes round-trip check, inference is silently wrong
   - Only affects hypothetical models with mixed group sizes within the same JANG checkpoint
   - All current JANG models use uniform group_size per checkpoint

---

## Upstream Unchanged

All 50+ existing LLM architectures, 15+ VLM architectures, tokenizer/downloader integration, LoRA/fine-tuning, ChatSession, MLXEmbedders, and all KVCache implementations are untouched. Evaluate.swift is restored to upstream state (compiled forward reverted). The only upstream addition is the speculative decoding cherry-pick.
