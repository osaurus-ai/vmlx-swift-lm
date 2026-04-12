# Regression Test Checklist

**MUST be verified on every build before merge/release.**

---

## 1. Float32 Type Promotion (AsType Cascade)

**Issue:** MLXArray(Float) without dtype creates float32 scalars that cause AsType cascade when multiplied with bfloat16 tensors. Results in 2-3x slowdown.

**Root cause:** Swift defaults to float32 for scalar MLXArray. Python infers dtype from context.

**Fix:** Always use `MLXArray(value, dtype: tensor.dtype)` for scalars that interact with model tensors.

**Files affected:** All model files that create MLXArray scalars.

**Test:**
```bash
cd TestRunner && swift build -c release
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench
# MUST show: AsType < 100, tok/s > 90
```

**Regression indicator:** AsType count > 200 in graph node dump = FAIL.

---

## 2. Gemma4 26B Speed Regression

**Issue:** Gemma4 26B 4-bit previously measured at 97-102 tok/s, now shows 26-28 tok/s.

**Possible causes:**
- Chat template not applied → garbage tokens → different perf characteristics
- RotatingKVCache overhead (compile skips RotatingKVCache)
- 32 remaining AsType ops in Gemma4Text.swift
- Branch-specific code changes regressing Gemma4

**Status:** OPEN — needs investigation.

**Test:**
```bash
.build/release/TestRunner <gemma4-26b-path> --raw-bench
# TARGET: > 80 tok/s
# CURRENT: 26-28 tok/s = FAILING
```

---

## 3. JANG Layer Detection / Loading Errors

**Issue:** JANG quantized models may fail to load or produce incorrect output when layer bit-width detection fails.

**Root cause:** JANG mixed-precision models store weights at different bit widths per layer. The loader must correctly detect per-layer bit widths via `bitWidthsUsed` disambiguation.

**Fix:** Commits 10d547e (strict round-trip, bitWidthsUsed disambiguation).

**Files affected:** Libraries/MLXLLM/JangLoader.swift

**Test:**
```bash
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-JANG_2S-TEXT "Hello" 
# MUST: load without crash, generate tokens
```

**Regression indicator:** Crash on load or "shape mismatch" error = FAIL.

---

## 4. JANG Models Not Getting bfloat16 Conversion

**Issue:** Qwen3.5 JANG_2S shows 1,105 AsType (vs 60 for MLX 4-bit). The JANG weight loader may store some parameters as float16 instead of bfloat16, bypassing the convertToBFloat16 fix.

**Root cause:** convertToBFloat16 converts float16/float32 → bfloat16. But JANG weights might have different dtype patterns that the converter misses, OR the JANG loader runs after the converter.

**Status:** OPEN — needs investigation.

**Test:**
```bash
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-JANG_2S-TEXT --raw-bench
# MUST show: AsType < 100 (currently 1,105 = FAILING)
# TARGET: > 80 tok/s (currently 59-73)
```

---

## 5. VLM Image/Video Recognition

**Issue:** VLM models (Gemma4 VLM, Qwen3.5 VLM) must correctly process images and produce meaningful descriptions.

**Files affected:** Libraries/MLXVLM/Models/

**Test:**
```bash
.build/release/TestRunner <vlm-model> "Describe this image" --vlm --image-test
# MUST: produce coherent image description, not garbage
```

---

## Quick Regression Suite

Run ALL of these before any merge:

```bash
cd TestRunner && swift build -c release

# 1. Qwen3.5 MLX 4-bit: must be > 90 tok/s, AsType < 100
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-4bit --raw-bench

# 2. Llama 3.2 1B: must be > 300 tok/s, 0 AsType, IDENTICAL output
.build/release/TestRunner <llama-path> "What is 2+2?" --compiled

# 3. Gemma4 26B: TARGET > 80 tok/s (CURRENTLY FAILING at 27)
.build/release/TestRunner <gemma4-26b-path> --raw-bench

# 4. JANG model: must load without crash, AsType < 100
.build/release/TestRunner ~/models/Qwen3.5-35B-A3B-JANG_2S-TEXT --raw-bench

# 5. swift build (main package, not just TestRunner)
cd .. && swift build
```

---

## Performance Baselines (M4 Max 128GB)

| Model | Target tok/s | Method |
|---|---|---|
| Qwen3.5-35B MLX 4-bit | > 90 (uncompiled), > 110 (compiled) | --raw-bench |
| Llama 3.2 1B 4-bit | > 300 | --raw-bench |
| Gemma4 E2B 4-bit | > 150 | --raw-bench |
| Gemma4 26B 4-bit | > 80 | --raw-bench |
| Qwen3.5-35B JANG_2S | > 80 | --raw-bench |
