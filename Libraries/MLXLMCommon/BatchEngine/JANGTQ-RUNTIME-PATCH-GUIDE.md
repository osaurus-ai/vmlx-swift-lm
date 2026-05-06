# JANGTQ / JANG runtime config patching — implementation guide for osaurus

> **Audience**: osaurus integrators and anyone consuming `vmlx-swift-lm`
> as a library. Documents the auto-patching contract that fixes JANG /
> JANGTQ / generic MLX-quantized bundles whose `config.json` quantization
> metadata has drifted out of sync with the actual safetensors.
>
> **TL;DR**: nothing to do on the osaurus side. Bundles load and decode
> correctly regardless of whether `config.json` was re-stamped, lost
> per-layer overrides, or never had a quant block. The runtime walks
> safetensors shapes and patches the in-memory config before
> instantiating any `QuantizedLinear`. A `[Load]` line on stderr tells
> you when a patch was applied.

---

## What problem this solves

JANG / JANGTQ converters write `config.json` with a top-level
`quantization` block plus per-layer overrides for routed-MoE / shared /
attention layers that use different bit widths from the default. Several
classes of bug or operator action can cause that metadata to disagree
with the actual on-disk safetensors:

1. **Re-stamped config**: someone (operator or another tool) regenerated
   `config.json` with a uniform `bits: 8` block while the bundle's
   routed-MoE codebook is still 2-bit. Reproduced on
   `JANGQ-AI/MiniMax-M2.7-Small-JANGTQ` and
   `JANGQ-AI/DeepSeekV4-Flash-JANGTQ` between sessions on this machine.
2. **Stripped config**: `quantization` block removed entirely (e.g. some
   converters or download mirrors), but `.scales` / `.biases` /
   `.weight` triples are still in the safetensors.
3. **Converter bug**: a per-layer override was emitted with the wrong
   bits or `group_size`. Common when the converter classify rule misses
   a new module name.
4. **Restrictive `bitWidthsUsed` from JangConfig defaults**: when the
   bundle's `jang_config.json` doesn't carry quantization metadata of
   its own (e.g. DSV4-Flash bundles ship `weight_format: "bf16"`), the
   library's `JangConfig` default sets `bitWidthsUsed = [2, 4, 6]`
   which excludes 8. Old code's fallback search couldn't find 8-bit
   attention layers and silently picked `(4, 32)` — wrong dequant,
   `rms_norm` shape trap mid-decode.

All four classes silently corrupt the runtime, producing either
garbage logits, a `(*weight) must have the same size as the last
dimension of x` fatal mid-decode, or a `TurboQuantSwitchLinear` "sidecar
not loaded" trap. None of them are caught by the existing
`mlx_quantize_matmul` validation because mlx trusts whatever
`(bits, group_size)` you tell it.

---

## How the runtime patches it

`Libraries/MLXLMCommon/Load.swift` runs a shape-authoritative inference
pass before any `QuantizedLinear` is instantiated. The pass walks every
`.scales` key in the bundle and derives `(bits, group_size)` from the
`(weight, scales)` shape pair using the unambiguous identity:

    weight.shape[-1] * 32 == bits * in_dim
    scales.shape[-1] * group_size == in_dim

Disambiguation when multiple `(bits, gs)` pairs satisfy the same
shape uses a fixed empirical preference order:

    (8, 32), (8, 64), (8, 128),
    (4, 32), (4, 64), (4, 128),
    (2, 32), (2, 64), (2, 128),
    (3, 32), (6, 32)

This ordering is correct for JANG / JANGTQ converters because they
classify attention / embed / lm_head / shared at the highest available
bit width with the smallest valid `gs`, and routed experts at the
matching low bits with the same `gs`. When two pairs share `bits * gs`
(e.g. `(8, 32)` ≡ `(4, 64)` ≡ `(2, 128)` all give the same packed-to-
scales ratio), the converter NEVER picks the low-bit variant for
attention. `(8, 32)` first picks correctly; routed experts fail the
`(8, *)` ratio check and fall through to `(4, *)` or `(2, *)`.

The pass branches on what `config.json` already provides:

| Path | Trigger | Behaviour |
|---|---|---|
| **A** | `jang_config.json` present | Walk shapes; emit per-layer overrides where `(bits, gs)` differs from `JangConfig.quantization.bitWidthsUsed.min()`. Always considers 8-bit even if `bitWidthsUsed` excludes it. |
| **B** | `config.json` has a `perLayerQuantization` block | Walk shapes and cross-check every config-supplied override. Disagreements are patched (shape wins). |
| **C** | `config.json` has only top-level `quantization` | Walk shapes; emit per-layer overrides over the supplied default. |
| **D** | no quant signal but `.scales` keys exist | Walk shapes fully; pick the most-frequent `(bits, gs)` as default and emit overrides for the rest. |

All four paths are **idempotent** — clean configs that already match
their safetensors produce zero per-layer overrides and the loader
proceeds with the supplied defaults unchanged.

---

## What osaurus sees on stderr

The runtime emits one diagnostic line per non-trivial patch action:

```
[Load] JANG shape walk produced 250 per-layer quant override(s) over default (bits=2, gs=32)
```

Lines you may observe and what they mean:

- `[Load] JANG shape walk produced N per-layer quant override(s) over default (bits=B, gs=G)` — Path A. Bundle has `jang_config.json`. `N` layers diverged from the chosen default; this is normal for any JANG bundle whose default isn't 8-bit (the attention layers are always 8-bit).
- `[Load] config per-layer quant disagreed with safetensors shapes — patched N layer(s) from shape walk` — Path B. Config-supplied per-layer overrides were wrong; `N` were corrected. Treat as a warning sign that the bundle's `config.json` was edited externally; consider re-pushing a clean copy to the source.
- `[Load] non-JANG shape walk produced N per-layer quant override(s) over default (bits=B, gs=G)` — Path C. Bundle has top-level `quantization` only; `N` layers needed per-layer overrides. Common for stripped-config bundles.
- `[Load] config has no quant block — shape walk inferred default (bits=B, gs=G) plus N override(s)` — Path D. Bundle has no `quantization` block at all but is in fact quantized. The runtime inferred the dominant `(bits, gs)`.

The lines fire BEFORE model instantiation, so they appear early in the
load timeline alongside `[ModelFactory]` lines.

If osaurus surfaces logs to a UI, these lines are useful as an
"informational" tier — they indicate the bundle was patched but not
that anything failed.

---

## What osaurus does NOT need to do

- ❌ No need to validate or pre-patch `config.json` before calling
  `loadModel(...)`.
- ❌ No need to inspect safetensors shapes or compute bit widths.
- ❌ No need to ship per-bundle workarounds for known-bad downloads.
- ❌ No need to set environment variables for the patch to engage —
  it always runs.

Just call `MLXLMCommon.loadModel(from: URL)` like always; the patch
runs transparently.

---

## Edge cases and what happens

| Scenario | Behaviour |
|---|---|
| Layer has `.scales` and `.weight` with valid shapes | Patched correctly. |
| Layer has `.weight` but no `.scales` | Treated as unquantized. Skipped by the walker. |
| Layer has `.scales` but no `.weight` | Walker logs nothing for this layer; `loadWeights` will throw later when the model tries to update it. |
| Shape pair gives non-divisible math | Walker returns the JANG-default `(4, gs)` for that layer. mlx will fatal-error if the weight is actually stored differently — surfaces the bug instead of silently corrupting. |
| MoE gate weights with different `group_size` than body | Primary path with `knownGroupSize` fails; fallback preference order picks the right `(bits, gs)`. Verified on Qwen3.5 / MiniMax / Nemotron MoE bundles. |
| MXTQ codebook tensors (`tq_packed` / `tq_norms`) | These are not `.scales` triples; the walker skips them. The `TurboQuantSwitchLinear` runtime path consumes them via `JANGTQRuntimeCache.shared` keyed by `(inFeatures, bits)` from the sidecar — see `DSV4_JANGTQ_BITS` and `DSV4_FORCE_JANGTQ` env knobs in the OSAURUS-INTEGRATION doc for the codebook bits resolution. |
| VLM models with `language_model.` prefix | Both prefixed and stripped key forms are kept in the per-layer map so VLM and LLM dispatch paths both work. |
| Bundle whose `.scales` keys disagree across the same logical layer (corrupted download) | Each `.scales` key is walked independently; the disagreeing layer gets a per-layer override that may produce wrong dequant. mlx will likely fatal-error at first forward — surfaces the corruption. |

---

## Knobs that interact with the patch

These environment variables are documented in
`Libraries/MLXLMCommon/BatchEngine/OSAURUS-INTEGRATION.md` but worth
mentioning here:

- `DSV4_FORCE_JANGTQ=1` — forces the JANGTQ model class for DSV4
  bundles whose `weight_format` stamp is `bf16` (mislabeled).
- `DSV4_JANGTQ_BITS={2,4}` — overrides the routed-MoE codebook bits.
  Only consulted by the JANGTQ codebook resolution path; orthogonal
  to the affine-quant shape walker covered here.
- `DSV4_KV_MODE={sliding,full,tq}` — DSV4 cache layout. Leave unset
  or set `sliding` in production so DSV4 keeps the SWA+CSA+HSA
  `DeepseekV4Cache` path. `full` and `tq` are diagnostics; global
  `GenerateParameters.kvMode = .turboQuant` must not auto-promote DSV4
  unless `DSV4_KV_MODE=tq` is set deliberately.

---

## Verification status (2026-04-25, M4 Max + M3 Ultra cross-runtime)

| Bundle | Path | Result | Live verification |
|---|---|---|---|
| `JANGQ-AI/MiniMax-M2.7-Small-JANGTQ` | A | 250 overrides, decode coherent | M4 Max — multi-turn + reasoning + tool + cache + disk |
| `JANGQ-AI/DeepSeekV4-Flash-JANGTQ` | A | 512 overrides, decode coherent | M4 Max — multi-turn + reasoning + cache + disk + FIM-vs-chat |
| Stock MLX 4-bit (any) | B | 0 overrides on clean bundles, patches on re-stamped | Inferred from path B logic; safe by idempotence |
| Stock MLX with stripped config | C | dominant `(bits, gs)` inferred | Inferred from path C logic |
| Bundle with no `quantization` block | D | full shape inference | Inferred from path D logic |

The cross-runtime parity contract is also locked at the model-internal
level — see `research/DSV4-HC-PRE-FP32-CAST-FIX-2026-04-25.md` for the
fp32 mHC RMSnorm cast that was required to keep M3 Ultra (Mac Studio)
producing coherent decode under bf16 SIMD.

---

## When to escalate

The runtime can't recover from these — they require operator action:

- **Bundle is missing the `jangtq_runtime.safetensors` sidecar AND
  `jang_config.json` declares `weight_format: "mxtq"`**. The loader
  fails at load time with a clear message naming the missing file.
  Re-download the bundle including the sidecar.
- **Two layers' shapes disagree about `bits * gs`** — i.e., the
  bundle is partially corrupted. The shape walker can't reconcile;
  the most-frequent default is chosen and the disagreeing layer
  produces wrong dequant. mlx will likely surface this at first
  forward as a shape-mismatch fatal.
- **`config.json` says the bundle is one model_type but the weights
  are another**. The factory dispatches on `model_type` before the
  shape walker runs; weight-key sanitize will fail at load.

---

## Implementation pointer

- Universal shape walker:
  `Libraries/MLXLMCommon/JangLoader.swift::inferPerLayerQuantizationFromShapes`
- Disambiguation helper:
  `Libraries/MLXLMCommon/JangLoader.swift::inferBitWidthAndGroupSize`
- Dispatch & branching:
  `Libraries/MLXLMCommon/Load.swift::loadWeights` (the
  `effectivePerLayerQuantization` block, ~lines 113-180)
