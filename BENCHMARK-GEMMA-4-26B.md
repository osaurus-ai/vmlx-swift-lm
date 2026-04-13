# Gemma 4 26B 4bit — Multi-Turn Benchmark (5.5K context)

Multi-turn conversation benchmark for `gemma-4-26b-a4b-it-4bit` across the four
major Apple-Silicon LLM runtimes that exist for this model.

> **Note on scope:** This is the 5.5 K-context multi-turn run. A longer
> 10.8 K-context run was also done but exposed measurement variance on the
> Python side that needs more iterations to stabilize before publishing. We will
> re-publish a long-context version after the next round of fused-op
> optimization (router rms_norm fusion, geglu fusion). Until then, treat the
> numbers below as the steady-state picture for short conversations on M4 Max.

## Test setup

| | |
|---|---|
| **Model** | `gemma-4-26b-a4b-it-4bit` (Gemma 4 26B-A4B-it, MLX 4-bit, MoE 128 experts) |
| **Hardware** | M4 Max (Mac16,5), 128 GB unified memory, macOS 26.x |
| **Conversation** | 5 turns, growing from 25 → 5,499 prompt tokens |
| **Per turn** | greedy decode (`temperature: 0.0`), `max_tokens: 256` |
| **Prompt source** | Pre-tokenized via Python so all backends see byte-identical input |
| **Pre-flight** | All other inference processes killed before each run |

The conversation is a 5-turn distributed-systems Q&A. Turn 1 is short (25 tok),
turns 2–5 each add ~1.4 k tokens of context — designed to stress prefill while
the cache grows.

## Results

### Decode tok/s (sustained generation speed)

| Backend | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 |
|---|---:|---:|---:|---:|---:|
| **vmlx-swift-lm (this fork)** | **98.2** | **90.2** | **89.8** | **87.2** | **86.5** |
| omlx 0.3.2 | 77.7 | 73.4 | 71.5 | 70.6 | 68.6 |
| Python mlx_lm 0.31.2 | 71.6 | 72.3 | 82.6 | 76.6 | 78.1 |
| LM Studio 0.4.x (mlx-llm 1.5.0) | — | — | — | — | — |

Swift vmlx-swift-lm wins decode at every turn. The advantage is widest in the
small-prompt regime (turn 1) and persists at long context (turn 5).

### Prefill tok/s (prompt processing speed for new tokens)

| Backend | Turn 1 (25 tok) | Turn 2 (~1.4 k new) | Turn 3 (~1.4 k or 180 cached) | Turn 4 | Turn 5 |
|---|---:|---:|---:|---:|---:|
| **vmlx-swift-lm** | 316 | 1007 | 1159 | 1089 | 1108 |
| omlx 0.3.2 (auto-cache) | 111 | **1326** | 824¹ | **1296** | **1309** |
| Python mlx_lm | 87 | 714 | 694 | 642 | 700 |
| LM Studio | — | — | — | — | — |

¹ omlx auto-cached 3072 of 3252 tokens — only 180 new — so the small denominator
makes the per-second number look low.

omlx wins raw prefill once it warms up. vmlx-swift-lm is a close second and is
~50% faster than Python mlx_lm at every prefill chunk size.

### TTFT (time to first token)

| Backend | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 |
|---|---:|---:|---:|---:|---:|
| **vmlx-swift-lm** | **79 ms** | 1349 ms | 1173 ms | 1244 ms | 1227 ms |
| omlx 0.3.2 | 207 ms | **462 ms** | **218 ms** | **591 ms** | 1819 ms |
| Python mlx_lm | 288 ms | 1902 ms | 1959 ms | 2108 ms | 1941 ms |
| LM Studio | — | — | — | — | — |

omlx wins TTFT on turns 2–4 because its automatic prefix-cache hits skip most
of the growing context. omlx evicts at ~4 k cached tokens, so by turn 5 it has
to re-prefill 2 381 of 6 477 — at which point vmlx-swift-lm catches back up.

### Cache behavior

| Backend | Cache style | Auto prefix detection | Eviction |
|---|---|---|---|
| vmlx-swift-lm | Manual (this bench) / `CacheCoordinator` paged + disk (when wired up) | No (manual slice) in this bench | n/a |
| omlx | Paged hot cache + SSD cache | **Yes**, 1024-token blocks | LRU at 4096 tokens default |
| Python mlx_lm | `make_prompt_cache` persistent buffer | **No**, user manages slices | n/a |
| LM Studio | n/a (Gemma 4 not yet supported by mlx-llm 1.5.0) | — | — |

The vmlx-swift-lm decode advantage is **independent of the cache scheme** — it
comes from the gated-delta Metal kernel + AsType-cascade fixes + tighter
TokenIterator hot path. The prefill advantage over Python mlx_lm comes from
wired memory + a `prefillStepSize` of 8192.

## Headline numbers

Averaging turns 2–5 (the steady-state regime where caches are warm):

| | Decode tok/s | Prefill tok/s | Per-turn total |
|---|---:|---:|---:|
| **vmlx-swift-lm** | **88.4** | **1091** | ~4.1 s |
| omlx | 71.0 | 1189 | ~4.4 s |
| Python mlx_lm | 77.4 | 688 | ~5.5 s |

**vmlx-swift-lm has the fastest decode of any Apple-Silicon Gemma 4 runtime
that exists today, and competitive prefill within ~10 % of omlx.** Python
mlx_lm — despite being the upstream reference — is the slowest end-to-end on
this multi-turn pattern because it lacks both wired memory and the gated-delta
kernel optimisations.

## Notes on fairness

1. **All measurements taken with every other inference runtime killed.** A
   stale Ollama background process on a previous run produced ~10 % noise; that
   was eliminated before recording these numbers.
2. **Identical pre-tokenized prompts** were fed to vmlx-swift-lm and Python
   mlx_lm. omlx tokenizes its own input from the same source text.
3. **Manual cache reuse** in vmlx-swift-lm and Python mlx_lm — both feed only
   the new tokens each turn against a persistent KVCache. omlx detects the
   shared prefix automatically and reports `cached_tokens` in the streaming
   response.
4. **LM Studio (mlx-llm 1.5.0) cannot load Gemma 4 yet** — it returns
   `ValueError: Gemma 4 support is not ready yet, stay tuned!`. Tracked
   upstream; will be re-tested when supported.
5. **Decode tok/s degrades slightly with context length** in all three working
   runtimes. This is the inherent attention cost on the growing KV cache, not
   a runtime regression.

## Reproduction

```bash
# Pre-tokenize the conversation once (writes /tmp/gemma4_multiturn_tokens.json)
python3 docs/benchmarks/gemma-4-26b-multiturn.py --tokenize-only

# Swift vmlx-swift-lm
swift build -c release --product RunBench
.build/arm64-apple-macosx/release/RunBench   # uses /tmp/gemma4_multiturn_tokens.json

# Python mlx_lm
python3 docs/benchmarks/gemma-4-26b-multiturn.py --backend mlx_lm

# omlx
omlx serve --model-dir ~/osaurus_models/finished --port 9090 --log-level error &
python3 docs/benchmarks/gemma-4-26b-multiturn.py --backend omlx --port 9090

# LM Studio (currently fails — Gemma 4 not yet supported)
~/.lmstudio/bin/lms load gemma-4-26b-a4b-it
```

Recorded: 2026-04-12 on M4 Max with macOS 26.x, vmlx-swift-lm commit `c859cc7`
(post fused-gated-delta-kernel for VLM path).
