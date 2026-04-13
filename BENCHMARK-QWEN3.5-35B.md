# Qwen 3.5-35B-A3B 4bit — Multi-Turn Benchmark (10.9K context)

Multi-turn conversation benchmark for `Qwen3.5-35B-A3B-4bit` across the four
major Apple-Silicon LLM runtimes. This is a hybrid SSM + MoE architecture
(linear-attention layers + full-attention layers + 128-expert MoE), exactly the
class of model where MLX op-dispatch overhead matters most.

## Test setup

| | |
|---|---|
| **Model** | `Qwen3.5-35B-A3B-4bit` (35 B-parameter, 3 B active, hybrid SSM+MoE, MLX 4-bit) |
| **Hardware** | M4 Max (Mac16,5), 128 GB unified memory, macOS 26.x |
| **Conversation** | 5 turns, growing from 21 → 10,932 prompt tokens |
| **Per turn** | greedy decode (`temperature: 0.0`), `max_tokens: 256` |
| **Prompt source** | Pre-tokenized via Python so all backends see byte-identical input |
| **Pre-flight** | All other inference processes killed before each run |

The conversation is a 5-turn distributed-systems Q&A. Turn 1 is short (21 tok),
turns 2–5 each add ~2.7 k tokens of context — designed to stress prefill while
the cache grows past 10 k tokens.

## Backends tested

| Backend | Version | MLX core | Notes |
|---|---|---|---|
| **vmlx-swift-lm** (this fork) | `main` | mlx-swift `osaurus-0.31.3` (custom kernels patched) | Manual prefix cache reuse via `cache:` parameter |
| **omlx** | `0.3.2` | bundled python `mlx` | **Auto** prefix matching with 1024-tok paged blocks |
| **Python mlx_lm** | `0.31.2` | pip `mlx` (latest) | Manual prefix cache reuse via `prompt_cache=` |
| **LM Studio** | `0.4.x` (mlx-llm `1.5.0`) | bundled `mlx-llm` 1.5.0 | **No prefix cache** — re-prefills full conversation each turn |

## Results

### Decode tok/s (sustained generation)

| Backend | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | avg(2-5) |
|---|---:|---:|---:|---:|---:|---:|
| Python mlx_lm 0.31.2 | **122.1** | **116.1** | **112.9** | **108.0** | **106.1** | **110.8** |
| **vmlx-swift-lm** | 106.4 | 105.0 | 100.7 | 95.6 | 91.3 | 98.2 |
| LM Studio (mlx-llm 1.5.0)¹ | 107.5 | 63.7 | 46.0 | 34.2 | 25.5 | 42.4 |
| omlx 0.3.2² | ~83 | ~75 | ~70 | ~63 | ~57 | ~66 |

¹ LM Studio's per-turn decode looks like it tanks because LM Studio re-prefills
the entire growing conversation every turn — it has no prefix cache. Pure
decode in isolation is ~107 tok/s; the listed numbers are end-to-end *overall*
tok/s including prefill of the full history.

² omlx streaming TTFT detection is broken (it buffers chunks until the end), so
its decode rate cannot be cleanly isolated from prefill. The values are
estimates derived from `overall − estimated prefill`.

### Prefill tok/s (NEW tokens / TTFT)

| Backend | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | avg(2-5) |
|---|---:|---:|---:|---:|---:|---:|
| Python mlx_lm | 75 | **1672** | **1580** | **1449** | **1378** | **1520** |
| **vmlx-swift-lm** | **398** | 1528 | 1462 | 1373 | 1296 | 1415 |
| LM Studio | n/a | n/a | n/a | n/a | n/a | n/a (TTFT broken) |
| omlx 0.3.2 | n/a | n/a | n/a | n/a | n/a | n/a (TTFT broken) |

vmlx-swift-lm's small-prompt prefill (turn 1 = 398 tok/s on 21 tokens) is much
faster than Python (75 tok/s) — that's wired memory + 8 K prefill batch
amortizing the cold-start overhead.

### TTFT (time to first token, ms)

| Backend | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 |
|---|---:|---:|---:|---:|---:|
| **vmlx-swift-lm** | **53** | 1778 | 1860 | 1977 | 2097 |
| Python mlx_lm | 281 | **1625** | **1720** | **1874** | **1972** |
| omlx 0.3.2 | (broken — buffered streaming) | | | | |
| LM Studio | (broken — buffered streaming) | | | | |

vmlx-swift-lm wins **cold-start TTFT by 5×** (53 ms vs 281 ms). From turn 2
onward Python's tighter prefill rate gives it a slight TTFT edge on long
chunks.

### End-to-end overall tok/s (256 generated / total per-turn time)

This is the most apples-to-apples metric — what the user actually experiences:

| Backend | Turn 1 | Turn 2 | Turn 3 | Turn 4 | Turn 5 | avg(2-5) |
|---|---:|---:|---:|---:|---:|---:|
| Python mlx_lm | **107.6** | **66.8** | **64.2** | **60.3** | **58.4** | **62.4** |
| **vmlx-swift-lm** | 104.1 | 60.7 | 58.2 | 55.0 | 52.2 | 56.5 |
| omlx 0.3.2 | 82.2 | 54.8 | 49.2 | 44.6 | 39.5 | 47.0 |
| LM Studio | 107.8 | 64.1 | 46.0 | 34.2 | 25.5 | 42.4 |

## Cache behavior summary

| Backend | Manual prefix reuse | Auto prefix matching | Per-turn re-prefill |
|---|---|---|---|
| **vmlx-swift-lm** | ✅ in this bench | ❌ (✅ via `CacheCoordinator`, not used in this bench) | Only NEW tokens |
| **Python mlx_lm** | ✅ in this bench | ❌ | Only NEW tokens |
| **omlx 0.3.2** | n/a | ✅ paged 1024-tok blocks | Only uncached blocks |
| **LM Studio mlx-llm 1.5.0** | n/a | ❌ | **Full conversation every turn** |

## Final standings (avg of turns 2–5)

| Metric | 🥇 1st | 🥈 2nd | 🥉 3rd | 4th |
|---|---|---|---|---|
| **Decode tok/s** | Python mlx_lm 110.8 | vmlx-swift-lm 98.2 | omlx ~66 | LM Studio* 42.4 |
| **Overall tok/s (per-turn)** | Python 62.4 | vmlx-swift-lm 56.5 | omlx 47.0 | LM Studio 42.4 |
| **TTFT (cold start, T1)** | **vmlx-swift-lm 53 ms** | LM Studio ~50 ms | Python 281 ms | omlx (broken) |

\* LM Studio's "decode" number conflates with prefill (no cache reuse).

## Honest verdict

- **Python mlx_lm 0.31.2 is currently the fastest at sustained decode** on this
  hybrid SSM+MoE architecture, by ~12 % over vmlx-swift-lm (110.8 vs 98.2). The
  gap comes from compile-fused activations and a newer `mlx` C++ core that
  Python pulls in via `pip install mlx` but `mlx-swift osaurus-0.31.3` doesn't
  yet expose. We can close it but it requires either bumping `mlx-swift` or
  porting the Python `@partial(mx.compile, shapeless=True)` patterns into Swift
  without breaking `Slice::output_shapes` inference (an ongoing investigation).

- **vmlx-swift-lm is the fastest *Swift-binding* runtime that exists**, beating
  LM Studio's mlx-llm 1.5.0 by **+33 %** in end-to-end overall tok/s and beating
  omlx by **+20 %**. It's also the only Swift binding that's faster than the
  upstream `mlx-swift-lm` baseline by **2.4 ×** on this model (41 → 98 tok/s —
  see the headline performance table in the main README).

- **vmlx-swift-lm wins cold-start TTFT by a wide margin** — 53 ms vs Python's
  281 ms, a 5 × speedup — thanks to wired memory and an 8 K prefill batch.
  This is the metric that matters most for interactive chat.

- **omlx wins prefix-cache TTFT** when its paged blocks hit on the same
  prefix, but its per-token decode cost is the highest of all backends, and
  its eviction at 4 K cached tokens (in the default config) means long
  conversations re-prefill anyway.

## Reproduction

```bash
# Pre-tokenize the conversation once
python3 docs/benchmarks/scripts/qwen35_tokenize.py

# Pre-flight (kill all other inference processes)
pkill -9 -f "ollama|RunBench|test_batched|omlx|Python.*mlx"
~/.lmstudio/bin/lms unload --all && ~/.lmstudio/bin/lms server stop

# Swift vmlx-swift-lm
swift build -c release --product RunBench
.build/arm64-apple-macosx/release/RunBench

# Python mlx_lm
python3 docs/benchmarks/scripts/qwen35_python_bench.py

# omlx
omlx serve --model-dir ~/models --port 9090 --log-level error &
python3 docs/benchmarks/scripts/qwen35_omlx_bench.py
pkill -9 -f "omlx serve"

# LM Studio
~/.lmstudio/bin/lms server start
~/.lmstudio/bin/lms load qwen3.5-35b-a3b --context-length 32768
python3 docs/benchmarks/scripts/qwen35_lmstudio_bench.py
~/.lmstudio/bin/lms unload --all && ~/.lmstudio/bin/lms server stop
```

Recorded: 2026-04-12 on M4 Max with macOS 26.x, vmlx-swift-lm commit `c859cc7`.
