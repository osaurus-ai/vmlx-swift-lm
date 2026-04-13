# Llama 3.2 1B Instruct 4bit — Multi-Turn Benchmark (11K context)

Multi-turn conversation benchmark for `mlx-community/Llama-3.2-1B-Instruct-4bit`
across all four major Apple-Silicon LLM runtimes. This is the **pure dense
baseline** — no MoE routing, no SSM layers, no exotic architecture — just plain
Llama transformer.

## Test setup

| | |
|---|---|
| **Model** | `mlx-community/Llama-3.2-1B-Instruct-4bit` (1.2 B params, dense, MLX 4-bit) |
| **Hardware** | M4 Max (Mac16,5), 128 GB unified memory, macOS 26.x |
| **Conversation** | 5 turns, growing from 47 → 11,022 prompt tokens |
| **Per turn** | greedy decode (`temperature: 0.0`), `max_tokens: 256` |
| **Prompt source** | Pre-tokenized via Python so all backends see byte-identical input |
| **Pre-flight** | All other inference processes killed before each run |

## Backends tested

| Backend | Version | Backend type | Notes |
|---|---|---|---|
| **vmlx-swift-lm** (this fork) | `main` `9aac9d3` | Swift + mlx-swift | Manual prefix cache reuse |
| **Python mlx_lm** | `0.31.2` | Python + mlx C++ | Manual prefix cache reuse |
| **LM Studio** | `0.4.x` (mlx-llm `1.5.0`) | Swift + mlx-llm | No prefix cache (full re-prefill) |
| **omlx** | `0.3.2` | Python + mlx C++ paged | Auto prefix matching (1024-tok blocks) |

## Results

### Decode tok/s (sustained generation, isolated from prefill)

| Backend | T1 | T2 | T3 | T4 | T5 | avg(2-5) |
|---|---:|---:|---:|---:|---:|---:|
| Python mlx_lm | **463.4** | **349.6** | **289.6** | **252.5** | **238.3** | **282.5** |
| **vmlx-swift-lm** | 440.9 | 336.0 | 281.4 | 239.6 | 228.7 | 271.4 |
| LM Studio¹ | n/a | n/a | n/a | n/a | n/a | n/a |
| omlx 0.3.2¹ | n/a | n/a | n/a | n/a | n/a | n/a |

¹ LM Studio and omlx don't expose a streaming TTFT split — their decode rate
cannot be cleanly isolated from prefill. See "overall" section below.

### Prefill tok/s (NEW tokens / TTFT)

| Backend | T1 | T2 | T3 | T4 | T5 | avg(2-5) |
|---|---:|---:|---:|---:|---:|---:|
| Python mlx_lm | 591 | **4847** | **4962** | **4299** | **3839** | **4487** |
| **vmlx-swift-lm** | **2240** | 4491 | 3941 | 3502 | 3131 | 3766 |

vmlx-swift-lm wins **cold-start prefill** (turn 1) by **3.8 ×** (2240 vs 591 tok/s)
because of wired memory and an 8 K prefill batch. From turn 2 onward Python's
prefill rate edges out by ~16 %.

### TTFT (time to first token, ms)

| Backend | T1 | T2 | T3 | T4 | T5 |
|---|---:|---:|---:|---:|---:|
| **vmlx-swift-lm** | **21** | 608 | 693 | 780 | 872 |
| Python mlx_lm | 79 | **564** | **551** | **635** | **712** |

Cold-start TTFT win for vmlx-swift-lm: **21 ms vs 79 ms (3.8 × faster)**.

### End-to-end overall tok/s (256 generated / total per-turn time)

This is the most apples-to-apples metric — what the user actually experiences:

| Backend | T1 | T2 | T3 | T4 | T5 | avg(2-5) |
|---|---:|---:|---:|---:|---:|---:|
| Python mlx_lm | **441.4** | **197.5** | **167.7** | **143.5** | **131.1** | **159.9** |
| **vmlx-swift-lm** | 427.4 | 192.0 | 164.2 | 140.2 | 128.5 | 156.2 |
| LM Studio | 413.7 | 193.3 | 164.3 | 138.9 | 126.7 | 155.8 |
| omlx 0.3.2 | 393.8 | 184.4 | 153.0 | 131.6 | 119.7 | 147.2 |

## Final standings

| Metric | 🥇 1st | 🥈 2nd | 🥉 3rd | 4th |
|---|---|---|---|---|
| **Decode tok/s avg** | Python 282.5 | **vmlx 271.4** *(-4 %)* | (n/a — others lump prefill) | |
| **Overall tok/s avg** | Python 159.9 | **vmlx 156.2** *(-2 %)* | LM Studio 155.8 *(-3 %)* | omlx 147.2 *(-8 %)* |
| **Cold-start TTFT** | **vmlx 21 ms** | Python 79 ms *(3.8 × slower)* | LM Studio (no split) | omlx (broken) |

## Key observations

1. **Dense models are essentially tied across all four backends.** All four
   land within ~8 % of each other on overall tok/s. The FFI-tax gap that
   hurts Swift on big MoE models barely matters when there's no MoE routing.

2. **Swift wins cold-start TTFT decisively.** 21 ms vs Python's 79 ms (3.8 ×
   faster). Wired memory + 8 K prefill batch + lower per-call overhead all
   contribute.

3. **vmlx-swift-lm decode is within 4 % of Python** on dense Llama, vs ~12 %
   behind on Gemma 4 26B and ~10 % behind on Qwen 3.5-35B. **The gap scales
   with the number of MoE-routing operations per token**, not model size.

4. **omlx is consistently last** at end-to-end speed despite having paged
   automatic prefix caching. Its per-token decode cost is genuinely higher
   than the other Swift bindings.

5. **LM Studio doesn't appear to do prefix caching** for the conversation —
   `prompt_tokens` matches the full conversation each turn (3035, 6024,
   9010, 11998), confirming full re-prefill. Despite this, its prefill rate
   keeps it competitive.

## Notes on fairness

1. Pre-tokenized prompts (via Python) ensure all backends see byte-identical
   input.
2. **All other inference runtimes killed before each backend run.** A stale
   Ollama background process or `test_batched_concurrent.py` Python process
   added measurable noise in earlier runs; pre-flight scripts now verify a
   clean state.
3. **Manual cache reuse** for vmlx-swift-lm and Python mlx_lm — both feed only
   the new tokens each turn against a persistent KVCache.
4. **omlx** detects the shared prefix automatically and reports
   `cached_tokens` in the streaming response.
5. **LM Studio** prompt token count proves no prefix cache reuse.

## Reproduction

```bash
# Pre-tokenize the conversation once
python3 docs/benchmarks/scripts/llama32_tokenize.py

# Pre-flight (kill all other inference processes)
pkill -9 -f "ollama|RunBench|test_batched|omlx|Python.*mlx"
~/.lmstudio/bin/lms unload --all && ~/.lmstudio/bin/lms server stop

# Swift vmlx-swift-lm
swift build -c release --product RunBench
.build/arm64-apple-macosx/release/RunBench

# Python mlx_lm
python3 docs/benchmarks/scripts/llama32_python_bench.py

# omlx
omlx serve --model-dir ~/.cache/huggingface/hub/models--mlx-community--Llama-3.2-1B-Instruct-4bit/snapshots --port 9090 --log-level error &
python3 docs/benchmarks/scripts/llama32_omlx_bench.py
pkill -9 -f "omlx serve"

# LM Studio
~/.lmstudio/bin/lms server start
~/.lmstudio/bin/lms load llama-3.2-1b-instruct --context-length 16384
python3 docs/benchmarks/scripts/llama32_lmstudio_bench.py
~/.lmstudio/bin/lms unload --all && ~/.lmstudio/bin/lms server stop
```

Recorded: 2026-04-12 on M4 Max with macOS 26.x, vmlx-swift-lm commit `9aac9d3`
(post `TokenIterator.step()` 1D/2D input fix).
