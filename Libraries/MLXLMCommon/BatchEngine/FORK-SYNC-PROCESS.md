# Fork-sync process — osaurus-ai/mlx-swift-lm ↔ ml-explore/mlx-swift-lm

**Status:** Landed on branch `fix/osaurus-integration-issues` (2026-04-20).
**Closes:** tpae's (2026-04-20) "are we keeping this up to date:
https://github.com/osaurus-ai/mlx-swift-lm"

## Three-remote topology

```
     ml-explore/mlx-swift-lm (upstream)          canonical Apple MLX tree
                 │
                 │  Apple merges new model families, v3 API tweaks,
                 │  tool-call format additions, bug fixes to original code.
                 ▼
     osaurus-ai/mlx-swift-lm (public)            Eric's STABLE public fork:
                 │                                upstream + ONLY carrying fixes
                 │                                needed for osaurus production.
                 │                                No BatchEngine / SpecDec /
                 │                                CacheCoordinator / TurboQuant.
                 ▼
     osaurus-ai/vmlx-swift-lm (origin, THIS REPO) Eric's DEV superset:
                                                  public + BatchEngine (batched
                                                  decode, compile, TurboQuant),
                                                  SpecDec (DFlash/DDTree),
                                                  Cache system (paged + disk),
                                                  JANG loader, etc.
```

Git remotes configured locally (verify with `git remote -v`):

```
origin    https://github.com/osaurus-ai/vmlx-swift-lm.git (fetch + push)
public    https://github.com/osaurus-ai/mlx-swift-lm.git (fetch + push)
upstream  https://github.com/ml-explore/mlx-swift-lm.git (fetch + push)
```

## Current state (2026-04-20)

| Measure | Count | What it means |
|---|---|---|
| `public/main` ahead of `upstream/main` | **75 commits** | Carrying fixes (Gemma4 VLM, Qwen3.5 norm shift, JANG loader, MXFP4, MLP overflow, Gemma4 multi-image/multi-turn, etc.). |
| `upstream/main` ahead of `public/main` | **18 commits** | Unmerged upstream changes. Need review-and-merge. |
| `origin/main` ahead of `public/main` | **120 commits** | BatchEngine + SpecDec + CacheCoordinator + TurboQuant + all additions layered on top of `public`. |
| `public/main` ahead of `origin/main` | **0 commits** | `origin` is a strict superset of `public`. |

## What the public fork carries

All 75 carrying patches in `public/main..upstream/main` fall into three
groups:

### Group A — Upstream-compatible bug fixes (candidate for upstream PR)

Small, focused, self-contained fixes that address real bugs in upstream
code. Each would likely be accepted as a standalone PR against
`ml-explore/mlx-swift-lm`:

- `800e68c` — JANG mixed-precision MLP float16 overflow (silu(gate) × up >
  65504) — extended to Qwen3.5/Gemma3/MiniMax in `57fe2e5`.
- `8486b4c` — JANG MoE gate dequantization shape ambiguity.
- `10d547e` — JANG per-layer bit inference strict round-trip.
- `0db30fb` + `b59586f` — SwitchGLU / compiledGeluApproximate crash on
  MLXNN Power primitive.
- `acf2b58` — MXFP4/MXFP8 nil-bias handling during model load.
- `1ddabd7` — Gemma4 VLM sRGB tone curve conversion (images silently
  dropped).
- `bd01662` + `c4d698c` — Gemma4 multi-image processing and padding.
- `285a736` + `7917108` — Gemma4 / Gemma3n multi-turn crash on 1D tokens
  without batch dim.
- `2671c4c` — Gemma4 VLM vision tower / processor / maskedScatter.
- `534e427` — Gemma4 VLM processor wrong image token scan.
- `e47259c` + `833bbf2` — Gemma4 sanitize: skip clipped linear params +
  audio tower weights.
- `847a8c7` — remove auto-wired memory limit (crashes on smaller GPUs).
- `1584724` — resolve symlinks in loadWeights.
- `c627326` — log factory errors instead of silently swallowing.

**Action:** file these as upstream PRs in batches by theme (Gemma4 VLM
bundle, JANG overflow bundle, MXFP loader bundle). `upstream` remote is
configured for `git push` — do NOT force-push; instead prepare a branch
and open a PR via `gh pr create --repo ml-explore/mlx-swift-lm`.

### Group B — Additions that aren't bug fixes (probably fork-only)

- `7324494` + `e92a722` — Gemma 4 E2B / E4B support (PLE, KV sharing,
  double-wide MLP). **Upstream got their own at #185; our version has
  more JANG integration points.** On next sync review whether upstream's
  version supersedes ours or if we should upstream the JANG pieces.
- `b78dc79` + `0e6c19b` — Qwen3.5 norm shift (VLM JANG norm detection).
- Miscellaneous diagnostic logging (`f298889`, `d80e723`).

### Group C — Perf tunes (fork-only)

`5a44ba1` / `b186468` / `6bf2cf1` / `6dfeef9` / `2278e91` / `5de7d15`
/ `f5d48bf` / `2f5e4f8` / `9a3c13f` — compile GatedDelta, compile
sigmoid ops, asyncEval tuning, stream-split experiment (reverted),
asType elimination in Qwen3.5/Qwen3Next. Measured-fragile territory;
keep in fork until upstream has comparable benchmarks.

## Sync procedure — upstream → public

For each upstream release / quarterly refresh:

```bash
# 1. Fetch fresh from both remotes.
git fetch upstream
git fetch public

# 2. Check how far upstream has moved.
git log --oneline public/main..upstream/main | head -30

# 3. Decide merge strategy:
#    - If upstream added ≤ ~10 changes and they don't conflict with
#      carrying patches: fast-forward merge (preserve linear history).
#    - If upstream added new model families / refactored files that
#      also have carrying patches: rebase public onto upstream/main.
#      Rebase is usually correct because carrying patches are all
#      bug fixes on top of upstream code — they shouldn't be
#      "replayed as merges."

# 4. Start a sync branch off public/main locally.
git checkout -B sync/upstream-YYYYMMDD public/main

# 5. Merge upstream (preserves public's carrying patches as "new"
#    commits on top of upstream; keeps upstream's linear history
#    reachable):
git merge upstream/main
# OR rebase:
git rebase upstream/main

# 6. Resolve conflicts. Common hot-spots (based on the 75 carry patches):
#    - Libraries/MLXLLM/Models/Gemma4Text.swift (both sides modify)
#    - Libraries/MLXVLM/Models/Gemma4.swift
#    - Libraries/MLXLMCommon/ModelConfiguration.swift
#    - Libraries/MLXLLM/Models/Qwen35.swift / Qwen3Next.swift
#    - Libraries/MLXLMCommon/Tool/ToolCallFormat.swift

# 7. Verify: full build + the package unit tests pass + the
#    RunBench smoke scenarios work on the real models in ~/.mlxstudio.
swift build
swift test --skip-build --filter BatchKVCache
swift test --skip-build --filter GenerationReasoning
swift test --skip-build --filter StopStringMatcher
# Real-model smoke:
VMLX_CHAT_TEMPLATE_OVERRIDE=$PWD/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja \
  BENCH_GEMMA4_STRESS=1 \
  BENCH_MODEL=~/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit \
  BENCH_MAX_TOKENS=40 swift run -c release RunBench

# 8. Push to public.
git push public sync/upstream-YYYYMMDD:main
```

## Sync procedure — public → origin (vmlx)

`origin` is already strictly ahead of `public`; the sync is just a
fast-forward-able merge:

```bash
git fetch public
git checkout main   # on vmlx-swift-lm
git merge public/main
# No conflicts expected: public's 75 patches are already included in
# origin's 120-commit superset as the fork's base layer.
git push origin main
```

If conflicts DO appear (e.g., because a carry patch was subsequently
refactored in vmlx's superset layer), resolve in favour of the vmlx
version — vmlx's refactored layer has downstream dependencies
(BatchEngine / SpecDec) that public doesn't.

## Upstream PRs — candidate batch

After the 2026-04-20 review, the following should be submitted as PRs
to `ml-explore/mlx-swift-lm`:

1. **"Fix float16 overflow in JANG MLP / Qwen3.5 / Gemma3 / MiniMax"**
   — squash `800e68c`, `57fe2e5`. Keep the before/after numerical
   examples in the PR body.
2. **"Fix Gemma4 VLM image pipeline — sRGB / multi-image / sanitize"**
   — squash `1ddabd7` + `bd01662` + `c4d698c` + `2671c4c` + `534e427` +
   `e47259c` + `833bbf2`. This is a single coherent story once cleaned
   up; upstream's Gemma4 VLM work (#180) likely overlaps.
3. **"Fix Gemma4 / Gemma3n multi-turn 1D-token crash"** — `285a736` +
   `7917108`. Small, clean, obvious bug.
4. **"Skip SwitchGLU compiledGeluApproximate path that crashes on
   MLXNN Power primitive"** — `0db30fb` + `b59586f`. Workaround-
   flavoured but until MLXNN is fixed this IS the fix.

Leave Group B + Group C in the fork until upstream has equivalent
benchmarking / tooling to evaluate them.

## CI and acceptance gate

Before pushing any public/main update:

- Package builds on macOS 14 + Xcode 16 (our target surface).
- Unit suites green: `BatchKVCacheRotatingSlotTests`,
  `GenerationReasoningEventTests`, `StopStringMatcherTests`,
  `ReasoningParserTests`, `BatchEngineTests`, `BatchCausalMaskTests`,
  `ToolCallEdgeCasesTests`, `ToolTests`, `CacheCoordinatorTests`.
- Real-model smoke against at least one of:
  - `~/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit`
    (Gemma-4 SWA, prompt > 1024 tokens — regression for the 2026-04-20
    broadcast_shapes crash).
  - `~/.mlxstudio/models/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-MXFP4`
    (reasoning emission + tool-call format wiring).

Canonical bench invocation:

```bash
pkill -f xctest; pkill -f RunBench; pkill -f ollama; pkill -f lms
VMLX_CHAT_TEMPLATE_OVERRIDE=$PWD/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja \
  BENCH_GEMMA4_STRESS=1 \
  BENCH_STOP_STRINGS="END,STOP" \
  BENCH_MODEL=~/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit \
  BENCH_MAX_TOKENS=80 \
  swift run -c release RunBench
```

Expected output:

```
=== BENCH_GEMMA4_STRESS: Gemma-4 SWA crash regression ===
  …
[Turn 1 (> 1024 tokens)]
  prompt tokens = 1371
  chunks=N reasoningDeltas=M toolCalls=0 stopReason=(stop|length)
  .chunk preview: "…"
=== BENCH_GEMMA4_STRESS: passed (no broadcast_shapes crash) ===
```

A FAIL at any stage blocks the sync until resolved.

## When NOT to sync

- Upstream is mid-refactor (e.g., v4 API is being drafted). Wait until
  upstream tags a release or at least stabilizes `main`.
- Our fork has an in-flight PR to upstream. Let that land first so the
  next sync includes a smaller delta.
- We are inside a merge freeze for a downstream osaurus release. The
  fork-sync happens AFTER the release ship.

## Ownership

Fork sync is Eric's responsibility — no CI job does it automatically.
The upstream repo's maintainers are Apple's MLX team; changes in their
main branch can land at any time. Target a quarterly sync cadence with
ad-hoc syncs when upstream lands a critical fix.

## Related

- `OSAURUS-INTEGRATION.md` — what osaurus consumes from this repo.
- `OSAURUS-API-SURFACE.md` — per-symbol public-API surface.
- `GEMMA4-SLIDING-WINDOW-CRASH.md` — the SWA regression this fork-sync
  doc was written alongside.
- `REASONING-STREAM-EVENT.md` — the `.reasoning(String)` event the
  2026-04-20 sync lands on the origin fork; public doesn't carry it
  yet (only origin does — BatchEngine-dependent).
- `STOP-SEQUENCES-CONTRACT.md` — the `extraStopStrings` field added
  in the same session.
