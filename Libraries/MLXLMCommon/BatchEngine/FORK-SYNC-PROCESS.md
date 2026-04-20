# Upstream sync process — ml-explore/mlx-swift-lm ↔ osaurus-ai/vmlx-swift-lm

**Status:** 2026-04-20. `osaurus-ai/mlx-swift-lm` (the "clean" public
fork) is **deprecated** — everything lives on
`osaurus-ai/vmlx-swift-lm` now. This doc describes how to keep vmlx
in sync with the single upstream that matters:
`ml-explore/mlx-swift-lm`.

**Closes:** tpae's (2026-04-20) "are we keeping this up to date:
https://github.com/osaurus-ai/mlx-swift-lm" — short answer: no, and
we don't need to. Point consumers at `osaurus-ai/vmlx-swift-lm`
instead.

## Two-remote topology

```
     ml-explore/mlx-swift-lm (upstream)          canonical Apple MLX tree.
                 │                                new model families, v3
                 │                                API tweaks, bug fixes.
                 ▼
     osaurus-ai/vmlx-swift-lm (origin, THIS REPO) single source of truth.
                                                  = upstream + Eric's
                                                  carrying patches +
                                                  BatchEngine + SpecDec +
                                                  CacheCoordinator +
                                                  TurboQuant + JANG loader.
```

Local git remotes (verify with `git remote -v`):

```
origin    https://github.com/osaurus-ai/vmlx-swift-lm.git (fetch + push)
upstream  https://github.com/ml-explore/mlx-swift-lm.git (fetch + push)
```

If `osaurus-ai/mlx-swift-lm` is still configured as a `public` remote
from an earlier session, drop it — we don't use it:

```bash
git remote remove public    # one-time cleanup
```

## Deprecation note for osaurus-ai/mlx-swift-lm

That repo was a curated "upstream + just the carrying bug fixes"
fork. Maintaining two forks (the clean public one AND the superset
vmlx) has no value for anyone — osaurus integrators always want the
superset because that's where `BatchEngine`, `CacheCoordinator`,
`GenerateParameters.draftStrategy`, `extraStopStrings`, and
`.reasoning(String)` live. Any drift between the two was a pure
operational tax with no consumer.

**Action for osaurus integrators:** change your Package.swift
dependency to:

```swift
.package(url: "https://github.com/osaurus-ai/vmlx-swift-lm", branch: "main")
```

and remove any reference to `osaurus-ai/mlx-swift-lm`.

## Current state (2026-04-20)

- `origin/main` is ahead of `upstream/main` by 195 commits (120 vmlx
  superset on top of the 75 carrying fixes that used to live on the
  public fork — all merged into origin here).
- `upstream/main` has some commits that haven't been brought into
  origin yet — those need review on each sync pass.

Peek the upstream-only changes any time:

```bash
git fetch upstream
git log --oneline origin/main..upstream/main
```

## Sync procedure — upstream → origin

For each upstream release / quarterly refresh:

```bash
# 1. Fetch upstream and see the delta.
git fetch upstream
git log --oneline origin/main..upstream/main

# 2. Start a local sync branch off main.
git fetch origin
git checkout -B sync/upstream-YYYYMMDD origin/main

# 3. Merge upstream (preserves origin's linear history, writes one
#    merge commit). Rebase is an option but with 195 carrying commits
#    the merge route is saner.
git merge upstream/main
# (or rebase, if the delta is small and you're willing to resolve
# conflicts per-commit instead of per-file.)

# 4. Resolve conflicts. Hotspot files (based on current carrying
#    patches):
#    - Libraries/MLXLLM/Models/Gemma4Text.swift
#    - Libraries/MLXVLM/Models/Gemma4.swift
#    - Libraries/MLXLLM/Models/Qwen35.swift / Qwen3Next.swift
#    - Libraries/MLXLMCommon/ModelConfiguration.swift
#    - Libraries/MLXLMCommon/Tool/ToolCallFormat.swift
#    - Libraries/MLXLMCommon/Evaluate.swift
#    - Libraries/MLXLMCommon/BatchEngine/*  (vmlx-only; conflicts
#      only if upstream introduces a same-named thing)

# 5. Verify.
swift build
swift test --skip-build --filter 'BatchKVCacheRotatingSlot'
swift test --skip-build --filter 'StopStringMatcher'
swift test --skip-build --filter 'ReasoningParser'
swift test --skip-build --filter 'Tool-Call Edge Cases'
# Real-model smoke — against a Gemma-4 model with prompt > 1024
# tokens to hit the SWA crash regression gate:
pkill -f xctest; pkill -f RunBench; pkill -f ollama; pkill -f lms
VMLX_CHAT_TEMPLATE_OVERRIDE=$PWD/Libraries/MLXLMCommon/ChatTemplates/Gemma4Minimal.jinja \
  BENCH_OSAURUS_MULTITURN=1 BENCH_OSAURUS_SIZE=medium \
  BENCH_MODEL=~/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit \
  BENCH_MAX_TOKENS=40 \
  ./.build/release/RunBench

# 6. Push to origin.
git push origin sync/upstream-YYYYMMDD:main
```

## Upstream PR candidates

The current vmlx tree contains a handful of fixes that are clean
bug-fixes against upstream code, not vmlx-only additions. These
could land upstream as discrete PRs, shrinking our carrying diff:

1. **"Fix float16 overflow in JANG MLP / Qwen3.5 / Gemma3 / MiniMax"**
   — `800e68c`, `57fe2e5`. silu(gate) × up exceeds 65504.
2. **"Fix Gemma4 VLM image pipeline — sRGB / multi-image / sanitize"**
   — `1ddabd7`, `bd01662`, `c4d698c`, `2671c4c`, `534e427`, `e47259c`,
   `833bbf2`. May overlap with upstream's own Gemma4 VLM PR (#180).
3. **"Fix Gemma4 / Gemma3n multi-turn 1D-token crash"** — `285a736`,
   `7917108`. Small, clean, obvious bug.
4. **"Skip SwitchGLU compiledGeluApproximate crash on MLXNN Power"** —
   `0db30fb`, `b59586f`.

Submit via:

```bash
gh pr create --repo ml-explore/mlx-swift-lm \
  --title "Fix X" --body-file pr-body.md
```

vmlx-only additions (`BatchEngine`, `SpecDec`, `CacheCoordinator`,
`TurboQuant`, `JANG loader`, `.reasoning` event, `extraStopStrings`)
are deliberately scoped to vmlx and not candidates for upstream.

## Acceptance gate before pushing main

- `swift build` green.
- Regression suites green: `BatchKVCacheRotatingSlot` (4),
  `StopStringMatcher` (14), `ReasoningParser` (37), `Tool-Call Edge
  Cases` (24), existing `SpecDec` suites (90), `BatchKVCache` + `BatchCausalMask`.
- Real-model smoke — at least one of:
  - `~/.mlxstudio/models/MLXModels/OsaurusAI/gemma-4-e2b-it-4bit`
    (Gemma-4 SWA, prompt > 1024 tokens — regression gate for the
    2026-04-20 broadcast_shapes crash).
  - `~/.mlxstudio/models/MLXModels/OsaurusAI/Qwen3.6-35B-A3B-MXFP4`
    (reasoning emission + tool-call format wiring).

## Ownership

Eric owns the sync. Target a quarterly cadence with ad-hoc syncs
when upstream lands a critical fix (crash class, new model family
osaurus wants).
