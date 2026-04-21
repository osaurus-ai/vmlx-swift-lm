# Ralph edge-case audit — state log

Machine-readable progress table for the edge-case audit driven by
`RALPH-EDGE-TASK.md`. Each iteration appends one row after
completing an item. Do NOT edit older rows.

| # | Item | Iter | Outcome | Commit | Notes |
|---|---|---|---|---|---|
| - | init | 0 | baseline | `fa0f9b5` | 90+ tests green, harmony bare `<\|channel>` start tag shipped |
| A1 | empty channel body | 1 | test-only ✓ | `23e9ae7` | Parser already correct — `<\|channel><channel\|>` yields empty reasoning + adjacent content. |
| A2 | nested opener inside reasoning | 1 | test-only ✓ | `23e9ae7` | Toggle state-machine keeps literal `<\|channel>` as bytes until first `<channel\|>` closes. |
| A3 | orphan closer before opener | 1 | test-only ✓ | `23e9ae7` | Closer without prior opener is literal content (state never flips). |
| A4 | closer split across feeds | 1 | test-only ✓ | `23e9ae7` | Holdback (`max(startTag,endTag).count - 1`) holds the partial tag. |
| A5 | opener split across feeds | 1 | test-only ✓ | `23e9ae7` | Same holdback mechanism works symmetrically. |
| A7 | truncated mid-opener | 1 | test-only ✓ | `23e9ae7` | Partial opener bytes remain as content (state never flipped). |
| A8 | truncated mid-closer | 1 | test-only ✓ | `23e9ae7` | Inside-state flush emits held bytes as reasoning. |
| B1 | enable_thinking=false with think_xml stamp | 1 | **fixed** ✓ | `23e9ae7` | New `ReasoningParser.forPrompt(stampName:promptTail:)` auto-detects. Plumbed into Evaluate.generate + BatchEngine.generate + SpecDecStream.streamViaStrategy via `_decodePromptTail` helper. |
| B2 | interleaved thinking | 1 | test-only ✓ | `23e9ae7` | Toggle state-machine handles multiple `<think>...</think>` blocks mid-response. |
| B4 | partial `</think>` at EOS | 1 | test-only ✓ | `23e9ae7` | Inside-state flush emits held bytes as reasoning. |
| B5 | entire output is reasoning | 1 | test-only ✓ | `23e9ae7` | With startInReasoning=true and no closer, flush drains all to reasoning. |
| C1 | 55K-token translation OOM | 2 | **shipped** ✓ | `35820ba` | Coordinator-owned KV sizing contract (`KV-SIZING-CONTRACT.md`). `CacheCoordinatorConfig.defaultKVMode` + `.defaultMaxKVSize` fill per-request gaps at `BatchEngine.admitPendingRequests`. 10 unit tests cover all resolution paths. |
