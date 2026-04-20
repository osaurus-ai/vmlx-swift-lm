# Ralph audit — reasoning/harmony/tool-call edge cases

**Branch:** `fix/reasoning-edge-audit` (off main @ `fa0f9b5`)
**Goal:** systematically harden the reasoning-parser + tool-call + harmony pipeline by working through every plausible edge case tpae's 2026-04-20 session or the model wire formats can produce. Per iteration: pick the NEXT unresolved item, reproduce, fix (or document as out-of-scope), commit. Finish with FF-merge to main + push.

**Stop condition:** all items in §Checklist marked ✓ (fixed) or ⊘ (explicitly out-of-scope with rationale), 90+ tests green, real-model smoke on Gemma-4-26B and Qwen3.6-35B clean, main pushed. Emit promise `EDGE_AUDIT_DONE`.

**Hard constraints:**
- Run model tests ONE AT A TIME (GPU RAM).
- Pre-flight `pkill -f xctest; pkill -f RunBench; pkill -f ollama; pkill -f lms` before any real-model run.
- Do NOT regress: `ReasoningParser`, `HarmonyParserStreaming`, `startInReasoning=true`, `ToolCallEdgeCases`, `BatchKVCacheRotatingSlot`, `StopStringMatcher`, `BatchKVCache`, `BatchCausalMask`, `Generation.reasoning event` — all must stay green.
- No AI attribution in commits.
- No force-push to main.

## Checklist (process top-to-bottom, one commit per item)

### Category A — harmony (Gemma-4) robustness

- [ ] **A1. Empty channel body** — model emits `<|channel><channel|>` (zero bytes inside). Reasoning delta should be empty string (but .reasoning event may or may not fire). Content should have nothing leaked. Write unit test.
- [ ] **A2. Channel body containing `<|channel>` substring** — model emits `<|channel>hi<|channel>nested<channel|>`. State machine should NOT re-enter; first `<channel|>` closes. Write unit test covering both behaviors and pick the safer one.
- [ ] **A3. `<channel|>` appearing in content before opener** — model emits `foo<channel|>bar<|channel>baz<channel|>qux`. Closer before opener should be literal content (no state flip). Write test.
- [ ] **A4. Closer split across `feed` calls** — opener arrives then closer split into `<chan` + `nel|>`. Holdback must keep the tail so no reasoning leaks as content. Write test.
- [ ] **A5. Opener split across feed calls** — `<|cha` then `nnel>thought\ninner<channel|>`. Parser must hold back safely and latch. Write test.
- [ ] **A6. `<|channel>` appearing literally in final answer (e.g., model explaining the format)** — this is an intentional mention. Document as a known limitation (we route to reasoning). Consider adding a "quoted" opt-out via GenerateParameters, but for now document.
- [ ] **A7. maxTokens hits mid-opener** — model started emitting `<|chan` and got truncated. Flush should emit what's held as content (since state never flipped to reasoning). Write test.
- [ ] **A8. maxTokens hits mid-closer inside reasoning** — `<|channel>thought\nlots of text<chann` truncated. Flush should emit the held bytes as reasoning (current `flush()` does). Confirm.
- [ ] **A9. Model re-emits harmony in a follow-up turn** — turn 2's prompt INCLUDES turn 1's `<|channel>thought\n…<channel|>` rendered by the chat template. Does BatchEngine accidentally feed those prompt bytes through the parser? (It shouldn't — parser runs on detokenized OUTPUT only.) Verify with a real-model multi-turn smoke.
- [ ] **A10. JANG-stamped Gemma-4 bundles** — check that `capabilities.reasoning_parser = "gemma4"` in jang_config.json resolves to the harmony parser.

### Category B — `<think>` family + Qwen3.x prefilled

- [ ] **B1. `enable_thinking=false` with think_xml stamp** — Qwen3.6 template prefills `<think>\n\n</think>\n\n` when `enable_thinking=false`. Model output starts in content mode (opener + closer already in PROMPT, not output). With startInReasoning=true default, the parser would treat all output as reasoning until seeing a `</think>` that never arrives, and FLUSH it as reasoning. **Bug.** Fix: either expose `GenerateParameters.promptEndsInReasoning: Bool` OR sniff the chat template to detect. Conservative: caller-provided field, defaulting based on stamp.
- [ ] **B2. Interleaved thinking (Qwen3.6 mid-response `<think>…</think>`)** — some Qwen 3.6 responses emit multiple `<think>` blocks interleaved with content. With startInReasoning=true + standard state machine toggle, should work (first closer → content, next opener → reasoning again, etc.). Unit test explicitly.
- [ ] **B3. `</think>` appearing in content** — same as A6 but for think-family. Known limitation, document.
- [ ] **B4. Partial `<think>` closer at EOS** — `<think>…<chann` truncated. Current flush: emits held bytes as reasoning. Confirm by test.
- [ ] **B5. No `</think>` in output despite startInReasoning** — model emits pure reasoning until max tokens. Current behavior: everything → reasoning, `.chunk` empty. Confirm.

### Category C — cross-path consistency

- [ ] **C1. BatchEngine vs Evaluate — byte-identical parser state** — verify both paths construct parser with identical args for the same ModelConfiguration. Grep for `ReasoningParser.fromCapabilityName(` call sites, diff.
- [ ] **C2. SpecDec (DFlash / DDTree) path** — `SpecDecStream.streamDflashLinear` / `streamDDTree` also construct a parser; confirm they use the same stamp and startInReasoning semantics.
- [ ] **C3. Per-request isolation in BatchEngine** — two concurrent requests (B=2) with different reasoningParserNames — does each request's parser stay isolated? Write a unit test using the Task-isolation harness.
- [ ] **C4. Parser resets across turns in multi-turn** — each `engine.generate(...)` call must build a fresh parser. Any hidden retained state would cross-contaminate. Grep + test.

### Category D — tool calls × reasoning

- [ ] **D1. Tool call INSIDE a reasoning block** — `<|channel>thought\nfoo<|tool_call>call<tool_call|>bar<channel|>`. Current pipeline: reasoning parser strips the outer envelope first, so the `<|tool_call>` bytes go to... wait, it strips THEM as reasoning content. Then the tool-call processor never sees them. **Bug surface.** Fix: tool-call processor should run INSIDE reasoning too, OR harmony channels that are "tool" / "action" should route to tool-call path. Decide + implement.
- [ ] **D2. Harmony ReAct-style JSON action block as `.toolCall`** — tpae's screenshot shows `<|channel> {"action": "google_search", "action_input": "..."}<channel|>`. Currently routes to `.reasoning` but the user's intent is to invoke a tool. Add a "harmony action extractor" that detects JSON-shaped reasoning content and emits `.toolCall` events. This is a real feature tpae asked for indirectly.
- [ ] **D3. Tool call AFTER reasoning block** — `<|channel>thought\nanalyzing<channel|><|tool_call>call_name<tool_call|>` — standard flow, should work. Confirm.
- [ ] **D4. Gemma-4 `<|tool_call>` envelope** — separate from harmony channel. Verify `ToolCallFormat.gemma4` still extracts correctly when reasoning parser is active.

### Category E — model-family coverage

- [ ] **E1. Mistral / Gemma 3 / LFM2 regression** — these have stamp=none. Confirm NO `.reasoning` events ever fire + no regression in output.
- [ ] **E2. GLM 4.x / DeepSeek-R1 / Kimi K2 / MiniMax M2 / Nemotron** — startInReasoning=true applies to all of them (via `fromCapabilityName`). Verify with a real-model smoke on at least one of them (whichever is locally available).
- [ ] **E3. gpt-oss (harmony)** — does gpt-oss use the same `<|channel>` envelope? If so, check that its stamp resolves to the harmony parser. If not, document the difference.

### Category F — documentation accuracy

- [ ] **F1. `ModelConfiguration.reasoningParserName` doc-comment** — enumerate all accepted stamps with envelope semantics. Already partially done.
- [ ] **F2. `REASONING-STREAM-EVENT.md`** — ensure family table matches what `fromCapabilityName` actually returns.
- [ ] **F3. `OSAURUS-API-SURFACE.md`** — Generation enum row matches reality. Check the Generation exhaustive example for `.reasoning` case.
- [ ] **F4. `TPAE-2026-04-20-TRIAGE.md`** — add any new edge-case rows discovered.
- [ ] **F5. Regression-test index** — each new test goes in the triage doc's "Test coverage added" section.

## Per-iteration workflow

For each unchecked item above:

1. **Understand.** Read relevant source + any existing tests.
2. **Reproduce.** Write a FAILING unit test demonstrating the bug. If the test passes already, the item is already handled — mark ✓ with "no action needed" note. Commit nothing.
3. **Fix.** Minimal edit to make the test pass. No drive-by refactors.
4. **Verify.** Run the full regression suite (`ReasoningParser|HarmonyParserStreaming|StartInReasoning|ToolCallEdgeCases|BatchKVCacheRotatingSlot|StopStringMatcher|Generation.reasoning event|BatchKVCache|BatchCausalMask|ToolCallFormat capability`). If ANY regression, revert and rethink.
5. **Commit.** One commit per item. Message format: `fix(reasoning): <item-id> <short desc>` or `test(reasoning): <item-id> <short desc>` if no source change needed.
6. **Mark.** Check the box in this doc; add a brief summary of what changed.

When all items are ✓ or ⊘:
- Run the two real-model smokes (`BENCH_HARMONY_CHECK` on Gemma-4-26B, `BENCH_QWEN_THINKING_CHECK` on Qwen3.6-35B). Confirm zero envelope markers in `.chunk`.
- Run the 3-turn `BENCH_QWEN_MULTITURN_TOOL` on both models.
- Final FF-merge to main, push.
- Emit `<promise>EDGE_AUDIT_DONE</promise>`.

## State file

A machine-readable state file is kept at
`Libraries/MLXLMCommon/BatchEngine/RALPH-EDGE-CASE-STATE.md`. Each
iteration MUST append a row to the state table noting which item was
processed, outcome (fixed / already-handled / out-of-scope), and the
commit SHA.
