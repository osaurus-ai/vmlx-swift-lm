# Ralph task — reasoning channel bugs (Gemma-4 + Qwen 3.6)

**Branch:** `fix/gemma4-harmony-reasoning` (off main @ 9cc7efe)
**Scope:** two related bugs in the new `.reasoning(String)` pipeline,
both reported by tpae on 2026-04-20 afternoon.

## Bug A — Gemma-4 harmony markers leak into `.chunk`

tpae screenshot 2026-04-20 2:59 PM shows Gemma-4-26B-A4B-mxfp4
emitting harmony-channel markers (opener like pipe-channel-pipe
followed by channel name `thought` + newline, closer like
channel-pipe) directly in `.chunk(String)` output.

tpae's verbatim note:
> formatting is fine just need to call the right events.
> <|channel|> is thinking tag, right

**Why it's broken:**
1. `LLMModelFactory._load` and `VLMModelFactory._load` heuristic
   stamps `reasoningParserName = "none"` for any model whose
   `modelType` contains `gemma` or `mistral` (see
   `LLMModelFactory.swift` ~L738, `VLMModelFactory.swift` ~L387).
2. `ReasoningParser.fromCapabilityName("gemma4")` returns `nil`
   (`ReasoningParser.swift:181`).
3. So `BatchEngine.generate` never constructs a parser for Gemma-4,
   and harmony bytes pass through to `.chunk`.

## Bug B — Qwen3.6 reasoning bleeds into `.chunk`

tpae screenshot 2026-04-20 (second) shows Qwen3.6-35B-A3B-MXFP4
emitting:

> The user wants a README for their game. I should first read the
> game file to understand what it is, then generate an appropriate
> R</think>
>
> Let me take a look at your game f

Parser IS wired on Qwen3.6 (`reasoningParserName = "think_xml"`),
but output starts **inside** the `<think>` block — the Qwen 3.6
chat template with `enable_thinking=true` appends `<think>\n` as
the last tokens of the prompt, so the model's generated output
starts AFTER the opening tag. Only `</think>` appears in the
stream.

**Why it's broken:** `ReasoningParser` starts in `.content` state
by default. When the first byte of model output is already
reasoning (no opening tag to latch onto), every byte up to
`</think>` is emitted as `.chunk` — the reasoning text leaks into
the visible answer.

## Stop condition

1. Real-model Gemma-4 run: `.chunk` contains zero harmony markers,
   and at least one `.reasoning(String)` delta fires.
2. Real-model Qwen3.6 run (enable_thinking): the pre-`</think>`
   reasoning text goes out as `.reasoning(String)` events, NOT
   `.chunk`. `.chunk` begins only with the final answer after
   `</think>`.
3. `ReasoningParser`, `BatchKVCacheRotatingSlot`, `StopStringMatcher`,
   `Tool-Call Edge Cases` suites stay green.
4. Fast-forward-merged to `main` + pushed to `origin/main`.
5. Emit promise `HARMONY_REASONING_FIXED`.

## Shape of the fix

### For Bug A (Gemma-4 harmony)

1. Diagnose exact tags by reading the Gemma-4 model's
   `tokenizer_config.json`, `chat_template.jinja`,
   `generation_config.json` under
   `~/.mlxstudio/models/MLXModels/mlx-community/gemma-4-26b-a4b-it-4bit/`.
2. Add a capability stamp (e.g. `harmony`) + a `ReasoningParser`
   instance wired with the right start/end tag strings (or a new
   `HarmonyChannelParser` if multi-channel routing is needed —
   thought / analysis / final channels map to reasoning / reasoning /
   content respectively).
3. Flip the factory heuristic: Gemma-4 defaults to the new stamp
   instead of `"none"`.

### For Bug B (Qwen3.6 prefilled `<think>`)

Add a `startInReasoning: Bool = false` parameter to
`ReasoningParser.init`. When true, the parser starts in `.reasoning`
state — every byte goes to `.reasoning` events until the first
`</think>` closes the block and switches to `.content`. The bytes
*before* the closer are reasoning; they must not leak.

Wire the flag:
- `ReasoningParser.fromCapabilityName` returns a parser with
  `startInReasoning: true` for capability stamps that correspond to
  chat templates that prefill `<think>\n` at the prompt tail. Qwen
  3.x family is the main case.
- Alternatively: expose a new `GenerateParameters.promptEndsInThinkingMode: Bool`
  flag that `BatchEngine.generate` / `Evaluate.generate` consult
  when constructing the parser. Caller sets it to true when they
  built the prompt with `applyChatTemplate(additionalContext:
  ["enable_thinking": true])`.

Preference: the second option (caller-declared) because it matches
actual wire reality per-request — some callers set enable_thinking
per turn, some don't. A capability stamp is too coarse.

### Tests

New @Suite entries in `Tests/MLXLMTests/ReasoningParserTests.swift`:
- Harmony parser: streaming, multi-block, unclosed on EOS flushes
  to `.reasoning`, channel-name variants.
- `startInReasoning=true` parser: first-byte-is-reasoning case,
  closer splits, unclosed on EOS flushes entire buffer to reasoning.

Existing suites must remain green:
`ReasoningParser` (37), `BatchKVCacheRotatingSlot` (4),
`StopStringMatcher` (14), `Tool-Call Edge Cases` (24), `BatchKVCache`,
`BatchCausalMask`.

### Docs

- `REASONING-STREAM-EVENT.md` — family table row for Gemma-4 flips
  from "none" to new stamp; add an "enable_thinking / prefilled
  `<think>`" subsection covering Bug B.
- `TPAE-2026-04-20-TRIAGE.md` — add 2:59 PM + afternoon-2 addendum
  rows mapping both screenshots to commits + regression tests.
- `OSAURUS-INTEGRATION.md` — doc map still accurate; cross-reference
  new test names.

### Commit per logical step

1. `diagnose`: capture byte sequences + Qwen chat template inspection
   into commit message body.
2. `fix(reasoning): add startInReasoning for Qwen prefilled think tag`
3. `fix(reasoning): add harmony channel parser + Gemma-4 stamp`
4. `test(reasoning): regression suites for both bugs`
5. `docs: update REASONING-STREAM-EVENT + TPAE-triage for both bugs`

No AI attribution in commit messages. Fast-forward-merge at the end.
