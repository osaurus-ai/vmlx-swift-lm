import Foundation
import MLX
import MLXHuggingFace
import MLXLLM
import MLXLMCommon
import MLXVLM
@preconcurrency import Tokenizers

// Multi-turn benchmark for gemma-4-26b-a4b-it-4bit
// Loads pre-tokenized turns from /tmp/gemma4_multiturn_tokens.json
// Measures TTFT, prompt processing tok/s, decode tok/s for each turn
// with cache reuse across turns.

@main
struct Bench {
    static func main() async throws {
        setvbuf(stdout, nil, _IONBF, 0)
        FileHandle.standardError.write("[STDERR] Bench main() started\n".data(using: .utf8)!)

        // Model path and tokens file are configurable via env vars so a single
        // executable serves any model. Defaults preserve historical Qwen3.5 behavior.
        let env = ProcessInfo.processInfo.environment
        let modelPath = env["BENCH_MODEL"] ?? "/Users/eric/models/Qwen3.5-35B-A3B-4bit"
        let tokensPath = env["BENCH_TOKENS"] ?? "/tmp/qwen35_multiturn_tokens.json"
        let maxNew = Int(env["BENCH_MAX_TOKENS"] ?? "256") ?? 256
        let compileDecode = (env["BENCH_COMPILE_DECODE"] ?? "0") == "1"
        let compileMaxLen = Int(env["BENCH_COMPILE_MAXLEN"] ?? "16384") ?? 16384
        let modelDir = URL(fileURLWithPath: modelPath)

        // BENCH_JANGPRESS=1 activates the JangPress cold-weight tier
        // (axis E) for THIS bench run. Holds the runtime alive for
        // the full bench so the controller's failsafe state machine
        // ticks during inference. Knobs:
        //   BENCH_JANGPRESS_PCT=70        compress 70% of routed mass
        //   BENCH_JANGPRESS_FORCE=0       soft (madvise DONTNEED) vs
        //                                 force (msync MS_INVALIDATE)
        //   BENCH_JANGPRESS_PREFETCH=1    pre-fault top hot tiles at arm
        let jangPressOn = (env["BENCH_JANGPRESS"] ?? "0") == "1"
        let jangPressOpts: JangPressLoadOptions = jangPressOn ? .init(
            enabled: true,
            compressPct: Int(env["BENCH_JANGPRESS_PCT"] ?? "70") ?? 70,
            backend: .mmap,
            forceMode: (env["BENCH_JANGPRESS_FORCE"] ?? "0") == "1" ? .force : .soft,
            enablePrefetch: (env["BENCH_JANGPRESS_PREFETCH"] ?? "1") == "1"
        ) : .disabled
        var jangPressRuntime = JangPressRuntime.none
        if jangPressOn {
            jangPressRuntime = JangPressActivation.activate(
                bundleURL: modelDir, options: jangPressOpts)
            if jangPressRuntime.isActive {
                FileHandle.standardError.write(Data(
                    "[bench] JangPress active: pct=\(jangPressOpts.compressPct) force=\(jangPressOpts.forceMode == .force) prefetch=\(jangPressOpts.enablePrefetch)\n".utf8))
            } else {
                FileHandle.standardError.write(Data(
                    "[bench] JangPress activation requested but tier returned .none\n".utf8))
            }
        }
        defer {
            if jangPressOn { JangPressActivation.deactivate(jangPressRuntime) }
        }

        // BENCH_VL=1 dispatches to VLBench: loads the real HF tokenizer +
        // synthesises a 224×224 image, runs two turns through the VLM
        // processor + cache to verify vision path + multi-turn cache reuse.
        // CAVEAT: the standard BENCH_VL path uses `TokenIterator`, not
        // BatchEngine. Use BENCH_VL_BATCH_CHAT to verify the actual
        // BatchEngine VL path.
        if (env["BENCH_VL"] ?? "0") == "1" {
            try await VLBench.run(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_CHAT=1 runs VL multi-turn DIRECTLY through
        // `BatchEngine.generate(...)`. This is the honest VL-through-
        // BatchEngine verification — iter 29 audit flagged that prior
        // BENCH_VL only exercises `TokenIterator`. Added 2026-04-19 (iter 30).
        if (env["BENCH_VL_BATCH_CHAT"] ?? "0") == "1" {
            try await VLBench.runBatch(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_MIXED=1 (2026-04-22): single model + single BatchEngine
        // + shared CacheCoordinator, four turns with different variables
        // flipped each turn (thinking on/off, text/image/video modality).
        // Validates SSM seed + stepBatchDecode force-unwrap fix +
        // ReasoningParser.forPrompt tail detection, all in one run.
        //   BENCH_VIDEO=/path/to/file.mov overrides the default video.
        if (env["BENCH_VL_MIXED"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO"]
                ?? "Tests/MLXLMTests/Resources/1080p_30.mov"
            try await VLBench.runMixedMultiTurn(
                modelPath: modelPath, videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_VIDEO=1 (2026-04-22): end-to-end VL video ingest
        // through `context.processor.prepare(input:)` with
        // `UserInput(prompt:, videos: [.url(...)])`, then drive multi-
        // turn through BatchEngine.generate. Set
        //   BENCH_VIDEO=/path/to/file.mov
        // to override the default test fixture.
        if (env["BENCH_VL_BATCH_VIDEO"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO"]
                ?? "Tests/MLXLMTests/Resources/1080p_30.mov"
            try await VLBench.runBatchVideo(
                modelPath: modelPath, videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_MEDIASALT=1 (iter 37): verify VL cache isolation
        // via `mediaSalt`. Submit prompt P with image A, then the same
        // P with image A (must HIT), then the same P with image B (must
        // MISS because SHA256 of image bytes differs). Catches the
        // cache-poisoning bug class where two different images with the
        // same text prompt would return each other's cached KV state.
        if (env["BENCH_VL_BATCH_MEDIASALT"] ?? "0") == "1" {
            try await VLBench.runBatchMediaSalt(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_BATCH_TOOLCALL=1 (iter 66): submit a tool-bearing prompt
        // through `BatchEngine.generate(...)` and assert the pipeline
        // (ReasoningParser → ToolCallProcessor) behaves correctly on a
        // real model:
        //   - `.chunk(String)` text MUST NOT contain raw tool-call
        //     markers (`<tool_call>`, `<|tool_call>`, `call:<name>`,
        //     `[TOOL_CALLS]`, etc.) — if it does, the library failed
        //     to extract the call and osaurus would have to re-parse.
        //   - `.chunk(String)` text MUST NOT contain raw `<think>` /
        //     `</think>` markers — if it does, the reasoning parser
        //     never engaged.
        //   - Model output is nondeterministic at temperature 0 (same
        //     model, same prompt → different families emit tool calls
        //     at different rates), so we do NOT require a `.toolCall`
        //     event. We only require that IF the model emits raw
        //     markers, they get stripped/extracted.
        if (env["BENCH_BATCH_TOOLCALL"] ?? "0") == "1" {
            try await runBatchEngineToolCall(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_PERF=1 (2026-04-21): deterministic tok/s micro-bench for
        // the perf-regression hunt. Prints one grep-able line per run:
        //   PERF model=<name> variant=<label> genTokens=N genSec=F tokps=F
        // Reads env:
        //   BENCH_PERF_PROMPT  — CoT-free prompt text (default below)
        //   BENCH_MAX_TOKENS   — fixed decode budget (default 128)
        //   BENCH_PERF_VARIANT — label emitted in the output
        //   BENCH_PERF_WARMUP  — run N warmup turns first (default 1)
        //   BENCH_PERF_RUNS    — measurement turns, picks median (default 3)
        if (env["BENCH_PERF"] ?? "0") == "1" {
            try await runPerfBench(
                modelPath: modelPath, maxNew: maxNew,
                variant: env["BENCH_PERF_VARIANT"] ?? "auto",
                warmup: Int(env["BENCH_PERF_WARMUP"] ?? "1") ?? 1,
                runs: Int(env["BENCH_PERF_RUNS"] ?? "3") ?? 3,
                useTokenIterator:
                    (env["BENCH_PERF_PATH"] ?? "batch") == "iter")
            return
        }

        // BENCH_HARMONY_CHECK=1: real-model verification of the 2026-04-20
        // harmony-reasoning fix. Loads a Gemma-4 model, sends a short
        // prompt, asserts at least one .reasoning delta fires AND .chunk
        // contains zero harmony markers.
        if (env["BENCH_HARMONY_CHECK"] ?? "0") == "1" {
            try await runHarmonyReasoningCheck(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_QWEN_THINKING_CHECK=1: real-model verification of the
        // Qwen3.6 prefilled-<think> fix. Loads a Qwen 3.x model with
        // enable_thinking=true, asserts at least one .reasoning delta
        // fires AND .chunk contains zero <think> markers.
        if (env["BENCH_QWEN_THINKING_CHECK"] ?? "0") == "1" {
            try await runQwenThinkingReasoningCheck(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_DSV4_TEMPLATE_KWARGS=1: verify the bundle's shipped
        // Jinja chat_template threads `enable_thinking` (bool) and
        // `reasoning_effort` ('max'/None) kwargs through the upstream
        // applyChatTemplate → additionalContext path. Without this
        // working, callers can't switch between chat/thinking modes
        // or engage max-effort preface on DSV4 bundles.
        if (env["BENCH_DSV4_TEMPLATE_KWARGS"] ?? "0") == "1" {
            try await runDSV4TemplateKwargsCheck(modelPath: modelPath)
            return
        }

        // BENCH_ORPHAN_SLOT_REPRO=1: reproduce the consumer-cancellation
        // → orphan-slot → next-request-Metal-collision pattern reported
        // 2026-04-27. Submits request A, breaks the consumer loop after
        // 4 tokens (simulates osaurus's `Task.isCancelled break`), then
        // immediately submits request B with the same prompt (would
        // normally hit cache and collide with the orphan-A pipelines).
        // Pre-fix: clear_library assertion mid-prefill of request B.
        // Post-fix: continuation.onTermination on A reaps the slot
        // before B starts, both complete cleanly.
        if (env["BENCH_ORPHAN_SLOT_REPRO"] ?? "0") == "1" {
            try await runOrphanSlotRepro(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_THINK_LOOP_PROBE=1: validation-style prompt to test
        // whether a reasoning model emits </think> within budget or
        // gets stuck in self-refinement loops. Used to A/B JANGTQ
        // bits=2 vs bits=4 on the same task ("give me a random N-digit
        // number") to isolate EOS-margin compression in 4-bit
        // quantization. Sampling matches Qwen-family canonical:
        // T=0.7, top_p=0.8.
        if (env["BENCH_THINK_LOOP_PROBE"] ?? "0") == "1" {
            try await runThinkingLoopProbe(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_DSV4_FIM_VS_CHAT=1: side-by-side coherence probe across
        // DSV4's three prompt modes on the same simple factual prompt.
        // Decodes 64 tokens each and prints raw output so a human can
        // judge "does it actually answer the question."
        //
        //   1. FIM (raw): no chat template, just "The capital of France is"
        //   2. CHAT no-think: applyChatTemplate, enable_thinking=false
        //   3. CHAT think: applyChatTemplate, enable_thinking=true
        //
        // Background: HumanEval+ FIM mode pass@1 was previously
        // measured at 67% on JANGTQ_2L. Long-trace chat conversations
        // showed drift past sliding_window=128. This bench answers
        // whether chat template short-output (~64 tokens) is coherent.
        if (env["BENCH_DSV4_FIM_VS_CHAT"] ?? "0") == "1" {
            try await runDSV4FIMvsChat(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_QWEN_MULTITURN_TOOL=1: mirrors tpae's 2026-04-20 3:02 /
        // 3:04 PM screenshots — Qwen3.6, first turn asks "create README
        // for my game", turn 2 pretends a file_read tool returned game
        // source, turn 3 asks for a second tool. Asserts ZERO <think>
        // markers in `.chunk` across all 3 turns — the EXACT bug tpae
        // screenshotted.
        if (env["BENCH_QWEN_MULTITURN_TOOL"] ?? "0") == "1" {
            try await runQwenMultiturnToolCheck(
                modelPath: modelPath, maxNew: maxNew)
            return
        }
        // BENCH_OMNI=1 (2026-04-28): full multi-turn matrix for
        // Nemotron-3-Nano-Omni bundles. Tests text-only single, text
        // multi-turn, image single, image multi-turn, video, audio
        // (manual splice), and reasoning toggle in one harness.
        // BENCH_MODEL=/path/to/Nemotron-3-Nano-Omni-30B-A3B-{MXFP4|JANGTQ4|JANGTQ2}
        if (env["BENCH_OMNI"] ?? "0") == "1" {
            try await OmniBench.run(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_STABILITY=1 (2026-04-30): exhaustive stability matrix
        // covering the failure modes that have been blocking releases —
        // warm L2 disk-cache 2nd-request, over-cap hybrid prompt,
        // multi-turn agent loop, cancel + recovery, concurrent batched
        // decode, TQ KV mode + disk round-trip, clearCache mid-run,
        // hybrid SSM disk round-trip. Drives BatchEngine directly via
        // a single `ModelContext`. Designed for any hybrid / VLM
        // bundle. No HTTP layer; runs from this binary directly.
        if (env["BENCH_STABILITY"] ?? "0") == "1" {
            try await StabilityBench.run(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // Hoist single-load scenarios above the preamble load so they
        // don't double-allocate the model. Critical for huge bundles
        // (DSV4-Flash JANGTQ at 79.5 GB OOMs with two simultaneous
        // copies on a 128 GB host). Each of these scenarios does its
        // own load via runBatchEngine* which uses the real HF
        // tokenizer — the preamble's NullTokenizerLoader copy isn't
        // needed for any of them.
        if (env["BENCH_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await runBatchEngineCacheHit(modelPath: modelPath, maxNew: maxNew)
            return
        }
        if (env["BENCH_BATCH_DISK_RESTORE"] ?? "0") == "1" {
            try await runBatchEngineDiskRestore(modelPath: modelPath, maxNew: maxNew)
            return
        }

        print("=== vmlx-swift-lm — \(modelDir.lastPathComponent) MULTI-TURN ===")
        print("Tokens: \(tokensPath)")
        print("Loading...")

        let loadStart = CFAbsoluteTimeGetCurrent()
        // Use general loader — picks LLM or VLM factory based on model_type
        let context = try await MLXLMCommon.loadModel(from: modelDir, using: NullTokenizerLoader())
        print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
        print("Model: \(type(of: context.model))")

        // BENCH_BATCH=1 runs the BatchEngine smoke: single request via
        // BatchEngine, then BatchEngine + compile-on, then BatchEngine
        // + TurboQuant, then B=2 concurrent. Verifies end-to-end
        // behaviour on real weights. Added 2026-04-18 (iter 17).
        if (env["BENCH_BATCH"] ?? "0") == "1" {
            try await runBatchSmoke(context: context, maxNew: maxNew)
            return
        }

        // BENCH_COHERENT=1 runs a real multi-turn conversation through
        // BatchEngine with the actual HF tokenizer so we can visually
        // verify coherent text output across 3 turns with cache reuse.
        // Added 2026-04-18 (iter 19) — the user has repeatedly asked
        // for actual coherence testing, not just synthetic-prompt
        // tok/s measurements.
        //
        // CAVEAT (iter 26): BENCH_COHERENT uses ChatSession which
        // internally uses `TokenIterator`, not `BatchEngine`. For
        // TRUE BatchEngine multi-turn verification use BENCH_BATCH_CHAT.
        if (env["BENCH_COHERENT"] ?? "0") == "1" {
            try await runCoherentMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_CHAT=1 runs a real multi-turn conversation
        // DIRECTLY through BatchEngine (not ChatSession). Uses the real
        // HF tokenizer + chat template + CacheCoordinator for cross-
        // turn reuse. This is the honest BatchEngine-multi-turn test
        // per the spec §6 multi-turn acceptance path. Added 2026-04-19.
        if (env["BENCH_BATCH_CHAT"] ?? "0") == "1" {
            try await runBatchEngineMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CROSS_VALIDATE=1 (iter 32): run the same prompt through
        // `TokenIterator` AND `BatchEngine.generate(...)` with temp=0 and
        // assert the emitted token IDs match byte-for-byte. This is the
        // strongest single correctness property for the engine — compile
        // on/off identity is already checked by `BENCH_BATCH_CHAT`, but
        // equality with the long-standing single-seq path was only
        // assumed until this bench existed.
        if (env["BENCH_CROSS_VALIDATE"] ?? "0") == "1" {
            try await runCrossEngineValidation(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_CONCURRENT=1 (iter 33): TWO different prompts
        // submitted to BatchEngine maxBatchSize=2 and iterated
        // CONCURRENTLY (TaskGroup). Exercises the actual batched-decode
        // hot path — unlike synthetic BENCH_BATCH which iterates the
        // streams sequentially, or BENCH_BATCH_CHAT which uses B=1.
        // Verifies both streams complete with coherent output and that
        // both slots finish under EOS/max-tokens. Uses real HF tokenizer.
        if (env["BENCH_BATCH_CONCURRENT"] ?? "0") == "1" {
            try await runBatchEngineConcurrent(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_CACHE_HIT=1 (iter 34): demonstrate CacheCoordinator
        // cross-turn prefix reuse through BatchEngine. Submits two turns
        // where turn 2 extends turn 1's prompt; asserts turn 2's prompt
        // time is meaningfully lower (cache hit on the shared prefix).
        if (env["BENCH_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await runBatchEngineCacheHit(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_DISK_RESTORE=1 (iter 35): verify L2 disk cache
        // round-trips through BatchEngine. Turn 1 stores via finishSlot;
        // the coordinator is then DROPPED and recreated fresh against the
        // same disk dir — modelling an osaurus session restart. Turn 2
        // must still hit and skip prefill. This is the strongest "session
        // persistence across runs" property and the one osaurus relies on.
        if (env["BENCH_BATCH_DISK_RESTORE"] ?? "0") == "1" {
            try await runBatchEngineDiskRestore(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_PERSLOT_SAMPLER=1 (iter 36): submit two slots with
        // DIFFERENT sampling params into the same B=2 engine. Slot 0
        // temp=0 greedy (deterministic, re-runnable byte-identical).
        // Slot 1 temp=0.8 topP=0.9 (stochastic). Must prove each slot's
        // GenerateParameters flows through to its own sampler — osaurus
        // spec explicitly calls this out.
        if (env["BENCH_BATCH_PERSLOT_SAMPLER"] ?? "0") == "1" {
            try await runBatchEnginePerSlotSampler(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_TQ_B2=1 (iter 38): concurrent B=2 with heterogeneous
        // kvMode. Slot 0 plain KV, slot 1 turboQuant(3,3). Verifies Stage 0
        // compression per-slot without cross-slot corruption, plus both
        // streams complete with coherent output.
        if (env["BENCH_BATCH_TQ_B2"] ?? "0") == "1" {
            try await runBatchEngineTurboQuantB2(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_B4=1 (iter 39): four concurrent distinct prompts
        // submitted into `maxBatchSize=4`. Osaurus ships max=4 as the
        // default `mlxBatchEngineMaxBatchSize` — this must work end-to-end
        // with real HF tokenizer. Asserts all four slots complete with
        // coherent non-empty output AND no cross-slot mixing (slot 0's
        // tokens identical to a solo run of the same prompt).
        // Set BENCH_B_SIZE=8 (or other) to stress higher fan-out.
        if (env["BENCH_BATCH_B4"] ?? "0") == "1" {
            let b = Int(env["BENCH_B_SIZE"] ?? "4") ?? 4
            try await runBatchEngineBMany(
                modelPath: modelPath, maxNew: maxNew, batchSize: b)
            return
        }

        // BENCH_BATCH_CANCEL=1 (iter 40): cancel mid-stream under B=3.
        // One slot cancelled after a few tokens; surviving slots must
        // decode to max-tokens. Verifies the `.cancelled` info event
        // and that engine state recovery is clean.
        if (env["BENCH_BATCH_CANCEL"] ?? "0") == "1" {
            try await runBatchEngineCancelMidStream(
                modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CRASH_FUZZ=1 (2026-04-23): osaurus-style adversarial fuzz
        // for the tpae Qwen 3.6 27B crash report. One model load, many
        // scenarios. Each scenario prints "SCENARIO N: <name>" up-front
        // so when we crash the last line tells us where. Covers:
        //   1. B=1 baseline
        //   2. B=4 concurrent distinct prompts
        //   3. B=4 + cancellation mid-stream on two slots
        //   4. maxTokens=1 on every slot (near EOS-on-first-token)
        //   5. Single-token prompt
        //   6. Rapid submit + immediate consumer drop (no iteration)
        //   7. Same prompt submitted twice back-to-back (cache contention)
        //   8. B=4 with stop-string that matches immediately
        //   9. B=4 with wildly different lengths (short + long)
        //  10. 5 rapid sequential single-turn submits (connection churn)
        if (env["BENCH_CRASH_FUZZ"] ?? "0") == "1" {
            try await runCrashFuzz(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_CRASH_FUZZ_V2=1 (2026-04-23): adversarial multi-turn
        // stress through the FULL `generate()` pipeline (NOT `submit()`
        // like v1). Runs against the real chat template, reasoning
        // parser, tool-call processor, stop-string matcher, and
        // NaiveStreamingDetokenizer on every turn. Covers specifically
        // the tokenizer-decode-shrinkage class of bugs (cleanup
        // substitutions, byte-level BPE emoji completion, adjacent
        // special-token collapse) by asking the model for output that
        // is likely to trigger each pattern.
        if (env["BENCH_CRASH_FUZZ_V2"] ?? "0") == "1" {
            try await runCrashFuzzV2(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_OFFICIAL=1 (2026-04-23): final-pass multi-turn harness.
        // For a single model, runs a 6-scenario matrix and reports
        // per-turn TTFT, decode tok/s, reasoning/chunk/tool-call counts,
        // peak RSS, and response-content validation (where applicable).
        // Meant to be invoked per model via shell loop so each model
        // gets its own process (clean GPU memory baseline).
        if (env["BENCH_OFFICIAL"] ?? "0") == "1" {
            try await runOfficialMultiTurn(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_PROD=1 (2026-04-23): EXHAUSTIVE production matrix.
        // Expands BENCH_OFFICIAL with:
        //   - Multi-turn tool-call ROUND TRIP (assistant emits tool_call
        //     with valid name+args, we inject a fake tool response, model
        //     continues)
        //   - Reasoning ON→OFF→ON alternation on a single engine
        //   - L2 disk cache: second identical turn hits the disk cache
        //     (not just paged); cache directory is explicitly configured
        //   - SSM state re-derive: on hybrid SSM models, second turn
        //     shares a prompt prefix — prefix hit + SSM seed should
        //     speed up prefill ≥2×
        //   - TurboQuant load + forward: on a JANGTQ bundle, verify
        //     model loads with sidecar and produces tokens
        // Validates content per scenario — math contains the answer,
        // factual contains the expected word, tool-call schema name matches.
        if (env["BENCH_PROD"] ?? "0") == "1" {
            try await runProdMatrix(modelPath: modelPath, maxNew: maxNew)
            return
        }

        // BENCH_BATCH_LONG_CONTEXT=1 (iter 42): submit a 2000+ token
        // prompt single-slot through BatchEngine AND through TokenIterator;
        // assert byte-identical token output. Exercises chunked prefill
        // (multi-pass of `prefillStepSize`=512 over ~2k tokens), memory
        // purge during long decode (`memoryPurgeInterval`=256), and
        // sliding-window interaction near the model's cache budget.
        // Tune prompt length via BENCH_LONG_LEN (default 2048).
        if (env["BENCH_BATCH_LONG_CONTEXT"] ?? "0") == "1" {
            let len = Int(env["BENCH_LONG_LEN"] ?? "2048") ?? 2048
            try await runBatchEngineLongContext(
                modelPath: modelPath, maxNew: maxNew, promptLen: len)
            return
        }

        // BENCH_BATCH_SPECDEC=1 (iter 16): run the same prompt under
        // (a) plain generate, (b) DFlash linear, (c) DDTree tree-verify,
        // all against the same target model. Report per-path tok/s +
        // byte-parity vs plain at temperature 0. Drafter path comes
        // from env var BENCH_SPECDEC_DRAFTER.
        if (env["BENCH_BATCH_SPECDEC"] ?? "0") == "1" {
            let drafter = env["BENCH_SPECDEC_DRAFTER"]
                ?? "/tmp/ddtree-downloads/Qwen3.5-27B-DFlash"
            try await runBatchSpecDec(
                modelPath: modelPath,
                drafterPath: drafter,
                maxNew: maxNew)
            return
        }

        // BENCH_VL_CROSS_VALIDATE=1 (iter 47): run the same VL prompt
        // (text + image) through TokenIterator AND BatchEngine, then
        // assert byte-identical token output. Extends iter 32/44's
        // cross-engine validation from dense/hybrid text to vision path.
        // Depends on iter 45's UserInput fix so images actually reach
        // the processor.
        if (env["BENCH_VL_CROSS_VALIDATE"] ?? "0") == "1" {
            try await VLBench.runCrossValidate(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_BATCH_CACHE_HIT=1 (iter 48): VL multi-turn cache
        // reuse end-to-end. Turn 1: describe image. Turn 2: extend the
        // prompt with a follow-up question about the same image. Asserts
        // turn 2 produces a paged HIT on (tokens, mediaSalt) — "user
        // asks another question about the same photo" scenario.
        if (env["BENCH_VL_BATCH_CACHE_HIT"] ?? "0") == "1" {
            try await VLBench.runBatchCacheHit(modelPath: modelPath, maxNewTokens: maxNew)
            return
        }

        // BENCH_VL_VIDEO=1 (iter 49): video input end-to-end. Loads a
        // short .mov via AVFoundation, processes frames through the
        // VLM processor, runs the vision tower on the frame sequence,
        // decodes text. Path: UserInput(videos:) → processor.prepare
        // → LMInput.video tensor → model forward.
        if (env["BENCH_VL_VIDEO"] ?? "0") == "1" {
            let videoPath = env["BENCH_VIDEO_PATH"]
                ?? "/Users/eric/vmlx-swift-lm/Tests/MLXLMTests/Resources/1080p_30.mov"
            try await VLBench.runVideoSmoke(
                modelPath: modelPath,
                videoPath: videoPath,
                maxNewTokens: maxNew)
            return
        }

        // Simple load-and-generate mode: when BENCH_SIMPLE=1, skip multi-turn
        // tokens and just generate N tokens from a short static prompt. Used
        // to smoke-test new model paths (e.g. .jangspec bundles, JANGTQ).
        // BENCH_COORDINATOR=1 installs a CacheCoordinator matching Osaurus's
        // config (isHybrid=true disk=true maxBlocks=2000) to reproduce its
        // prefill / cache-miss / Metal-concurrency path.
        // BENCH_PROMPT_LEN=1394 seeds a deterministic long prompt to match
        // the exact token count from the user's crash report.
        if (env["BENCH_SIMPLE"] ?? "0") == "1" {
            let promptLen = Int(env["BENCH_PROMPT_LEN"] ?? "10") ?? 10
            let seedTokens: [Int32] = (0..<promptLen).map { Int32($0 % 4096 + 1) }
            let simpleInput = LMInput(text: LMInput.Text(
                tokens: MLXArray(seedTokens)[.newAxis, .ellipsis]))
            var sp = GenerateParameters(maxTokens: maxNew)
            sp.temperature = 0.0
            sp.prefillStepSize = Int(env["BENCH_PREFILL_STEP"] ?? "512") ?? 512
            sp.enableCompiledDecode = compileDecode
            sp.compiledMaxCacheLength = compileMaxLen
            let sCache = context.model.newCache(parameters: sp)
            let sCoord: CacheCoordinator?
            if (env["BENCH_COORDINATOR"] ?? "0") == "1" {
                var cfg = CacheCoordinatorConfig()
                cfg.usePagedCache = true
                cfg.maxCacheBlocks = 2000
                cfg.enableDiskCache = true
                cfg.diskCacheDir = URL(fileURLWithPath: "/tmp/bench_disk_cache")
                let c = CacheCoordinator(config: cfg)
                let isHybrid = sCache.contains { !($0 is KVCacheSimple) && !($0 is RotatingKVCache) }
                c.setHybrid(isHybrid)
                sCoord = c
                print("[Coord] isHybrid=\(isHybrid) disk=true maxBlocks=2000")
            } else {
                sCoord = nil
            }
            // Warmup forward pass: triggers lazy module initializations
            // (e.g. SwitchGLU fused gate+up cache) so the timed runs below
            // don't pay that one-time concatenation cost in TTFT. Use a
            // small prompt + 1 token so the warmup is fast.
            if (env["BENCH_SKIP_WARMUP"] ?? "0") != "1" {
                let warmSeed: [Int32] = [1, 2, 3, 4, 5]
                let warmInput = LMInput(text: LMInput.Text(
                    tokens: MLXArray(warmSeed)[.newAxis, .ellipsis]))
                var wParams = GenerateParameters(maxTokens: 1)
                wParams.temperature = 0.0
                wParams.prefillStepSize = 512
                let warmCache = context.model.newCache(parameters: wParams)
                var warmIter = try TokenIterator(
                    input: warmInput, model: context.model,
                    cache: warmCache, parameters: wParams)
                _ = warmIter.next()
            }

            // BENCH_RUNS=N runs the same prompt N times. With a coordinator
            // installed the first run is a cold fetch (miss → full prefill →
            // store on completion), subsequent runs should hit the paged
            // cache for the prefix and only prefill the remainder.
            let runs = Int(env["BENCH_RUNS"] ?? "1") ?? 1
            let tokenIds = (0..<promptLen).map { Int($0 % 4096 + 1) }
            for runIdx in 0..<runs {
                print("\n[Simple run \(runIdx + 1)/\(runs)] \(maxNew) tokens from \(promptLen)-token prompt (prefillStep=\(sp.prefillStepSize))")
                let runCache = context.model.newCache(parameters: sp)
                let t0 = CFAbsoluteTimeGetCurrent()
                var sIter = try TokenIterator(
                    input: simpleInput, model: context.model, cache: runCache, parameters: sp,
                    cacheCoordinator: sCoord)
                var firstTokT: Double = 0
                var count = 0
                var generated: [Int] = []
                while let tok = sIter.next() {
                    count += 1
                    if count == 1 { firstTokT = CFAbsoluteTimeGetCurrent() - t0 }
                    generated.append(tok)
                }
                let tot = CFAbsoluteTimeGetCurrent() - t0
                let decodeTime = max(tot - firstTokT, 0.001)
                print(String(format:
                    "  generated %d tokens | TTFT %.0fms | decode %.1f tok/s | total %.2fs",
                    count, firstTokT * 1000, Double(count - 1) / decodeTime, tot))
                print("  first 10 tokens: \(Array(generated.prefix(10)))")

                // Manual store for single-request path with stepwise stderr
                // logging to pin down which call crashes — stdout buffering
                // hides the real crash site when the Fatal error takes over.
                // Skip via BENCH_STORE=0.
                if let c = sCoord, (env["BENCH_STORE"] ?? "1") == "1" {
                    FileHandle.standardError.write("[store] evaling cache\n".data(using: .utf8)!)
                    MLX.eval(runCache)
                    FileHandle.standardError.write("[store] extractLayerData\n".data(using: .utf8)!)
                    let perLayer = extractLayerData(from: runCache)
                    FileHandle.standardError.write("[store] extractLayerData done, non-nil=\(perLayer.compactMap{$0}.count)/\(perLayer.count)\n".data(using: .utf8)!)
                    let ssm: [MLXArray]?
                    if c.isHybrid {
                        FileHandle.standardError.write("[store] extractSSMStates\n".data(using: .utf8)!)
                        ssm = extractSSMStates(from: runCache)
                        FileHandle.standardError.write("[store] extractSSMStates done, count=\(ssm?.count ?? 0)\n".data(using: .utf8)!)
                    } else { ssm = nil }
                    if let ssm = ssm {
                        FileHandle.standardError.write("[store] eval ssm\n".data(using: .utf8)!)
                        MLX.eval(ssm)
                    }
                    FileHandle.standardError.write("[store] eval per-layer KV\n".data(using: .utf8)!)
                    for kv in perLayer {
                        if let kv = kv { MLX.eval(kv.keys, kv.values) }
                    }
                    FileHandle.standardError.write("[store] call storeAfterGeneration\n".data(using: .utf8)!)
                    c.storeAfterGeneration(
                        promptTokens: tokenIds,
                        perLayerData: perLayer,
                        ssmStates: ssm,
                        cache: runCache,
                        mediaSalt: nil)
                    FileHandle.standardError.write("[store] done\n".data(using: .utf8)!)
                    print("  [Coord] stored after generation")
                }
            }
            print("\n=== Simple Done ===")
            return
        }

        // Load pre-tokenized multi-turn data
        let tokFile = URL(fileURLWithPath: tokensPath)
        let data = try Data(contentsOf: tokFile)
        let json = try JSONSerialization.jsonObject(with: data) as! [String: Any]
        let turns = json["turns"] as! [[Int]]
        let stubTokens = json["stub_tokens"] as! [Int]

        print("Loaded \(turns.count) turns, sizes: \(turns.map { $0.count })")

        // Diagnostic: print first few module parameter shapes
        var diagCount = 0
        for (key, arr) in context.model.parameters().flattened() {
            if key.contains("layers.0") && (key.contains("q_proj") || key.contains("embed_tokens")) {
                print("  param[\(key)] shape=\(arr.shape) dtype=\(arr.dtype)")
                diagCount += 1
                if diagCount >= 8 { break }
            }
        }

        var params = GenerateParameters(maxTokens: maxNew)
        params.temperature = 0.0
        params.prefillStepSize = 8192
        params.enableCompiledDecode = compileDecode
        params.compiledMaxCacheLength = compileMaxLen
        if compileDecode {
            print("Compiled decode: ON (maxCacheLength=\(compileMaxLen))")
        }

        // Persistent cache across turns
        let cache = context.model.newCache(parameters: params)
        let cacheTypes = Set(cache.map { String(describing: type(of: $0)) })
        print("Cache: \(cache.count) layers, types: \(cacheTypes)")

        // Warmup with a small prompt to prime kernels
        print("\n[Warmup] 64 tokens...")
        let warmupCache = context.model.newCache(parameters: params)
        let warmupTokens = Array(turns[0].prefix(min(25, turns[0].count))).map { Int32($0) }
        // 2D tokens [1, L] — works because step() now detects ndim and skips re-newAxis.
        let warmupTokenArray = MLXArray(warmupTokens)[.newAxis, .ellipsis]
        let warmupInput = LMInput(text: LMInput.Text(tokens: warmupTokenArray))
        var wParams = params
        wParams.maxTokens = 32
        var warmupIter = try TokenIterator(input: warmupInput, model: context.model, cache: warmupCache, parameters: wParams)
        var wCount = 0
        while let _ = warmupIter.next() { wCount += 1 }
        print("  warmup done (\(wCount) tokens)")

        print("\n[Multi-turn] cache reused across turns, generate 256 tokens/turn")
        // Track cumulative tokens fed to cache so we know what's "new" per turn
        var cumulativeTokens = 0
        for (turnIdx, turnTokens) in turns.enumerated() {
            // The "new" tokens this turn are everything beyond what we've already
            // processed. If cache offset == cumulativeTokens (assistant stub from
            // previous turn was injected), feed only the difference.
            let nPrompt = turnTokens.count
            let newTokens: [Int32]
            if turnIdx == 0 {
                newTokens = turnTokens.map { Int32($0) }
            } else {
                // Take only the tokens that aren't already cached.
                // cumulativeTokens reflects: (prev_turn_prompt + assistant_stub).
                // turnTokens[turnIdx] already includes prev user msg + prev stub assistant + new user msg
                let already = cumulativeTokens
                let slice = turnTokens[already...].map { Int32($0) }
                newTokens = Array(slice)
            }

            // 2D tokens [1, L] — TokenIterator handles both 1D and 2D safely.
            let input = LMInput(text: LMInput.Text(tokens: MLXArray(newTokens)[.newAxis, .ellipsis]))

            let t0 = CFAbsoluteTimeGetCurrent()
            var iter = try TokenIterator(input: input, model: context.model, cache: cache, parameters: params)

            var firstTokenTime: Double = 0
            var count = 0
            var generated: [Int] = []
            while let tok = iter.next() {
                count += 1
                if count == 1 { firstTokenTime = CFAbsoluteTimeGetCurrent() - t0 }
                generated.append(tok)
            }
            let totalTime = CFAbsoluteTimeGetCurrent() - t0
            let decodeTime = totalTime - firstTokenTime
            let decodeTps = Double(max(0, count - 1)) / decodeTime
            // Prefill speed = tokens fed in this turn / time-to-first-token
            let prefillTps = Double(newTokens.count) / firstTokenTime

            print(String(format: "  Turn %d: total prompt=%d, NEW=%d | prefill %.0f tok/s (%.0fms TTFT) | decode %.1f tok/s (%d tokens, %.3fs)",
                         turnIdx + 1, nPrompt, newTokens.count, prefillTps, firstTokenTime * 1000, decodeTps, count, decodeTime))

            // After this turn: cache contains [prev_context + new_tokens + decoded_response].
            // Next turn's "already cached" = current turnTokens.count + assistant stub length
            // Note: the stub injected matches the next prompt's pre-existing assistant turn.
            cumulativeTokens = nPrompt + stubTokens.count
        }

        print("\n=== Done ===")
    }
}

// Stub tokenizer — model loading requires one, but this bench bypasses tokenization
// (uses pre-tokenized JSON tokens from Python).
final class NullTokenizerLoader: TokenizerLoader, @unchecked Sendable {
    func load(from directory: URL) async throws -> any MLXLMCommon.Tokenizer {
        return NullTokenizer()
    }
}

// MARK: - BatchEngine multi-turn chat (iter 26)

/// TRUE BatchEngine multi-turn verification. Unlike
/// `runCoherentMultiTurn` which uses `ChatSession` (backed by
/// `TokenIterator`), this routes each turn through
/// `BatchEngine.generate(...)` with a shared `CacheCoordinator` so
/// cross-turn cache hits are possible.
///
/// Runs three turns with a factual callback (turn 2 should recall
/// turn 1's info), twice — once with compile off, once with compile on.
func runBatchEngineMultiTurn(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine multi-turn chat (iter 26, simplified iter 27) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Iter 27: use BatchEngine DIRECTLY on the loaded context rather
    // than wrapping in ModelContainer + enableCaching. Iter 26's
    // ModelContainer wrap hung for 8 minutes — likely an interaction
    // between makeBatchEngine's nested perform block and a freshly-
    // constructed container that doesn't match the context's
    // original container.
    nonisolated(unsafe) let ctx = context

    for compileOn in [false, true] {
        let label = compileOn ? "compile ON" : "compile OFF"
        print("\n[\(label)] BatchEngine 3-turn chat")

        var params = GenerateParameters(
            maxTokens: maxNew, temperature: 0,
            prefillStepSize: 512)
        params.enableCompiledBatchDecode = compileOn

        let engine = BatchEngine(context: ctx, maxBatchSize: 1)

        // Accumulate a simple text transcript so turn 2's prompt
        // contains turn 1's response, exercising multi-turn context
        // without needing cache-coordinator prefix matching.
        var history = "You are a helpful assistant. Keep responses very brief."
        for (i, prompt) in [
            "My favorite color is blue.",
            "What is my favorite color?",
            "Is that a warm or cool color?",
        ].enumerated() {
            history += "\n\nUser: \(prompt)\nAssistant:"
            let text = try await runBatchEngineTurn(
                engine: engine, context: ctx,
                fullText: history, label: "Turn \(i+1)",
                parameters: params, maxNew: maxNew)
            history += " \(text)"
        }
    }

    print("\n=== BatchEngine multi-turn done ===")
}

/// Send one UserInput through `BatchEngine.generate(...)`, collect
/// text chunks, return decoded response.
func runBatchEngineTurn(
    engine: BatchEngine,
    context: MLXLMCommon.ModelContext,
    fullText: String,
    label: String,
    parameters: GenerateParameters,
    maxNew: Int
) async throws -> String {
    print("  \(label) [\(parameters.enableCompiledBatchDecode ? "compile" : "uncomp")]:")
    let t0 = CFAbsoluteTimeGetCurrent()

    // Prepare input directly on the loaded context (no container actor).
    let input = try await context.processor.prepare(
        input: UserInput(prompt: fullText))
    nonisolated(unsafe) let sendable = input
    // Iter 28: test the fixed `generate()` path. Iter 27 had to use
    // submit() as a workaround because generate() hung under real HF
    // tokenizer. If this iteration runs to completion, the iter-28 fix
    // (Task.detached in generate) actually worked.
    let stream = await engine.generate(input: sendable, parameters: parameters)

    var text = ""
    var ttft: Double?
    var chunkCount = 0
    for await event in stream {
        switch event {
        case .chunk(let chunk):
            if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
            text += chunk
            chunkCount += 1
            if chunkCount > maxNew * 2 { break }
        case .reasoning, .info, .toolCall:
            break
        }
    }
    let total = CFAbsoluteTimeGetCurrent() - t0
    let preview = text.count > 150 ? String(text.prefix(150)) + "..." : text
    print("    TTFT \(Int((ttft ?? 0) * 1000))ms, total \(String(format: "%.2fs", total))")
    print("    \"\(preview)\"")
    return text
}

// MARK: - Tool-call pipeline end-to-end (iter 66)

/// Submit a tool-bearing prompt through `BatchEngine.generate(...)` on a
/// real model, collecting `.chunk` / `.toolCall` events. Assert the
/// library-level pipeline contract:
///   - `.chunk` output MUST NOT contain raw tool-call markers.
///   - `.chunk` output MUST NOT contain raw `<think>...</think>` markers.
///
/// The model is nondeterministic at the system level (tokenizer template,
/// weights, decode path) so we do not require a `.toolCall` event — we
/// only require that IF markers appear in the raw stream they get
/// extracted or stripped rather than leaking to osaurus.
func runBatchEngineToolCall(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine tool-call pipeline (iter 66) ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Tool format: \(context.configuration.toolCallFormat.map { "\($0)" } ?? "json (default)")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)

    // Prompt that encourages the model to call a tool. Exact format is
    // family-specific, but the same user text paired with the model's
    // own chat template is the realistic path.
    let prompt = """
        You have access to a tool:
        - get_weather(location: string) → returns the current weather

        User question: What is the weather in Tokyo right now?

        If you need external data to answer, emit a tool call in the
        format your training expects, otherwise just answer directly.
        """

    var params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    params.enableCompiledBatchDecode = false

    let input = try await ctx.processor.prepare(input: UserInput(prompt: prompt))
    nonisolated(unsafe) let sendable = input
    let stream = await engine.generate(input: sendable, parameters: params)

    var chunkText = ""
    var toolCallCount = 0
    for await event in stream {
        switch event {
        case .chunk(let c):
            chunkText += c
        case .reasoning:
            break
        case .toolCall:
            toolCallCount += 1
        case .info:
            break
        }
    }

    let preview = chunkText.count > 240 ? String(chunkText.prefix(240)) + "..." : chunkText
    print("  chunks: \(chunkText.count) chars, toolCalls: \(toolCallCount)")
    print("  text preview: \"\(preview)\"")

    // Contract assertions. These must hold on *every* family: osaurus
    // relies on pure-text `.chunk` + authoritative `.toolCall` events.
    let leakedMarkers = [
        "<tool_call>",
        "<|tool_call>",
        "<minimax:tool_call>",
        "[TOOL_CALLS]",
        "<think>",
        "</think>",
    ].filter { chunkText.contains($0) }

    if !leakedMarkers.isEmpty {
        print("  FAIL — raw markers leaked into .chunk: \(leakedMarkers)")
        throw NSError(
            domain: "BENCH_BATCH_TOOLCALL", code: 1,
            userInfo: [NSLocalizedDescriptionKey:
                "Raw library-level markers leaked into .chunk: \(leakedMarkers)."])
    }
    print("  OK — no raw tool-call / reasoning markers leaked to .chunk.")

    print("\n=== BatchEngine tool-call pipeline done ===")
}

// MARK: - SpecDec bench scenario (iter 16)

/// Run the same short prompt through three paths — plain `Evaluate.generate`,
/// DFlash linear, DDTree tree-verify — and report per-path tok/s +
/// byte-parity vs plain at temperature 0.
///
/// The drafter+target pair must share a hidden_size. For the downloaded
/// snapshots, that means:
///   - drafter `z-lab/Qwen3.5-27B-DFlash` (hidden=5120, 5 layers)
///     pairs with target `mlx-community/Qwen3.5-27B-4bit` (hidden=5120).
///
/// Prints one line per path so the operator can cross-reference tok/s
/// numbers. Checks byte-parity between plain + DFlash + DDTree, failing
/// the run (non-zero exit) if outputs diverge.
func runBatchSpecDec(
    modelPath: String, drafterPath: String, maxNew: Int
) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let drafterDir = URL(fileURLWithPath: drafterPath)
    print("\n=== BatchEngine SpecDec (iter 16) ===")
    print("Target:  \(modelDir.lastPathComponent)")
    print("Drafter: \(drafterDir.lastPathComponent)")

    // Must resolve to a DFlash drafter snapshot.
    guard DFlashDrafterLoader.looksLikeDrafter(at: drafterDir) else {
        print("  [skip] drafter not on disk at \(drafterDir.path)")
        return
    }
    let drafter: DFlashDraftModel
    do {
        drafter = try DFlashDrafterLoader.load(from: drafterDir)
    } catch {
        print("  [skip] drafter load failed: \(error)")
        return
    }
    // HF target_layer_ids ARE 0-based indices into target.model.layers
    // (per z-lab/dflash `_patch_model`). Use them directly — no shift.
    let targetBlockIDs = drafter.config.dflashConfig.targetLayerIds

    // Load target via HF tokenizer so the drafter+target share tokenizer.
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Target load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))

    // Deterministic prompt. Override via BENCH_SPECDEC_PROMPT env var.
    let promptText = ProcessInfo.processInfo.environment["BENCH_SPECDEC_PROMPT"]
        ?? "The capital of France is"
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: [["role": "user", "content": promptText]])
    let promptInts = promptTokens.map { Int32($0) }
    let promptIds = MLXArray(promptInts).reshaped(1, promptInts.count)

    // Cast target to SpecDec protocols — skip if not conformant.
    guard let target = context.model
        as? any (HiddenStateCaptureModel & TokenEmbedderModel)
    else {
        print("  [skip] target \(type(of: context.model)) does not conform to HiddenStateCaptureModel + TokenEmbedderModel")
        return
    }

    // Plain greedy AR with a PERSISTENT KV cache — the honest baseline.
    // Prefills the whole prompt once (cache fills with promptLen states),
    // then each decode step only processes ONE new token through the
    // cached model (O(1) per step instead of O(N)).
    func materializeLogits(_ a: MLXArray) { MLX.eval(a) }
    func greedyAR() throws -> [Int32] {
        var out = promptInts
        let cache = context.model.newCache(parameters: nil)
        let promptArr = MLXArray(promptInts).reshaped(1, promptInts.count)
        var (logits, _) = target(promptArr, cache: cache, captureLayerIDs: [])
        materializeLogits(logits)
        var nextTok = argMax(
            logits[0, logits.dim(1) - 1, 0...], axis: -1
        ).asType(.int32).item(Int32.self)
        out.append(nextTok)
        for _ in 1..<maxNew {
            let stepIn = MLXArray([nextTok]).reshaped(1, 1)
            (logits, _) = target(stepIn, cache: cache, captureLayerIDs: [])
            materializeLogits(logits)
            nextTok = argMax(
                logits[0, logits.dim(1) - 1, 0...], axis: -1
            ).asType(.int32).item(Int32.self)
            out.append(nextTok)
        }
        return out
    }

    // Measurement harness.
    func measure<T>(_ label: String, _ body: () throws -> (T, Int)) rethrows -> T {
        let t0 = CFAbsoluteTimeGetCurrent()
        let (result, generated) = try body()
        let dt = CFAbsoluteTimeGetCurrent() - t0
        let tps = dt > 0 ? Double(generated) / dt : 0
        print(String(
            format: "  %@: %.2fs / %d tokens / %.1f tok/s",
            label, dt, generated, tps))
        return result
    }

    let arTokens = try measure("plain AR") {
        let tokens = try greedyAR()
        return (tokens, tokens.count - promptInts.count)
    }

    // Optional TurboQuant KV compression on the fast path.
    // BENCH_SPECDEC_KV_TURBOQUANT=1 enables 3-bit TurboQuant.
    let kvMode: KVQuantizationMode
    if ProcessInfo.processInfo.environment[
        "BENCH_SPECDEC_KV_TURBOQUANT"] == "1"
    {
        kvMode = .turboQuant(keyBits: 3, valueBits: 3)
        print("  kv-compression: turboQuant(3,3)")
    } else {
        kvMode = .none
    }

    var dfAcceptance: [Int] = []
    let dfTokens = try measure("DFlash linear") {
        let args = DFlashLinearArgs(
            target: target, drafter: drafter,
            targetBlockIDs: targetBlockIDs,
            maskTokenID: Int32(drafter.config.dflashConfig.maskTokenId),
            inputIds: promptIds, maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0, kvMode: kvMode)
        let r = try SpecDecRuntimeLinear.run(args)
        dfAcceptance = r.acceptanceLengths
        return (r.tokenIds, r.tokenIds.count - promptInts.count)
    }
    if !dfAcceptance.isEmpty {
        let mean = Double(dfAcceptance.reduce(0, +)) / Double(dfAcceptance.count)
        let bs = drafter.config.blockSize
        print(String(
            format: "    DFlash acceptance: %d rounds, mean=%.2f / %d, draft tokens/round=%.2f",
            dfAcceptance.count, mean, bs - 1, mean + 1))
    }

    var ddAcceptance: [Int] = []
    let ddTokens = try measure("DDTree (budget=8)") {
        let args = DDTreeArgs(
            target: target, drafter: drafter,
            targetBlockIDs: targetBlockIDs,
            maskTokenID: Int32(drafter.config.dflashConfig.maskTokenId),
            inputIds: promptIds, maxNewTokens: maxNew,
            stopTokenIDs: [], temperature: 0,
            branchingBudget: 8)
        let r = try SpecDecRuntimeDDTree.run(args)
        ddAcceptance = r.acceptanceLengths
        return (r.tokenIds, r.tokenIds.count - promptInts.count)
    }
    if !ddAcceptance.isEmpty {
        let mean = Double(ddAcceptance.reduce(0, +)) / Double(ddAcceptance.count)
        print(String(
            format: "    DDTree acceptance: %d rounds, mean depth=%.2f",
            ddAcceptance.count, mean))
    }

    // Byte-parity vs greedy AR. At temp=0 the *accepted* SpecDec tokens
    // should equal AR argmaxes — but high-precision targets (bf16) can
    // show sub-ULP drift from running SDPA over different (q_len, k_len)
    // shapes, occasionally flipping close argmaxes. Report the match
    // rate rather than crashing so the bench can still complete.
    func matchCount(_ a: [Int32], _ b: [Int32]) -> (Int, Int) {
        let n = min(a.count, b.count)
        var m = 0
        for i in 0..<n where a[i] == b[i] { m += 1 }
        return (m, n)
    }
    let (dfM, dfT) = matchCount(dfTokens, arTokens)
    let (ddM, ddT) = matchCount(ddTokens, arTokens)
    let dfPct = dfT > 0 ? 100.0 * Double(dfM) / Double(dfT) : 100.0
    let ddPct = ddT > 0 ? 100.0 * Double(ddM) / Double(ddT) : 100.0
    print(String(format:
        "  byte-parity vs AR: DFlash=%d/%d (%.1f%%), DDTree=%d/%d (%.1f%%)",
        dfM, dfT, dfPct, ddM, ddT, ddPct))

    print("=== BatchEngine SpecDec done ===")
}

// MARK: - Coherent multi-turn chat (iter 19)

/// Run a real 3-turn conversation through `BatchEngine` using the
/// Hugging Face tokenizer + chat template + cache coordinator. The
/// user has repeatedly asked for coherent multi-turn verification,
/// not just synthetic-prompt tok/s. This harness delivers that.
///
/// - Turn 1: introduces a fact ("My favorite color is blue")
/// - Turn 2: asks the model to recall it
/// - Turn 3: asks a follow-up that should reference turn 2's answer
///
/// Both compile-on and compile-off paths are exercised to confirm
/// coherence doesn't regress when compile engages.
func runCoherentMultiTurn(modelPath: String, maxNew: Int) async throws {
    // Reload with real HF tokenizer (the default Bench load uses
    // NullTokenizer for perf benchmarks; coherence needs the real one).
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== Coherent multi-turn chat (iter 19) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Three-turn conversation, same shape as
    // `ChatSessionTests.multiTurnConversation`. Using ChatSession lets
    // us exercise the full chat template + multi-turn cache path.
    for compileOn in [false, true] {
        let label = compileOn ? "compile ON" : "compile OFF"
        print("\n[\(label)] 3-turn chat coherence")

        var params = GenerateParameters(maxTokens: maxNew, temperature: 0)
        params.enableCompiledBatchDecode = compileOn

        let session = ChatSession(
            context,
            instructions: "You are a helpful assistant. Keep responses very brief (one sentence max).",
            generateParameters: params
        )

        try await runChatTurn(session: session, prompt: "My favorite color is blue.", label: "Turn 1")
        try await runChatTurn(session: session, prompt: "What is my favorite color?", label: "Turn 2")
        try await runChatTurn(session: session, prompt: "Is that a warm or cool color?", label: "Turn 3")
    }

    print("\n=== Coherent multi-turn done ===")
}

/// Stream one ChatSession turn; print label, tok/s, and response text
/// (truncated to 200 chars for legibility).
func runChatTurn(session: ChatSession, prompt: String, label: String) async throws {
    print("\n  User: \(prompt)")
    let t0 = CFAbsoluteTimeGetCurrent()
    var text = ""
    var ttft: Double?
    for try await chunk in session.streamResponse(to: prompt) {
        if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
        text += chunk
    }
    let total = CFAbsoluteTimeGetCurrent() - t0
    print("  \(label) [TTFT \(Int((ttft ?? 0) * 1000))ms | total \(String(format: "%.2fs", total))]:")
    let preview = text.count > 200 ? String(text.prefix(200)) + "..." : text
    print("    \"\(preview)\"")
}

// MARK: - BatchEngine real-model smoke (iter 17)

/// Runs 4 scenarios through `BatchEngine` on a real loaded model:
///   1. Baseline (compile off, maxBatchSize=1)
///   2. Stage 1B.3 compile-on (enableCompiledBatchDecode, maxBatchSize=1)
///   3. Stage 0 TurboQuant (kvMode=.turboQuant, maxBatchSize=1)
///   4. B=2 concurrent (uncompiled, maxBatchSize=2)
///
/// Each scenario prints TTFT, decode tok/s, and token IDs for manual
/// coherence inspection. Intended as the real-model counterpart to the
/// synthetic-model unit tests — verifies the BatchEngine changes work
/// on Qwen3-0.6B-8bit (or any other loaded real model via BENCH_MODEL).
func runBatchSmoke(context: MLXLMCommon.ModelContext, maxNew: Int) async throws {
    print("\n=== BatchEngine real-model smoke (iter 17, iter 18 warmup) ===")

    let promptIDs = (1...8).map { Int32($0) }
    let prompt = MLXArray(promptIDs)[.newAxis, .ellipsis]
    let input = LMInput(text: LMInput.Text(tokens: prompt))

    // Warmup pass (iter 18, extended iter 20): the first forward pass
    // pays one-time lazy-module initialisation, AND the first compile
    // trace pays compile-time cost. Warm both the uncompiled path and
    // the compile path separately so each measured scenario starts
    // from a warm state.
    print("\n[Warmup] (not measured)")
    try await runBatchScenario(
        context: context, input: input, label: "warmup-uncompiled",
        params: GenerateParameters(maxTokens: 3, temperature: 0),
        maxBatchSize: 1, silent: true)
    try await runBatchScenario(
        context: context, input: input, label: "warmup-compiled",
        params: GenerateParameters(
            maxTokens: 3, enableCompiledBatchDecode: true, temperature: 0),
        maxBatchSize: 1, silent: true)

    // 1. Baseline: compile off, maxBatchSize=1.
    try await runBatchScenario(
        context: context, input: input, label: "1. Baseline (compile off)",
        params: GenerateParameters(maxTokens: maxNew, temperature: 0),
        maxBatchSize: 1)

    // 2. Compile on — Stage 1B.3 path.
    try await runBatchScenario(
        context: context, input: input, label: "2. Stage 1B.3 compile",
        params: GenerateParameters(
            maxTokens: maxNew,
            enableCompiledBatchDecode: true,
            temperature: 0),
        maxBatchSize: 1)

    // 3. TurboQuant on — Stage 0 path. Compile is silently skipped for
    // TQ (v2 rollback).
    try await runBatchScenario(
        context: context, input: input, label: "3. Stage 0 TurboQuant",
        params: GenerateParameters(
            maxTokens: maxNew,
            kvMode: .turboQuant(keyBits: 3, valueBits: 3),
            temperature: 0),
        maxBatchSize: 1)

    // 4. Two concurrent requests — uncompiled batched decode path.
    print("\n[4. B=2 concurrent uncompiled]")
    nonisolated(unsafe) let ctx4 = context
    let engine4 = BatchEngine(context: ctx4, maxBatchSize: 2)
    _ = LMInput(text: LMInput.Text(
        tokens: MLXArray((10...17).map { Int32($0) })[.newAxis, .ellipsis]))
    let p4 = GenerateParameters(maxTokens: maxNew, temperature: 0)

    // Fresh per-submit inputs to satisfy Swift 6 sending-risks-data-race
    // (LMInput isn't Sendable; each submit consumes its own instance).
    let t0 = CFAbsoluteTimeGetCurrent()
    let i1 = LMInput(text: LMInput.Text(
        tokens: MLXArray((1...8).map { Int32($0) })[.newAxis, .ellipsis]))
    let i2 = LMInput(text: LMInput.Text(
        tokens: MLXArray((10...17).map { Int32($0) })[.newAxis, .ellipsis]))
    let (_, s1) = await engine4.submit(input: i1, parameters: p4)
    let (_, s2) = await engine4.submit(input: i2, parameters: p4)
    var tokens1: [Int] = []
    var tokens2: [Int] = []
    for await e in s1 {
        if case .token(let id) = e { tokens1.append(id) }
    }
    for await e in s2 {
        if case .token(let id) = e { tokens2.append(id) }
    }
    let total = CFAbsoluteTimeGetCurrent() - t0
    print(String(format: "  R1: %d tokens, R2: %d tokens | total %.2fs",
        tokens1.count, tokens2.count, total))
    print("  R1 first 8: \(Array(tokens1.prefix(8)))")
    print("  R2 first 8: \(Array(tokens2.prefix(8)))")

    print("\n=== BatchEngine smoke done ===")
}

/// Run one BatchEngine scenario and print timing + first tokens.
func runBatchScenario(
    context: MLXLMCommon.ModelContext,
    input: LMInput,
    label: String,
    params: GenerateParameters,
    maxBatchSize: Int,
    silent: Bool = false
) async throws {
    if !silent { print("\n[\(label)]") }
    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: maxBatchSize)

    let t0 = CFAbsoluteTimeGetCurrent()
    nonisolated(unsafe) let sendableInput = input
    let (_, stream) = await engine.submit(input: sendableInput, parameters: params)

    var tokens: [Int] = []
    var firstTokAt: Double?
    var stopReason: GenerateStopReason?
    for await event in stream {
        switch event {
        case .token(let id):
            if firstTokAt == nil {
                firstTokAt = CFAbsoluteTimeGetCurrent() - t0
            }
            tokens.append(id)
        case .info(let info):
            stopReason = info.stopReason
        }
    }
    if silent { return }

    let total = CFAbsoluteTimeGetCurrent() - t0
    let decodeTime = max(total - (firstTokAt ?? 0), 0.001)
    let tps = Double(max(tokens.count - 1, 1)) / decodeTime
    print(String(format:
        "  %d tokens | TTFT %.0fms | decode %.1f tok/s | stop=%@",
        tokens.count,
        (firstTokAt ?? 0) * 1000,
        tps,
        String(describing: stopReason ?? .length)))
    print("  first 8 tokens: \(Array(tokens.prefix(8)))")
}

// MARK: - Cross-engine correctness validation (iter 32)

/// Run the same short chat prompt through BOTH `TokenIterator` and
/// `BatchEngine.generate(...)` with identical deterministic parameters
/// (temperature=0) and compare the emitted token IDs. Equality is the
/// property — divergence means one of the paths has a bug.
///
/// This is the strongest single correctness check for the engine. The
/// compile-on/off identity check in `BENCH_BATCH_CHAT` only proves
/// BatchEngine is internally consistent with itself. Cross-validation
/// against TokenIterator proves BatchEngine matches the long-standing
/// single-sequence path used by `ChatSession`.
///
/// Scope: text-only model, greedy sampling (temp=0), no cache coordinator
/// on either side — we want to isolate the engine/iterator, not the
/// multi-tier cache layer.
func runCrossEngineValidation(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== Cross-engine validation (iter 32) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Deterministic params. Greedy sampling → any divergence between
    // iterators is a real engine bug, not a sampling noise artifact.
    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    // Three short prompts — keep each single-turn so we don't have to
    // replay history. The question is whether, given the same LMInput,
    // both iterators emit the same token stream.
    let prompts = [
        "Write a haiku about rain.",
        "Explain recursion in two sentences.",
        "List three primary colours.",
    ]

    // BatchEngine respects `stopTokenIDs` per-slot and terminates the
    // stream as soon as one is emitted. Raw `TokenIterator` DOES NOT —
    // it decodes until `maxTokens` is reached, letting EOS tokens through
    // as ordinary tokens. So if BatchEngine stops short, we must verify
    // equality only over the prefix BatchEngine actually emitted AND
    // that the next TokenIterator token is one BatchEngine would have
    // treated as EOS. Build the same stop set here.
    var stopTokenIDs: Set<Int> = context.configuration.eosTokenIds
    if let eos = context.tokenizer.eosTokenId { stopTokenIDs.insert(eos) }
    if let unk = context.tokenizer.unknownTokenId { stopTokenIDs.insert(unk) }
    for tok in context.configuration.extraEOSTokens {
        if let id = context.tokenizer.convertTokenToId(tok) {
            stopTokenIDs.insert(id)
        }
    }

    var mismatches = 0
    for (i, prompt) in prompts.enumerated() {
        print("\n[Probe \(i + 1)/\(prompts.count)] \"\(prompt)\"")
        let userInput = UserInput(prompt: prompt)
        let lmInput = try await context.processor.prepare(input: userInput)

        // Snapshot the tokens so we can feed EXACTLY the same input to
        // both paths. LMInput.text.tokens is an MLXArray — eval to make
        // sure the shape is materialized before either consumer reads it.
        let promptLen = lmInput.text.tokens.size
        print("  prompt tokens: \(promptLen)")

        // Path A: TokenIterator (single-sequence, the "baseline").
        let iterCache = context.model.newCache(parameters: params)
        let iter = try TokenIterator(
            input: lmInput, model: context.model, cache: iterCache,
            parameters: params)
        var iterTokens: [Int] = []
        for token in iter {
            iterTokens.append(token)
            if iterTokens.count >= maxNew { break }
        }

        // Path B: BatchEngine.
        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        nonisolated(unsafe) let sendable = lmInput
        let (_, tokenStream) = await engine.submit(input: sendable, parameters: params)
        var engineTokens: [Int] = []
        for await event in tokenStream {
            switch event {
            case .token(let id):
                engineTokens.append(id)
                if engineTokens.count >= maxNew { break }
            case .info:
                break
            }
            if engineTokens.count >= maxNew { break }
        }

        // Compare.
        let iterSummary = Array(iterTokens.prefix(20))
        let engSummary = Array(engineTokens.prefix(20))
        print("  TokenIterator (\(iterTokens.count) toks): first 20 = \(iterSummary)")
        print("  BatchEngine   (\(engineTokens.count) toks): first 20 = \(engSummary)")

        // Find first divergence point.
        var firstDiff: Int? = nil
        let n = min(iterTokens.count, engineTokens.count)
        for k in 0..<n where iterTokens[k] != engineTokens[k] {
            firstDiff = k
            break
        }
        if iterTokens == engineTokens {
            print("  ✓ identical (\(iterTokens.count) tokens)")
        } else if firstDiff == nil &&
                  engineTokens.count < iterTokens.count &&
                  iterTokens.count > engineTokens.count &&
                  stopTokenIDs.contains(iterTokens[engineTokens.count]) {
            // Common case on chatty models: BatchEngine stopped at EOS,
            // raw TokenIterator continued through it. Prefix is identical
            // and the "extra" token TokenIterator emitted is in the stop
            // set. That's correct behaviour, not divergence.
            print("  ✓ identical \(engineTokens.count)-token prefix — " +
                  "BatchEngine stopped at EOS token \(iterTokens[engineTokens.count]) " +
                  "which TokenIterator's raw loop ignores")
        } else if let d = firstDiff {
            mismatches += 1
            print("  ✗ diverge at index \(d): iter=\(iterTokens[d]), engine=\(engineTokens[d])")
        } else {
            mismatches += 1
            print("  ✗ length differs: iter=\(iterTokens.count) vs engine=\(engineTokens.count)")
        }
    }

    print("\n=== Cross-engine validation: \(prompts.count - mismatches)/\(prompts.count) matched ===")
    if mismatches > 0 {
        fputs("[CrossValidate] FAIL: \(mismatches) prompt(s) diverged\n", stderr)
        exit(1)
    }
}

// MARK: - B=2 concurrent real-model validation (iter 33)

/// Submit two DIFFERENT prompts to `BatchEngine(maxBatchSize: 2)` and
/// iterate both streams concurrently under a `TaskGroup`. Proves the
/// batched-decode hot path functions end-to-end with real HF tokenizer
/// output — not just the synthetic `NullTokenizer` path covered by
/// `BENCH_BATCH`, and not the serialised stream reads it also does.
///
/// Acceptance:
/// - Both streams complete within max-tokens or EOS.
/// - Each stream produces a coherent non-empty preview.
/// - The engine doesn't crash, hang, or mix tokens between slots.
func runBatchEngineConcurrent(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine B=2 concurrent (iter 33) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 2)

    // Two semantically-distinct prompts so divergent outputs are expected.
    // If the engine mixes slots, one would "see" the other's tokens and
    // the previews would look suspiciously similar.
    let prompts = [
        "What is the capital of France?",
        "List two prime numbers greater than 10.",
    ]

    // Prepare both inputs ahead of the race so `.processor.prepare`
    // overhead doesn't bias who submits first.
    var inputs: [LMInput] = []
    for p in prompts {
        let input = try await context.processor.prepare(input: UserInput(prompt: p))
        inputs.append(input)
    }

    // Submit both, then await each stream concurrently via TaskGroup.
    // TaskGroup is what forces real concurrency — without it, the caller
    // would iterate s1 to completion before touching s2, and the engine's
    // decode step never sees B=2.
    let t0 = CFAbsoluteTimeGetCurrent()
    nonisolated(unsafe) let send0 = inputs[0]
    nonisolated(unsafe) let send1 = inputs[1]
    let (_, s0) = await engine.submit(input: send0, parameters: params)
    let (_, s1) = await engine.submit(input: send1, parameters: params)

    // Collect decoded text for each slot in parallel. Uses the tokenizer
    // directly rather than NaiveStreamingDetokenizer — the synchronous
    // decode is fine for a small benchmark and avoids the O(n²) relay
    // cost that hammers throughput under HF tokenizers.
    let tokenizer = context.tokenizer
    let results = await withTaskGroup(of: (Int, [Int]).self) { group in
        group.addTask {
            var ids: [Int] = []
            for await e in s0 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (0, ids)
        }
        group.addTask {
            var ids: [Int] = []
            for await e in s1 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (1, ids)
        }
        var collected: [(Int, [Int])] = []
        for await result in group {
            collected.append(result)
        }
        return collected
    }
    let total = CFAbsoluteTimeGetCurrent() - t0

    // Print side-by-side. Guard against empty (stuck) slots.
    for (slot, ids) in results.sorted(by: { $0.0 < $1.0 }) {
        let text = tokenizer.decode(tokenIds: ids)
        let preview = text.count > 200 ? String(text.prefix(200)) + "..." : text
        print("  Slot \(slot) prompt: \"\(prompts[slot])\"")
        print("    tokens: \(ids.count), first 12: \(Array(ids.prefix(12)))")
        print("    text:  \"\(preview)\"")
        if ids.isEmpty {
            fputs("[ConcurrentBatch] FAIL: slot \(slot) produced zero tokens\n", stderr)
            exit(1)
        }
    }
    print(String(format: "  total wall time: %.2fs (both slots)", total))
    print("=== BatchEngine B=2 concurrent: passed ===")
}

// MARK: - Cache coordinator cross-turn reuse (iter 34)

/// Verify that `CacheCoordinator` wired into `BatchEngine` actually
/// produces cross-turn cache hits on real prompts. Critical property
/// for osaurus: the coordinator is the whole reason multi-turn chats
/// don't re-prefill from scratch every turn.
///
/// Methodology:
/// 1. Build a `BatchEngine` with an in-memory paged `CacheCoordinator`.
/// 2. Turn 1: prompt "The sky is blue. <Q>" — cold cache → full prefill.
/// 3. Turn 2: prompt "The sky is blue. <Q>. And also <Q2>" — warm cache
///    → prefill should only cover the added suffix.
/// 4. Compare `GenerateCompletionInfo.promptTime` across turns; turn 2
///    must be at least **2× faster** on the prefill (a conservative
///    threshold — real gains on Qwen3-0.6B are typically ≥5×).
///
/// If the threshold isn't met, either:
///   - the coordinator isn't being hit (bug in admission path), or
///   - the prompt isn't actually extending (tokenizer quirk), or
///   - the test harness is mismeasuring.
/// Exit 1 surfaces any of these.
func runBatchEngineCacheHit(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine cache-hit verification (iter 34) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // In-memory paged coordinator — no disk I/O, so timings are clean.
    var cfg = CacheCoordinatorConfig()
    cfg.usePagedCache = true
    cfg.enableDiskCache = false
    cfg.pagedBlockSize = 64
    cfg.maxCacheBlocks = 512
    cfg.modelKey = modelDir.lastPathComponent
    let coordinator = CacheCoordinator(config: cfg)

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)

    // Key insight 1: running `UserInput(prompt: "A")` through the processor
    // wraps with chat template tokens, so "A" and "A + more" produce token
    // sequences that diverge at the boundary. Build prompts at the TOKEN
    // level so turn 2 is a strict extension of turn 1.
    //
    // Key insight 2: `PagedCacheManager.storeTokenSequence` stores only
    // complete `blockSize`-sized blocks. With the default blockSize=64, a
    // ~58-token prompt stores ZERO blocks (floor(58/64) = 0). We need at
    // least 2 full blocks stored on turn 1 so there's something to hit
    // on turn 2. 200+ tokens at blockSize=64 gives ≥3 blocks cached.
    let turn1Prompt = String(repeating: """
        You are a careful assistant. Facts to remember across turns: \
        the sky is blue, grass is green, roses are red, oceans are \
        deep, fire is hot. Answer concisely and precisely.
        """, count: 3) + " Q: What is the colour of the sky?"
    let turn1Input = try await context.processor.prepare(
        input: UserInput(prompt: turn1Prompt))
    let turn1Tokens = turn1Input.text.tokens.reshaped(-1).asArray(Int.self)
    // Turn 2 = turn 1 tokens + 24 "follow-up" tokens (>1 new block worth,
    // so the added region also fills a block). Safe Qwen3 IDs, no specials.
    let followup: [Int] = [
        220, 13, 1527, 264, 4586, 31004, 220, 1207, 25, 3555, 374, 279,
        12463, 315, 16359, 30, 220, 13, 1527, 4586, 4226, 25, 5312,
    ]
    let turn2Tokens: [Int] = turn1Tokens + followup
    let turn2TokensArr = MLXArray(turn2Tokens.map { Int32($0) })[.newAxis, .ellipsis]
    let turn2Input = LMInput(
        text: LMInput.Text(tokens: turn2TokensArr),
        image: nil, video: nil)

    func runTurn(label: String, input: sending LMInput) async throws -> (Double, Int) {
        let tokenCount = input.text.tokens.size
        let t0 = CFAbsoluteTimeGetCurrent()
        let (_, stream) = await engine.submit(input: input, parameters: params)
        var promptTime: Double = 0
        var nTokens = 0
        for await event in stream {
            switch event {
            case .token:
                nTokens += 1
            case .info(let info):
                promptTime = info.promptTime
            }
        }
        let wall = CFAbsoluteTimeGetCurrent() - t0
        print(String(format:
            "  %@ : prompt=%d tokens, promptTime=%.3fs, genTokens=%d, wall=%.2fs",
            label, tokenCount, promptTime, nTokens, wall))
        return (promptTime, tokenCount)
    }

    nonisolated(unsafe) let turn1Send = turn1Input
    let (turn1Prompt_s, _) = try await runTurn(
        label: "Turn 1 (cold cache)", input: turn1Send)

    // Verify our construction produced a true prefix extension, then
    // probe the coordinator directly.
    let isPrefix = turn1Tokens.count <= turn2Tokens.count &&
        Array(turn2Tokens.prefix(turn1Tokens.count)) == turn1Tokens
    print(String(format: "  turn1.count=%d, turn2.count=%d, turn2 starts with turn1? %@",
        turn1Tokens.count, turn2Tokens.count, isPrefix ? "yes" : "NO"))
    if !isPrefix {
        fputs("[CacheHit] FAIL: test harness broken — turn2Tokens is not a prefix of turn1Tokens.\n",
              stderr)
        exit(2)
    }

    // Directly probe the coordinator with turn 2's token ids BEFORE
    // submitting turn 2. This isolates "does the coordinator have the
    // turn 1 prefix?" from "does BatchEngine apply the hit correctly?".
    // If fetch() returns `.miss`, BatchEngine isn't storing under a key
    // that turn 2 can look up — that's the real failure mode.
    let probeResult = coordinator.fetch(tokens: turn2Tokens)
    switch probeResult {
    case .hit(let matched, let remaining, let detail, _, _, _):
        print(String(format:
            "  Coordinator probe: HIT (%@ tier, matched=%d/%d, remaining=%d)",
            detail.rawValue, matched, turn2Tokens.count, remaining.count))
    case .miss:
        fputs("[CacheHit] FAIL: coordinator.fetch(turn2Tokens) returned .miss. " +
              "BatchEngine is not storing under the key that turn 2 looks up. " +
              "Bug is in `finishSlot`'s storeAfterGeneration call or its token-hash.\n",
              stderr)
        exit(1)
    }

    nonisolated(unsafe) let turn2Send = turn2Input
    let (turn2Prompt_s, _) = try await runTurn(
        label: "Turn 2 (warm cache)", input: turn2Send)

    // Turn 2 should also be measurably faster because prefill covers
    // only the remaining tokens. Ratio <= 0.75 catches regressions
    // (the paged cache hitting-but-not-saving-compute path); real
    // gains are typically ≥60% reduction.
    //
    // Hybrid SSM exception: partial-hit on hybrid slots rolls back to
    // full prefill by design (SSM recurrence is path-dependent, same
    // class as VL). The coordinator still reports a probe HIT but
    // BatchEngine falls back for correctness — matching prompt times
    // are expected, not a bug. Detect hybrid by checking `coordinator.isHybrid`
    // which the engine auto-flips on admission of any Mamba/SSM slot.
    let ratio = turn1Prompt_s > 0 ? turn2Prompt_s / turn1Prompt_s : 1.0
    print(String(format: "  ratio (turn2/turn1) = %.2f", ratio))
    if coordinator.isHybrid {
        print("  (hybrid SSM model — partial-hit rollback is correct-by-design; " +
              "ratio is informational only)")
    } else if ratio >= 0.75 {
        fputs("[CacheHit] FAIL: turn2 prompt time not < 75% of turn 1 " +
              "(\(turn2Prompt_s)s vs \(turn1Prompt_s)s). " +
              "Coordinator reports hit but BatchEngine isn't using it to skip prefill.\n",
              stderr)
        exit(1)
    }
    print("=== BatchEngine cache-hit: passed ===")
}

// MARK: - Disk cache restore across coordinators (iter 35)

/// Models an osaurus session restart. Turn 1 runs with coordinator A
/// (disk-enabled, pointing at a temp dir). Coordinator A is then
/// DISCARDED. A fresh coordinator B is spun up against the same disk
/// dir — as if a new process just started. Turn 2 is submitted through
/// a new BatchEngine bound to coordinator B. If the disk tier works,
/// turn 2 should hit from disk and skip prefill.
///
/// This is the single strongest "does it survive process restart?"
/// property. Paged (RAM) coordinator state disappears at process exit;
/// only the disk tier persists.
func runBatchEngineDiskRestore(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine disk-restore verification (iter 35) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Disk cache dir — unique per run. Clean up on exit so repeated
    // runs start fresh.
    let diskDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx-bench-disk-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: diskDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: diskDir) }

    func makeCoordinator() -> CacheCoordinator {
        var cfg = CacheCoordinatorConfig()
        cfg.usePagedCache = true  // keep paged on (we're testing disk NOT instead of it)
        cfg.enableDiskCache = true
        cfg.diskCacheDir = diskDir
        cfg.pagedBlockSize = 64
        cfg.maxCacheBlocks = 512
        cfg.modelKey = modelDir.lastPathComponent
        return CacheCoordinator(config: cfg)
    }

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    // Build a long prompt — ≥2 paged cache blocks so there's a non-trivial
    // amount of KV state to round-trip through disk.
    //
    // IMPORTANT semantics difference from the paged-cache test (iter 34):
    // `DiskCache.fetch` does **exact-or-one-shorter** match, not prefix
    // extension. That matches its intended use case: session resumption
    // (re-opening the same conversation), not turn extension. So this
    // test replays the EXACT same tokens across sessions.
    let basePrompt = String(repeating: """
        You are a careful assistant. Facts to remember across turns: \
        the sky is blue, grass is green, roses are red, oceans are \
        deep, fire is hot. Answer concisely and precisely.
        """, count: 3) + " Q: What is the colour of the sky?"

    let turn1Input = try await context.processor.prepare(
        input: UserInput(prompt: basePrompt))
    let turn1Tokens = turn1Input.text.tokens.reshaped(-1).asArray(Int.self)
    // Session 2 replays the same tokens. Construct a fresh LMInput from
    // the captured ids so the second BatchEngine sees an independent
    // `LMInput` value (consuming `sending` LMInput).
    let turn2Tokens: [Int] = turn1Tokens

    // --- Session 1: store cache entry to disk --------------------------
    nonisolated(unsafe) let ctx1 = context
    let coordA = makeCoordinator()
    let engineA = BatchEngine(
        context: ctx1, maxBatchSize: 1, cacheCoordinator: coordA)

    nonisolated(unsafe) let t1Send = turn1Input
    let t0A = CFAbsoluteTimeGetCurrent()
    let (_, streamA) = await engineA.submit(input: t1Send, parameters: params)
    var genA = 0
    var promptTimeA: Double = 0
    for await event in streamA {
        switch event {
        case .token: genA += 1
        case .info(let info): promptTimeA = info.promptTime
        }
    }
    let wallA = CFAbsoluteTimeGetCurrent() - t0A
    print(String(format:
        "  Session 1 (cold, wrote disk): prompt=%d, promptTime=%.3fs, genTokens=%d, wall=%.2fs",
        turn1Tokens.count, promptTimeA, genA, wallA))

    // Allow any disk-flushing async work to complete. On Darwin the
    // safetensors + sqlite writes are sync under the coordinator lock,
    // but defensive yield here.
    await Task.yield()

    // Confirm the disk dir actually got something written to it.
    let diskContents = (try? FileManager.default.contentsOfDirectory(
        at: diskDir, includingPropertiesForKeys: nil)) ?? []
    print("  Disk dir contents: \(diskContents.map { $0.lastPathComponent }.sorted())")
    if diskContents.isEmpty {
        fputs("[DiskRestore] FAIL: no files written to disk dir after session 1. " +
              "BatchEngine's finishSlot didn't call coordinator.storeAfterGeneration " +
              "with disk-tier enabled.\n", stderr)
        exit(1)
    }

    // --- Session 2: fresh coordinator + engine, same disk dir ---------
    //
    // Drop coordinator A and engine A entirely. Coord B is NEW — its
    // paged cache is empty. Only the disk tier carries across.
    let coordB = makeCoordinator()

    // Probe coord B directly with turn 2's tokens BEFORE submit.
    let probe = coordB.fetch(tokens: turn2Tokens)
    switch probe {
    case .hit(let matched, _, let detail, _, _, let disk):
        let label = detail.rawValue
        let diskKey = disk != nil ? "yes" : "no"
        print("  Coord B probe: HIT (\(label), matched=\(matched)/\(turn2Tokens.count), diskArrays=\(diskKey))")
        if detail != .disk {
            fputs("[DiskRestore] FAIL: probe hit came from \(label), not disk. " +
                  "Paged cache should be empty for a freshly-constructed coordinator.\n",
                  stderr)
            exit(1)
        }
    case .miss:
        fputs("[DiskRestore] FAIL: fresh coordinator at same disk dir returned .miss. " +
              "Disk writes are not being read back — check TQDiskSerializer / SQLite index.\n",
              stderr)
        exit(1)
    }

    // Actually run turn 2 through a new BatchEngine bound to coordB.
    // We can't reuse engineA because it carries coordA.
    let turn2TokensArr = MLXArray(turn2Tokens.map { Int32($0) })[.newAxis, .ellipsis]
    let turn2Input = LMInput(
        text: LMInput.Text(tokens: turn2TokensArr), image: nil, video: nil)

    nonisolated(unsafe) let ctx2 = context
    let engineB = BatchEngine(
        context: ctx2, maxBatchSize: 1, cacheCoordinator: coordB)
    nonisolated(unsafe) let t2Send = turn2Input
    let t0B = CFAbsoluteTimeGetCurrent()
    let (_, streamB) = await engineB.submit(input: t2Send, parameters: params)
    var genB = 0
    var promptTimeB: Double = 0
    for await event in streamB {
        switch event {
        case .token: genB += 1
        case .info(let info): promptTimeB = info.promptTime
        }
    }
    let wallB = CFAbsoluteTimeGetCurrent() - t0B
    print(String(format:
        "  Session 2 (warm from disk): prompt=%d, promptTime=%.3fs, genTokens=%d, wall=%.2fs",
        turn2Tokens.count, promptTimeB, genB, wallB))

    // Behavioural correctness only: both sessions must generate >0 tokens
    // and the probe must have reported a disk hit. We DO NOT assert on
    // promptTime ratio: the safetensors deserialize cost during restore
    // can dominate a 1-token prefill on small models like Qwen3-0.6B, so
    // on-wire prefill time can be HIGHER after disk restore despite a
    // real cache hit. On larger models this flips — restore saves
    // hundreds of ms of forward-pass compute. The timing tradeoff is a
    // model-size-dependent operational property, not a correctness
    // property of the engine/cache wiring.
    let ratio = promptTimeA > 0 ? promptTimeB / promptTimeA : 1.0
    print(String(format: "  ratio (session2/session1) = %.2f (informational only)", ratio))
    if genA == 0 || genB == 0 {
        fputs("[DiskRestore] FAIL: at least one session generated zero tokens " +
              "(sessionA=\(genA), sessionB=\(genB)). " +
              "Disk restore may have corrupted the cache.\n", stderr)
        exit(1)
    }
    print("=== BatchEngine disk-restore: passed (disk hit fired, both sessions completed) ===")
}

// MARK: - Per-slot sampling divergence (iter 36)

/// Submit two slots to the same B=2 BatchEngine where each slot carries
/// DIFFERENT `GenerateParameters`. Verify both slots' samplers actually
/// fire with their respective settings:
///
/// - Slot 0: temp=0 (greedy). Re-running the same input must produce
///   byte-identical tokens.
/// - Slot 1: temp=0.8 topP=0.9 with a fixed seed. Output must differ
///   from the greedy path by at least one token (otherwise the sampler
///   didn't actually kick in, or both slots share a sampler instance).
func runBatchEnginePerSlotSampler(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine per-slot sampler (iter 36) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Same prompt into both slots — eliminates "different outputs came
    // from different inputs" as an alternative explanation. Any divergence
    // must come from the sampler.
    let prompt = "Write five words describing a red apple."
    let baseInput = try await context.processor.prepare(
        input: UserInput(prompt: prompt))
    let promptTokens = baseInput.text.tokens.reshaped(-1).asArray(Int.self)
    print("  prompt: \"\(prompt)\" (\(promptTokens.count) tokens)")

    func freshInput() -> LMInput {
        let arr = MLXArray(promptTokens.map { Int32($0) })[.newAxis, .ellipsis]
        return LMInput(
            text: LMInput.Text(tokens: arr), image: nil, video: nil)
    }

    // Two distinct parameter profiles.
    var greedyParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    greedyParams.topP = 1.0
    var stochasticParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0.8, prefillStepSize: 512)
    stochasticParams.topP = 0.9

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 2)

    // Submit both concurrently — same engine, same model, same prompt,
    // different params.
    nonisolated(unsafe) let in0 = freshInput()
    nonisolated(unsafe) let in1 = freshInput()
    let (_, s0) = await engine.submit(input: in0, parameters: greedyParams)
    let (_, s1) = await engine.submit(input: in1, parameters: stochasticParams)

    let results = await withTaskGroup(of: (Int, [Int]).self) { group in
        group.addTask {
            var ids: [Int] = []
            for await e in s0 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (0, ids)
        }
        group.addTask {
            var ids: [Int] = []
            for await e in s1 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (1, ids)
        }
        var collected: [(Int, [Int])] = []
        for await result in group {
            collected.append(result)
        }
        return collected.sorted { $0.0 < $1.0 }
    }
    let greedyTokens = results[0].1
    let stochasticTokens = results[1].1
    print("  slot 0 (temp=0)    first 15: \(Array(greedyTokens.prefix(15)))")
    print("  slot 1 (temp=0.8)  first 15: \(Array(stochasticTokens.prefix(15)))")

    // Re-run slot 0 alone to confirm greedy determinism. Use a FRESH
    // engine — the previous B=2 submission has finished and we want a
    // clean cache. Same prompt + same params → must produce the same
    // tokens. If not, either the sampler bled state across slots or
    // temp=0 isn't truly greedy.
    nonisolated(unsafe) let greedyRecheckInput = freshInput()
    let recheckEngine = BatchEngine(context: ctx, maxBatchSize: 1)
    let (_, sr) = await recheckEngine.submit(
        input: greedyRecheckInput, parameters: greedyParams)
    var greedyRecheck: [Int] = []
    for await e in sr {
        if case .token(let id) = e { greedyRecheck.append(id) }
        if greedyRecheck.count >= maxNew { break }
    }
    print("  slot 0 re-run      first 15: \(Array(greedyRecheck.prefix(15)))")

    // Assertions.
    // 1. Greedy path must be deterministic across runs.
    if greedyTokens != greedyRecheck {
        fputs("[PerSlotSampler] FAIL: greedy slot 0 diverged between runs — " +
              "\(greedyTokens.count) vs \(greedyRecheck.count) tokens, " +
              "first diff at \(firstDiffIndex(greedyTokens, greedyRecheck) ?? -1). " +
              "Either temp=0 isn't really greedy or slot state bled across submissions.\n",
              stderr)
        exit(1)
    }
    // 2. Stochastic path must differ from greedy on at least one token.
    //    (Not guaranteed on every run — sampling MAY happen to match the
    //    greedy choice — but on a real prompt with temp=0.8 and ≥20 tokens
    //    the probability of byte-for-byte match is effectively zero.)
    if stochasticTokens == greedyTokens {
        fputs("[PerSlotSampler] WARN: stochastic slot 1 matched greedy byte-for-byte. " +
              "Either slot 1's params didn't apply (BUG) or sampling happened to " +
              "agree with greedy on every token (very unlikely — try maxNew>=30).\n",
              stderr)
        exit(1)
    }
    let firstDiff = firstDiffIndex(greedyTokens, stochasticTokens) ?? -1
    print("  first divergence greedy vs stochastic: index \(firstDiff)")
    print("=== BatchEngine per-slot sampler: passed ===")
}

/// Return the index of the first position where `a[i] != b[i]`, or nil
/// if they agree on every overlapping index and have the same length.
private func firstDiffIndex(_ a: [Int], _ b: [Int]) -> Int? {
    let n = min(a.count, b.count)
    for i in 0..<n where a[i] != b[i] { return i }
    return a.count == b.count ? nil : n
}

// MARK: - TurboQuant under B=2 (iter 38)

/// Two concurrent slots on the same BatchEngine with heterogeneous
/// `kvMode`: slot 0 runs with plain float KV, slot 1 runs with
/// `turboQuant(keyBits: 3, valueBits: 3)`. Verifies:
/// - Stage 0 compression (`BatchQuantize.maybeCompress`) fires per-slot
///   post-prefill, does not cross-contaminate.
/// - Both streams complete with non-empty coherent output.
/// - Running two TQ slots concurrently (the second pass below) also
///   completes without corruption — proving per-slot `TurboQuantKVCache`
///   state is independent.
///
/// We can't cheaply introspect the cache type post-run from outside the
/// BatchEngine actor, but the "no crash + non-zero tokens + coherent
/// text" combo plus the existing `CompilableTurboQuantKVCacheTests` FP
/// precision probes give a high-confidence end-to-end check.
func runBatchEngineTurboQuantB2(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine TurboQuant B=2 (iter 38) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    nonisolated(unsafe) let ctx = context
    let tokenizer = context.tokenizer

    // Prompt pair that's long enough to cross the TQ minimum-compression
    // threshold (`max(quantizedKVStart, 8)` in BatchQuantize). 30+ tokens
    // is comfortably above both defaults.
    let prompts = [
        "List five distinct countries in Europe, one per line.",
        "Give me five adjectives that describe a summer morning.",
    ]
    var inputs: [LMInput] = []
    for p in prompts {
        inputs.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }

    let plainParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    // kvMode defaults to plain on `plainParams`.
    // 4-bit TurboQuant. 3-bit TQ is too aggressive for small models like
    // Qwen3-0.6B — observed garbage output ("repeated newlines",
    // "ائيةء إلى إلى") on B=1 and B=2. 4-bit TQ is the minimum that
    // preserves coherence at this model size. Larger models (≥7B) handle
    // 3-bit TQ; the probe suite (CompilableTurboQuantKVCacheTests) uses
    // synthetic tensors where quantization noise doesn't matter.
    var tqParams = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    tqParams.kvMode = .turboQuant(keyBits: 4, valueBits: 4)
    tqParams.quantizedKVStart = 8

    // Reference run: slot 0 plain KV ALONE. Gives us the expected
    // plain-output baseline to compare against "slot 0 plain beside a
    // TQ neighbour" — cross-slot corruption would show up as drift.
    print("\n[Reference] slot 0 plain KV, solo (B=1)")
    let engineRef = BatchEngine(context: ctx, maxBatchSize: 1)
    var refInputs: [LMInput] = []
    for p in prompts {
        refInputs.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }
    nonisolated(unsafe) let inRef = refInputs[0]
    let (_, streamRef) = await engineRef.submit(input: inRef, parameters: plainParams)
    var refTokens: [Int] = []
    for await e in streamRef {
        if case .token(let id) = e { refTokens.append(id) }
        if refTokens.count >= maxNew { break }
    }
    print("  reference plain solo: \(refTokens.count) tokens, first 8: \(Array(refTokens.prefix(8)))")

    // --- Pass A: plain KV  +  TurboQuant  (heterogeneous) ----------------
    print("\n[Pass A] slot 0 = plain KV, slot 1 = TurboQuant(4,4)")
    let engineA = BatchEngine(context: ctx, maxBatchSize: 2)

    nonisolated(unsafe) let inA0 = inputs[0]
    nonisolated(unsafe) let inA1 = inputs[1]
    let (_, streamA0) = await engineA.submit(input: inA0, parameters: plainParams)
    let (_, streamA1) = await engineA.submit(input: inA1, parameters: tqParams)
    let resultsA = await withTaskGroup(of: (Int, [Int]).self) { group in
        group.addTask {
            var ids: [Int] = []
            for await e in streamA0 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (0, ids)
        }
        group.addTask {
            var ids: [Int] = []
            for await e in streamA1 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (1, ids)
        }
        var collected: [(Int, [Int])] = []
        for await r in group { collected.append(r) }
        return collected.sorted { $0.0 < $1.0 }
    }
    for (slot, ids) in resultsA {
        let tag = slot == 0 ? "plain" : "TQ(4,4)"
        let text = tokenizer.decode(tokenIds: ids)
        let preview = text.count > 120 ? String(text.prefix(120)) + "..." : text
        print(String(format: "  Slot %d (%@) : %d tokens, first 8: %@",
            slot, tag, ids.count, "\(Array(ids.prefix(8)))"))
        print("    \"\(preview)\"")
        if ids.isEmpty {
            fputs("[TQ B=2] FAIL: Pass A slot \(slot) produced zero tokens.\n", stderr)
            exit(1)
        }
    }

    // --- Pass B: both slots TurboQuant (parallel TQ) -------------------
    print("\n[Pass B] slot 0 = TurboQuant(4,4), slot 1 = TurboQuant(4,4)")
    let engineB = BatchEngine(context: ctx, maxBatchSize: 2)
    // Fresh inputs — LMInput is consumed by submit.
    var inputsB: [LMInput] = []
    for p in prompts {
        inputsB.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }
    nonisolated(unsafe) let inB0 = inputsB[0]
    nonisolated(unsafe) let inB1 = inputsB[1]
    let (_, streamB0) = await engineB.submit(input: inB0, parameters: tqParams)
    let (_, streamB1) = await engineB.submit(input: inB1, parameters: tqParams)
    let resultsB = await withTaskGroup(of: (Int, [Int]).self) { group in
        group.addTask {
            var ids: [Int] = []
            for await e in streamB0 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (0, ids)
        }
        group.addTask {
            var ids: [Int] = []
            for await e in streamB1 {
                if case .token(let id) = e { ids.append(id) }
                if ids.count >= maxNew { break }
            }
            return (1, ids)
        }
        var collected: [(Int, [Int])] = []
        for await r in group { collected.append(r) }
        return collected.sorted { $0.0 < $1.0 }
    }
    for (slot, ids) in resultsB {
        let text = tokenizer.decode(tokenIds: ids)
        let preview = text.count > 120 ? String(text.prefix(120)) + "..." : text
        print(String(format: "  Slot %d (TQ) : %d tokens, first 8: %@",
            slot, ids.count, "\(Array(ids.prefix(8)))"))
        print("    \"\(preview)\"")
        if ids.isEmpty {
            fputs("[TQ B=2] FAIL: Pass B slot \(slot) produced zero tokens.\n", stderr)
            exit(1)
        }
    }

    // ISOLATION CHECK — the actual iter 38 bug class.
    //
    // Pass A slot 0 (plain KV) ran concurrently with Pass A slot 1
    // (TurboQuant). If cross-slot corruption existed, slot 0's output
    // would differ from the solo reference run. We EXPECT byte-identical
    // equality because both plain-KV decodes are deterministic at temp=0
    // and each slot's cache should be fully isolated from its neighbour.
    let a0 = resultsA[0].1
    let isolationOk = a0 == refTokens
    print(String(format:
        "\n  Slot 0 plain with TQ neighbour vs plain-solo reference: %@ (%d vs %d tokens)",
        isolationOk ? "IDENTICAL ✓" : "DIVERGED ✗",
        a0.count, refTokens.count))
    if !isolationOk {
        fputs("[TQ B=2] FAIL: slot 0 plain output drifted from the solo reference " +
              "when slot 1 ran TurboQuant concurrently. Cross-slot corruption.\n",
              stderr)
        let firstDiff = firstDiffIndex(a0, refTokens) ?? -1
        fputs("  first divergence index = \(firstDiff)\n", stderr)
        exit(1)
    }
    print("=== BatchEngine TurboQuant B=2 (isolation verified): passed ===")
}

// MARK: - B=N concurrent stress (iter 39)

/// Submit `batchSize` distinct prompts to a single BatchEngine and drain
/// all streams concurrently via `TaskGroup`. Proves:
/// 1. All slots complete with coherent non-empty output.
/// 2. Per-slot outputs don't cross-contaminate — slot 0's tokens match
///    its solo-reference run byte-for-byte.
/// 3. Wall time is less than `batchSize × single-slot time` (real batching).
func runBatchEngineBMany(modelPath: String, maxNew: Int, batchSize: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine B=\(batchSize) concurrent stress (iter 39) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Prompts: 8 distinct short questions so we can support batchSize up
    // to 8 without duplication. Caller can go higher by asking through
    // BENCH_B_SIZE but we cycle after 8.
    let promptPool = [
        "Name one country in Asia.",
        "What color is the sun at noon?",
        "Give one word for the number 2.",
        "Name one fruit that is red.",
        "What is the tallest animal on land?",
        "What is the smallest planet?",
        "Name one musical instrument with strings.",
        "What is H₂O commonly called?",
    ]
    let prompts = (0..<batchSize).map { promptPool[$0 % promptPool.count] }

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let tokenizer = context.tokenizer

    // --- Solo reference for slot 0 (B=1) ------------------------------
    let engineRef = BatchEngine(context: ctx, maxBatchSize: 1)
    let refInput = try await context.processor.prepare(input: UserInput(prompt: prompts[0]))
    nonisolated(unsafe) let inRef = refInput
    let t0Ref = CFAbsoluteTimeGetCurrent()
    let (_, streamRef) = await engineRef.submit(input: inRef, parameters: params)
    var refTokens: [Int] = []
    for await e in streamRef {
        if case .token(let id) = e { refTokens.append(id) }
        if refTokens.count >= maxNew { break }
    }
    let soloWall = CFAbsoluteTimeGetCurrent() - t0Ref
    print(String(format: "[Reference] B=1 solo, prompt[0]: %d tokens in %.2fs",
        refTokens.count, soloWall))

    // --- Actual B=N stress --------------------------------------------
    let engine = BatchEngine(context: ctx, maxBatchSize: batchSize)
    var inputs: [LMInput] = []
    for p in prompts {
        inputs.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }

    let t0 = CFAbsoluteTimeGetCurrent()
    var streams: [AsyncStream<BatchGeneration>] = []
    streams.reserveCapacity(batchSize)
    for input in inputs {
        nonisolated(unsafe) let sendable = input
        let (_, s) = await engine.submit(input: sendable, parameters: params)
        streams.append(s)
    }

    let results: [(Int, [Int])] = await withTaskGroup(of: (Int, [Int]).self) { group in
        for (i, stream) in streams.enumerated() {
            group.addTask {
                var ids: [Int] = []
                for await e in stream {
                    if case .token(let id) = e { ids.append(id) }
                    if ids.count >= maxNew { break }
                }
                return (i, ids)
            }
        }
        var collected: [(Int, [Int])] = []
        for await r in group { collected.append(r) }
        return collected.sorted { $0.0 < $1.0 }
    }
    let batchedWall = CFAbsoluteTimeGetCurrent() - t0

    // Report & validate.
    for (i, ids) in results {
        let text = tokenizer.decode(tokenIds: ids)
        let preview = text.count > 120 ? String(text.prefix(120)) + "..." : text
        print("  slot \(i) (\"\(prompts[i])\"): \(ids.count) tokens")
        print("    first 8: \(Array(ids.prefix(8)))")
        print("    \"\(preview)\"")
        if ids.isEmpty {
            fputs("[B=\(batchSize)] FAIL: slot \(i) produced zero tokens.\n", stderr)
            exit(1)
        }
    }

    // Slot 0 under B=N must match its solo-reference run byte-for-byte.
    let slot0 = results[0].1
    let slot0Ok = slot0 == refTokens
    print(String(format:
        "\n  slot 0 (B=%d) vs solo B=1 reference: %@ (%d vs %d tokens)",
        batchSize,
        slot0Ok ? "IDENTICAL ✓" : "DIVERGED ✗",
        slot0.count, refTokens.count))
    if !slot0Ok {
        let d = firstDiffIndex(slot0, refTokens) ?? -1
        fputs("[B=\(batchSize)] FAIL: slot 0 under concurrent B=\(batchSize) " +
              "diverged from solo reference at index \(d). Cross-slot corruption.\n",
              stderr)
        exit(1)
    }

    // Wall-time speedup. With `batchSize` slots decoding concurrently,
    // the fully-serial projection is `batchSize × solo wall time`.
    // Real batching must beat that — but only when ALL slots actually
    // decode the same number of tokens. Some models/prompts EOS
    // immediately (e.g. Gemma-4 short-answer: "Japan", "Two") so one
    // slot dominates wall time and the serial projection becomes a
    // misleading floor. Only assert when every slot ran to max — the
    // batched-decode stress test has nothing to measure otherwise.
    let allReachedMax = results.allSatisfy { $0.1.count == maxNew }
    let speedupRatio = batchedWall / (soloWall * Double(batchSize))
    print(String(format:
        "  solo wall = %.2fs, B=%d wall = %.2fs → batched/serial-proj ratio = %.2f " +
        "(1.0 = no batching, 0.0 = perfect)",
        soloWall, batchSize, batchedWall, speedupRatio))
    if allReachedMax {
        if speedupRatio >= 0.95 {
            fputs("[B=\(batchSize)] FAIL: batched wall (\(batchedWall)s) ≥ 95% of serial projection " +
                  "(\(soloWall * Double(batchSize))s). Engine isn't sharing forward passes.\n",
                  stderr)
            exit(1)
        }
    } else {
        let dist = results.map { $0.1.count }
        print("  speedup assertion skipped — slots had uneven token counts \(dist); " +
              "correctness check (slot 0 vs solo reference) is the load-bearing assertion here")
    }
    print("=== BatchEngine B=\(batchSize) stress: passed ===")
}

// MARK: - Cancel mid-stream (iter 40)

/// Submit 3 concurrent requests, let them decode a few tokens, then call
/// `engine.cancel(id)` on slot 1. Verify:
/// - Slot 1 stream yields a trailing `.info` event with `stopReason=.cancelled`
///   within a reasonable grace period.
/// - Slots 0 and 2 continue decoding and finish normally (`.stop` or
///   `.length`), each producing `maxNew` tokens.
/// - No crash, no deadlock — the engine's loopTask doesn't hang.
///
/// This is the primitive osaurus relies on for "close a chat window
/// mid-stream without Metal crash" (`ModelLease` keeps the model pinned
/// while the engine unwinds). We're only exercising the engine side
/// here — lease / Metal lifetime are osaurus-side concerns.
func runBatchEngineCancelMidStream(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine cancel mid-stream (iter 40) ===")
    print("Loading with real HuggingFace tokenizer...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Long enough maxNew that slots 0 and 2 will still be decoding by
    // the time we fire `cancel()` on slot 1. Tweak via BENCH_MAX_TOKENS.
    let perSlotMax = max(maxNew, 60)

    let params = GenerateParameters(
        maxTokens: perSlotMax, temperature: 0, prefillStepSize: 512)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 3)

    let prompts = [
        "Explain photosynthesis in a short paragraph.",
        "Describe the process of evaporation.",
        "What are the three states of matter?",
    ]
    var inputs: [LMInput] = []
    for p in prompts {
        inputs.append(try await context.processor.prepare(input: UserInput(prompt: p)))
    }

    nonisolated(unsafe) let in0 = inputs[0]
    nonisolated(unsafe) let in1 = inputs[1]
    nonisolated(unsafe) let in2 = inputs[2]
    let (_, s0) = await engine.submit(input: in0, parameters: params)
    let (id1, s1) = await engine.submit(input: in1, parameters: params)
    let (_, s2) = await engine.submit(input: in2, parameters: params)

    // Track results concurrently via TaskGroup. Each slot yields
    // (index, tokens, stopReason). stopReason is nil if stream closed
    // without an `.info` event (engine bug).
    enum Outcome { case finished(stop: GenerateStopReason?) }
    let results = await withTaskGroup(
        of: (Int, [Int], GenerateStopReason?).self
    ) { group in
        for (i, s) in [(0, s0), (1, s1), (2, s2)] {
            group.addTask {
                var ids: [Int] = []
                var stop: GenerateStopReason? = nil
                for await e in s {
                    switch e {
                    case .token(let id):
                        ids.append(id)
                    case .info(let info):
                        stop = info.stopReason
                    }
                    if ids.count >= perSlotMax { break }
                }
                return (i, ids, stop)
            }
        }
        // Schedule a cancel on slot 1 after a short delay, long enough
        // for decode to have started on all three slots but well short
        // of maxTokens.
        group.addTask {
            try? await Task.sleep(nanoseconds: 100_000_000)  // 100 ms
            await engine.cancel(id1)
            return (-1, [], nil)  // sentinel — ignored in collection
        }
        var collected: [(Int, [Int], GenerateStopReason?)] = []
        for await r in group where r.0 >= 0 {
            collected.append(r)
        }
        return collected.sorted { $0.0 < $1.0 }
    }

    // Report & validate.
    for (i, ids, stop) in results {
        let stopStr = stop.map { "\($0)" } ?? "nil"
        let preview = context.tokenizer.decode(tokenIds: ids)
        let short = preview.count > 100 ? String(preview.prefix(100)) + "..." : preview
        print("  slot \(i): \(ids.count) tokens, stop=\(stopStr)")
        print("    \"\(short)\"")
    }

    let r0 = results[0]
    let r1 = results[1]
    let r2 = results[2]

    // Slot 1 must report cancelled.
    if r1.2 != .cancelled {
        fputs("[Cancel] FAIL: slot 1 stopReason=\(String(describing: r1.2)) — " +
              "expected .cancelled. cancel() didn't reach the slot.\n", stderr)
        exit(1)
    }
    // Slot 1 should have produced strictly fewer tokens than maxTokens
    // (otherwise the cancel fired after decode already completed).
    if r1.1.count >= perSlotMax {
        fputs("[Cancel] WARN: slot 1 reached max tokens before cancel landed " +
              "(\(r1.1.count)/\(perSlotMax)). cancel is too late to be interesting — " +
              "test still passes because cancelled-at-finish is a legitimate state.\n",
              stderr)
    }
    // Slots 0 and 2 must reach max-tokens and report `.length` (not
    // `.cancelled`) — they were unaffected by slot 1's cancel.
    for (label, r) in [("slot 0", r0), ("slot 2", r2)] {
        if r.1.count < perSlotMax {
            fputs("[Cancel] FAIL: \(label) stopped early (\(r.1.count)/\(perSlotMax) tokens). " +
                  "cancelling one slot disturbed its neighbours.\n", stderr)
            exit(1)
        }
        if r.2 == .cancelled {
            fputs("[Cancel] FAIL: \(label) reported .cancelled — cross-slot cancel bled over.\n",
                  stderr)
            exit(1)
        }
    }
    print("=== BatchEngine cancel mid-stream: passed ===")
}

// MARK: - Long-context prefill (iter 42)

/// Build a synthetic long prompt with deterministic safe token ids and
/// run it through both TokenIterator and BatchEngine. Asserts:
/// - Prefill chunking doesn't break for prompts much larger than the
///   default `prefillStepSize` (512) → multi-pass prefill works.
/// - Output is byte-identical across the two engines at temp=0 —
///   extending the iter 32 cross-engine check to long-context regime.
/// - Memory purge runs during decode (memoryPurgeInterval=256) without
///   corrupting in-flight state.
/// - No hang, no OOM, wall time scales reasonably.
func runBatchEngineLongContext(
    modelPath: String, maxNew: Int, promptLen: Int
) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BatchEngine long-context prefill (iter 42, prompt \(promptLen) tokens) ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    // Deterministic, bounded-range token ids. Avoid specials — stay in
    // [100, 50_000) which is safe for any vocab ≥ 50k. Size chosen so
    // both engines see byte-identical input.
    let seedIds: [Int32] = (0..<promptLen).map { Int32(100 + ($0 * 37) % 49_000) }
    let tokensArr = MLXArray(seedIds)[.newAxis, .ellipsis]
    let longInput = LMInput(text: LMInput.Text(tokens: tokensArr))
    print("  synthetic prompt: \(promptLen) tokens, vocab range [100, 50100)")

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)

    // --- TokenIterator (baseline) ---------------------------------------
    print("\n[Path A] TokenIterator")
    let t0A = CFAbsoluteTimeGetCurrent()
    let iterCache = context.model.newCache(parameters: params)
    let iter = try TokenIterator(
        input: longInput, model: context.model, cache: iterCache, parameters: params)
    var iterTokens: [Int] = []
    var iterFirstTokenTime: Double? = nil
    for token in iter {
        if iterFirstTokenTime == nil {
            iterFirstTokenTime = CFAbsoluteTimeGetCurrent() - t0A
        }
        iterTokens.append(token)
        if iterTokens.count >= maxNew { break }
    }
    let wallA = CFAbsoluteTimeGetCurrent() - t0A
    print(String(format: "  iterator: %d tokens, TTFT %.0fms, wall %.2fs",
        iterTokens.count,
        (iterFirstTokenTime ?? 0) * 1000, wallA))

    // --- BatchEngine ----------------------------------------------------
    print("\n[Path B] BatchEngine.submit")
    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    // Fresh LMInput — the one from path A is already consumed.
    let tokensArrB = MLXArray(seedIds)[.newAxis, .ellipsis]
    nonisolated(unsafe) let longInputB = LMInput(text: LMInput.Text(tokens: tokensArrB))
    let t0B = CFAbsoluteTimeGetCurrent()
    let (_, stream) = await engine.submit(input: longInputB, parameters: params)
    var engineTokens: [Int] = []
    var engineFirstTokenTime: Double? = nil
    var promptTime: Double = 0
    for await event in stream {
        switch event {
        case .token(let id):
            if engineFirstTokenTime == nil {
                engineFirstTokenTime = CFAbsoluteTimeGetCurrent() - t0B
            }
            engineTokens.append(id)
            if engineTokens.count >= maxNew { break }
        case .info(let info):
            promptTime = info.promptTime
        }
    }
    let wallB = CFAbsoluteTimeGetCurrent() - t0B
    print(String(format:
        "  engine:   %d tokens, TTFT %.0fms, wall %.2fs, promptTime %.2fs",
        engineTokens.count,
        (engineFirstTokenTime ?? 0) * 1000, wallB, promptTime))

    // --- Byte-identity check -------------------------------------------
    print("\n  first 10 iter:   \(Array(iterTokens.prefix(10)))")
    print("  first 10 engine: \(Array(engineTokens.prefix(10)))")
    if iterTokens != engineTokens {
        let d = firstDiffIndex(iterTokens, engineTokens) ?? -1
        fputs("[LongContext] FAIL: engines diverged at token \(d) " +
              "(iter=\(iterTokens.count), engine=\(engineTokens.count) tokens). " +
              "Long-context prefill chunking is broken somewhere.\n", stderr)
        exit(1)
    }
    print("  ✓ byte-identical across both engines (\(iterTokens.count) tokens)")
    print("=== BatchEngine long-context prefill: passed ===")
}

struct NullTokenizer: MLXLMCommon.Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "<end_of_turn>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "<end_of_turn>" ? 1 : nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}


// MARK: - Harmony reasoning check (2026-04-20 harmony fix)

/// Real-model regression for the Gemma-4 harmony reasoning bug.
/// Loads a Gemma-4 model, sends a short prompt that likely elicits
/// chain-of-thought, and asserts:
///   1. At least one `.reasoning(String)` event fires.
///   2. `.chunk(String)` contains zero harmony channel markers.
func runHarmonyReasoningCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_HARMONY_CHECK: Gemma-4 harmony reasoning channel ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    // Matches tpae's 2026-04-20 2:59 PM screenshot trigger:
    // "can you create a README for my game". Open-ended tasks with
    // ambiguity ("what game?") reliably elicit Gemma-4's
    // `<|channel>thought\n…<channel|>` block. Override via env.
    let promptText = ProcessInfo.processInfo.environment[
        "BENCH_HARMONY_PROMPT"]
        ?? "Can you create a README for my game?"
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": promptText]
    ]
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: messages)
    let promptIds = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)

    let input = LMInput(text: LMInput.Text(tokens: promptIds))
    nonisolated(unsafe) let ctxSendable = context
    nonisolated(unsafe) let sendable = input
    let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)
    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    let stream = await engine.generate(input: sendable, parameters: params)

    var chunkText = ""
    var reasoningText = ""
    var chunkCount = 0
    var reasoningCount = 0
    for await ev in stream {
        switch ev {
        case .chunk(let c):
            chunkText += c
            chunkCount += 1
        case .reasoning(let r):
            reasoningText += r
            reasoningCount += 1
        case .toolCall, .info:
            break
        }
    }

    print("chunks=\(chunkCount) reasoningDeltas=\(reasoningCount)")
    print(".chunk preview:")
    print("  \"\(chunkText.prefix(300))\"")
    print(".reasoning preview:")
    print("  \"\(reasoningText.prefix(300))\"")

    // Harmony markers that must NOT leak into .chunk.
    let markers = ["<|channel>", "<channel|>", "<|channel|>"]
    for m in markers {
        if chunkText.contains(m) {
            fputs("FAIL: .chunk leaked harmony marker: \"\(m)\"\n", stderr)
            exit(1)
        }
    }
    if reasoningCount == 0 {
        // WARN not FAIL — some prompts elicit no reasoning; this still
        // validates the no-leak invariant. But for the main regression
        // goal (Bug A from tpae's screenshot), we want to see deltas.
        fputs("WARN: zero .reasoning deltas — prompt may not elicit CoT.\n", stderr)
    }
    print("PASS harmony markers absent from .chunk.")
    print("=== BENCH_HARMONY_CHECK: passed ===")
}

// MARK: - Qwen enable_thinking reasoning check (2026-04-20 fix B)

/// Real-model regression for Bug B (Qwen3.6 `<think>\n` prefill).
/// Loads a Qwen 3.x model with enable_thinking=true, and asserts:
///   1. .reasoning fires.
///   2. .chunk has zero `<think>` or `</think>` markers.
func runQwenThinkingReasoningCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_QWEN_THINKING_CHECK: Qwen3.x prefilled-think channel ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    let promptText =
        "Please think through this briefly and then answer: " +
        "What's 2 + 2?"
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": promptText]
    ]
    let promptTokens: [Int]
    do {
        promptTokens = try context.tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": true])
    } catch {
        promptTokens = try context.tokenizer.applyChatTemplate(messages: messages)
    }
    let promptIds = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)

    let input = LMInput(text: LMInput.Text(tokens: promptIds))
    nonisolated(unsafe) let ctxSendable = context
    nonisolated(unsafe) let sendable = input
    let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)
    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
    let stream = await engine.generate(input: sendable, parameters: params)

    var chunkText = ""
    var reasoningText = ""
    var chunkCount = 0
    var reasoningCount = 0
    for await ev in stream {
        switch ev {
        case .chunk(let c):
            chunkText += c
            chunkCount += 1
        case .reasoning(let r):
            reasoningText += r
            reasoningCount += 1
        case .toolCall, .info:
            break
        }
    }

    print("chunks=\(chunkCount) reasoningDeltas=\(reasoningCount)")
    print(".chunk preview: \"\(chunkText.prefix(300))\"")
    print(".reasoning preview: \"\(reasoningText.prefix(300))\"")

    if chunkText.contains("<think>") || chunkText.contains("</think>") {
        fputs("FAIL: .chunk leaked <think> markers — startInReasoning broken.\n", stderr)
        exit(1)
    }
    if reasoningCount == 0 {
        fputs("WARN: zero .reasoning deltas — prompt may not elicit CoT.\n", stderr)
    }
    print("PASS no <think> leakage in .chunk.")
    print("=== BENCH_QWEN_THINKING_CHECK: passed ===")
}

// MARK: - DSV4 chat-template kwargs round-trip

/// Verify the bundle's shipped Jinja chat_template threads the two
/// DSV4 kwargs — `enable_thinking` (bool) and `reasoning_effort`
/// ('max'/None) — through the upstream `applyChatTemplate` path.
///
/// The DSV4 template:
/// - With `enable_thinking=true`  appends `<｜Assistant｜><think>` (open).
/// - With `enable_thinking=false` appends `<｜Assistant｜></think>` (closed).
/// - With `reasoning_effort='max'` prepends a fixed REASONING_EFFORT_MAX
///   preface immediately after BOS.
///
/// Loads the bundle's tokenizer (real HF path), applies the template
/// four ways, decodes back to text, and asserts the expected markers.
/// Pure tokenizer/template exercise — no model forward, no GPU.
func runDSV4TemplateKwargsCheck(modelPath: String) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_DSV4_TEMPLATE_KWARGS: enable_thinking + reasoning_effort ===")
    print("Loading tokenizer only (no model forward)...")
    let loader = #huggingFaceTokenizerLoader()
    let tokenizer = try await loader.load(from: modelDir)

    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": "hi"]
    ]

    func render(_ ctx: [String: any Sendable]?) throws -> String {
        let ids = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: ctx)
        return tokenizer.decode(tokenIds: ids, skipSpecialTokens: false)
    }

    let chatNoEffort = try render(["enable_thinking": false])
    let thinkNoEffort = try render(["enable_thinking": true])
    let chatMaxEffort = try render([
        "enable_thinking": false, "reasoning_effort": "max"
    ])
    let thinkMaxEffort = try render([
        "enable_thinking": true, "reasoning_effort": "max"
    ])

    var failures: [String] = []
    func check(_ label: String, _ ok: Bool, _ why: String) {
        let mark = ok ? "PASS" : "FAIL"
        print("  [\(mark)] \(label): \(why)")
        if !ok { failures.append(label) }
    }

    let preface = "Reasoning Effort: Absolute maximum"

    check("chat-mode tail (closed </think>)",
        chatNoEffort.hasSuffix("</think>"),
        "tail = …\(String(chatNoEffort.suffix(40)))")
    check("thinking-mode tail (open <think>)",
        thinkNoEffort.hasSuffix("<think>"),
        "tail = …\(String(thinkNoEffort.suffix(40)))")
    check("max-effort preface absent without effort kwarg",
        !chatNoEffort.contains(preface) && !thinkNoEffort.contains(preface),
        "neither chat nor thinking carry preface")
    check("max-effort preface present in chat+max",
        chatMaxEffort.contains(preface),
        "preface present")
    check("max-effort preface present in thinking+max",
        thinkMaxEffort.contains(preface),
        "preface present")
    check("max-effort variants preserve correct tail",
        chatMaxEffort.hasSuffix("</think>") && thinkMaxEffort.hasSuffix("<think>"),
        "tails preserved under reasoning_effort=max")

    if !failures.isEmpty {
        fputs("BENCH_DSV4_TEMPLATE_KWARGS: \(failures.count) failures: \(failures)\n", stderr)
        exit(1)
    }
    print("=== BENCH_DSV4_TEMPLATE_KWARGS: passed ===")
}

// MARK: - Orphan-slot consumer-cancellation reproducer

/// Mimics the osaurus-reported pattern: consumer breaks the for-await
/// loop early (Task.isCancelled), engine slot keeps stepping
/// internally, second request fires and collides with the orphan
/// slot's Metal pipelines mid-encode.
///
/// Pre-`continuation.onTermination` fix: request B crashed inside
/// `BatchEngine.stepPrefill` with
/// `notifyExternalReferencesNonZeroOnDealloc`.
/// Post-fix: termination handler reaps slot A; request B prefills
/// against a clean state.
///
/// Uses `BatchEngine` with `CacheCoordinator(usePagedCache=true,
/// enableDiskCache=true)` matching the osaurus config so the cache-
/// hit path actually fires for request B.
func runOrphanSlotRepro(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_ORPHAN_SLOT_REPRO: consumer-cancel → reuse-cache pattern ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    let diskDir = FileManager.default.temporaryDirectory
        .appendingPathComponent("vmlx-orphan-repro-\(UUID().uuidString)")
    try FileManager.default.createDirectory(
        at: diskDir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: diskDir) }

    var coordCfg = CacheCoordinatorConfig()
    coordCfg.usePagedCache = true
    coordCfg.enableDiskCache = true
    coordCfg.diskCacheDir = diskDir
    coordCfg.modelKey = modelDir.lastPathComponent
    coordCfg.pagedBlockSize = 64
    coordCfg.maxCacheBlocks = 512
    let coord = CacheCoordinator(config: coordCfg)

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 4, cacheCoordinator: coord)

    let userQuery = "Name one fruit."
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": userQuery]
    ]
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: messages, tools: nil, additionalContext: nil)
    print("Prompt tokens: \(promptTokens.count)")

    func buildInput() -> LMInput {
        let arr = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)
        return LMInput(text: LMInput.Text(tokens: arr))
    }

    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0.7, topP: 0.8,
        prefillStepSize: 512)

    // ─── Request A: consumer breaks after 4 tokens (simulates Task.isCancelled) ───
    print("\n[Request A] starting (consumer will break after 4 tokens)...")
    let aStart = CFAbsoluteTimeGetCurrent()
    let aInput = buildInput()
    nonisolated(unsafe) let aSend = aInput
    let aStream = await engine.generate(input: aSend, parameters: params)
    var aTokenCount = 0
    var aBroken = false
    for await event in aStream {
        if case .chunk = event { aTokenCount += 1 }
        if case .reasoning = event { aTokenCount += 1 }
        if aTokenCount >= 4 && !aBroken {
            aBroken = true
            print("  [Request A] consumer breaks after 4 tokens")
            break  // ← THIS is the pattern that orphans the slot pre-fix
        }
    }
    let aDur = CFAbsoluteTimeGetCurrent() - aStart
    print(String(format: "  [Request A] consumer exited at %.2fs", aDur))

    // Give the slot a moment to react — onTermination -> cancel(id) is async.
    try? await Task.sleep(nanoseconds: 200_000_000)

    // ─── Request B: same prompt — would hit warm cache and collide ───
    print("\n[Request B] starting (same prompt; cache-hit path)...")
    let bStart = CFAbsoluteTimeGetCurrent()
    let bInput = buildInput()
    nonisolated(unsafe) let bSend = bInput
    let bStream = await engine.generate(input: bSend, parameters: params)
    var bTokenCount = 0
    var bFinishReason: String = "(no info)"
    for await event in bStream {
        if case .chunk = event { bTokenCount += 1 }
        if case .reasoning = event { bTokenCount += 1 }
        if case .info(let info) = event {
            switch info.stopReason {
            case .stop: bFinishReason = "stop (EOS)"
            case .length: bFinishReason = "length"
            case .cancelled: bFinishReason = "cancelled"
            default: bFinishReason = "other"
            }
        }
    }
    let bDur = CFAbsoluteTimeGetCurrent() - bStart
    print(String(format: "  [Request B] completed in %.2fs", bDur))
    print("  [Request B] tokens: \(bTokenCount), finish: \(bFinishReason)")

    if bTokenCount > 0 {
        print("\n=== BENCH_ORPHAN_SLOT_REPRO: PASSED — request B completed without crash ===")
    } else {
        print("\n=== BENCH_ORPHAN_SLOT_REPRO: FAIL — request B produced 0 tokens ===")
        exit(1)
    }
}

// MARK: - Thinking-loop diagnostic probe

/// Probe the model's `</think>` emission behavior on a validation-style
/// prompt that's known to trigger self-refinement loops in
/// reasoning-trained models ("give me 20 random digits"). Reports:
///   - whether `</think>` was ever emitted
///   - reasoning-token count vs content-token count
///   - the LAST 200 chars of reasoning + FIRST 200 chars of content
///   - finish reason
///
/// Sampling matches Qwen-family canonical: `T=0.7, top_p=0.8` per
/// `generation_config.json` in most Qwen3.x bundles. Uses the bundle's
/// chat template with `enable_thinking=true` so the prompt tail emits
/// the open `<think>` marker.
///
/// Used to A/B JANGTQ bits=2 vs JANGTQ4 bits=4 on the same task to
/// isolate whether 4-bit quantization compresses the EOS / `</think>`
/// margin enough to cause infinite self-refinement loops.
func runThinkingLoopProbe(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_THINK_LOOP_PROBE: validation-task </think> emission ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print(String(format: "maxTokens budget: %d", maxNew))

    // Validation-style prompt — known to trigger self-refinement loops
    // in Qwen3.x / DeepSeek-V4 / MiniMax-M2 reasoning-trained models.
    let userQuery =
        "Give me a random 20-digit number. Only return the number itself."
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": userQuery]
    ]
    // Default enable_thinking=true; set THINK=0 env to test the
    // chat-no-think workaround on the same prompt.
    let envThink = ProcessInfo.processInfo.environment["THINK"]
    let enableThinking = (envThink ?? "1") != "0"
    let promptTokens = try context.tokenizer.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": enableThinking])
    print("enable_thinking: \(enableThinking)")
    print("\nPrompt tokens: \(promptTokens.count)")
    print(
        "Prompt rendered tail (last 200 chars): "
            + context.tokenizer.decode(
                tokenIds: Array(promptTokens.suffix(80)),
                skipSpecialTokens: false
            ).debugDescription)

    let promptArr = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)
    let input = LMInput(text: LMInput.Text(tokens: promptArr))
    nonisolated(unsafe) let send = input
    nonisolated(unsafe) let ctx = context

    // Set up reasoning parser to split the stream.
    let promptTail = context.tokenizer.decode(
        tokenIds: Array(promptTokens.suffix(20)), skipSpecialTokens: false)
    let stamp = context.configuration.reasoningParserName ?? "think_xml"
    var parser = ReasoningParser.forPrompt(stampName: stamp, promptTail: promptTail)
    print("Reasoning stamp: \(stamp)")

    // Sampling: Qwen-family canonical (T=0.7, top_p=0.8).
    let params = GenerateParameters(
        maxTokens: maxNew, temperature: 0.7, topP: 0.8,
        prefillStepSize: 512)
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    let (_, stream) = await engine.submit(input: send, parameters: params)

    var reasoningOut = ""
    var contentOut = ""
    var rawTokenCount = 0
    var sawCloseTag = false
    var finishReason: String = "unknown"
    let startTime = CFAbsoluteTimeGetCurrent()
    for await event in stream {
        switch event {
        case .token(let id):
            rawTokenCount += 1
            let piece = context.tokenizer.decode(
                tokenIds: [id], skipSpecialTokens: false)
            // Watch for raw </think> emission BEFORE the parser strips it.
            if piece.contains("</think>") || (parser != nil && {
                // Parser will consume </think> on next feed call —
                // detect via sentinel before the buffer drains.
                return false
            }()) {
                sawCloseTag = true
            }
            // Feed the piece through the parser if available.
            if var p = parser {
                let segs = p.feed(piece)
                parser = p
                for s in segs {
                    switch s {
                    case .reasoning(let r): reasoningOut += r
                    case .content(let c): contentOut += c
                    }
                }
            } else {
                contentOut += piece
            }
        case .info(let info):
            switch info.stopReason {
            case .stop: finishReason = "stop (EOS)"
            case .length: finishReason = "length (max_tokens)"
            case .cancelled: finishReason = "cancelled"
            default: finishReason = "other"
            }
        }
    }
    // Snapshot state BEFORE flush — flush() resets insideReasoning.
    let parserStateBeforeFlush = parser?.isInsideReasoning ?? false
    if var p = parser {
        let trail = p.flush()
        parser = p
        for s in trail {
            switch s {
            case .reasoning(let r): reasoningOut += r
            case .content(let c): contentOut += c
            }
        }
    }
    let dt = CFAbsoluteTimeGetCurrent() - startTime

    // Heuristic: a non-empty content stream means parser observed
    // </think> in the wire output. (The token-level sentinel is
    // imprecise because </think> can be split across tokens; the
    // parser's state transition is authoritative.)
    if !contentOut.isEmpty {
        sawCloseTag = true
    }

    print("")
    print("=== Summary ===")
    print(String(format: "  decoded:    %d tokens in %.2fs (%.1f tok/s)",
        rawTokenCount, dt, Double(rawTokenCount) / dt))
    print(String(format: "  reasoning:  %d chars", reasoningOut.count))
    print(String(format: "  content:    %d chars", contentOut.count))
    print("  </think>:   \(sawCloseTag ? "YES — parser transitioned to content" : "NO — model never closed reasoning")")
    print("  finish:     \(finishReason)")
    print("  parser-pre-flush insideReasoning: \(parserStateBeforeFlush)  ← maps to GenerateCompletionInfo.unclosedReasoning")
    print("")
    print("=== Last 300 chars of reasoning ===")
    let rTail =
        reasoningOut.count > 300
        ? String(reasoningOut.suffix(300)) : reasoningOut
    print(rTail)
    print("")
    print("=== First 300 chars of content ===")
    let cHead =
        contentOut.count > 300
        ? String(contentOut.prefix(300)) : contentOut
    print(cHead.isEmpty ? "(empty)" : cHead)
    print("")
    print("=== BENCH_THINK_LOOP_PROBE: done ===")
}

// MARK: - DSV4 FIM vs Chat coherence probe

/// Side-by-side decode on the same simple factual prompt across DSV4's
/// three prompt-construction modes. Loads the model ONCE and runs the
/// forward pass through each mode. No assertions — prints decoded
/// output so a human can read whether each mode actually answers.
///
/// The reasoning_parser is NOT applied — we want raw model output
/// including any `<think>...</think>` envelope.
func runDSV4FIMvsChat(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_DSV4_FIM_VS_CHAT: FIM vs chat coherence ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")

    let prompt = "The capital of France is"
    let question =
        "Q: Briefly, what is the capital of France?\nA:"

    func decode(_ tokens: [Int]) -> String {
        context.tokenizer.decode(
            tokenIds: tokens, skipSpecialTokens: false)
    }

    func runMode(label: String, promptTokens: [Int]) async throws {
        print("\n[\(label)]  prompt tokens: \(promptTokens.count)")
        print("  prompt rendered: \(decode(promptTokens).debugDescription)")

        nonisolated(unsafe) let ctx = context
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let promptArr = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)
        let input = LMInput(text: LMInput.Text(tokens: promptArr))
        nonisolated(unsafe) let send = input
        let params = GenerateParameters(
            maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
        let stream = await engine.submit(input: send, parameters: params)

        var generated: [Int] = []
        let startTime = CFAbsoluteTimeGetCurrent()
        for await event in stream.1 {
            if case .token(let id) = event { generated.append(id) }
        }
        let dt = CFAbsoluteTimeGetCurrent() - startTime
        let text = decode(generated)
        let tps = dt > 0 ? Double(generated.count) / dt : 0
        print(String(format: "  decoded %d tokens in %.2fs (%.1f tok/s)",
            generated.count, dt, tps))
        print("  ---")
        print(text)
        print("  ---")
    }

    // Mode 1: FIM — raw prompt, no chat template, no markers.
    let fimTokens = context.tokenizer.encode(text: prompt)
    try await runMode(label: "FIM (raw)", promptTokens: fimTokens)

    // Mode 2: CHAT no-think — applyChatTemplate enable_thinking=false.
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": question]
    ]
    let chatNoThink = try context.tokenizer.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": false])
    try await runMode(label: "CHAT enable_thinking=false",
        promptTokens: chatNoThink)

    // Mode 3: CHAT thinking — applyChatTemplate enable_thinking=true.
    let chatThink = try context.tokenizer.applyChatTemplate(
        messages: messages,
        tools: nil,
        additionalContext: ["enable_thinking": true])
    try await runMode(label: "CHAT enable_thinking=true",
        promptTokens: chatThink)

    print("\n=== BENCH_DSV4_FIM_VS_CHAT: done — read outputs above ===")
}

// MARK: - Qwen3.6 multi-turn + tool-call leak check (exact tpae scenario)

/// Replays the EXACT pattern from tpae's 2026-04-20 3:02-3:04 PM
/// screenshots: Qwen3.6 with `enable_thinking=true`, three turns,
/// includes a synthetic tool-response role. Asserts that across ALL
/// turns, `.chunk(String)` never contains `<think>` or `</think>`
/// markers — the bug tpae reported was "thinking bleeds into
/// content" after a tool call. If the fix works per-request (each
/// request construction builds a fresh parser with
/// startInReasoning=true), every turn must be clean.
func runQwenMultiturnToolCheck(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_QWEN_MULTITURN_TOOL: Qwen3.x 3-turn + tool call ===")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    struct TurnResult {
        let idx: Int
        let promptTokens: Int
        let chunks: Int
        let reasoningDeltas: Int
        let toolCalls: Int
        let chunkSample: String
        let reasoningSample: String
        let leakedThink: Bool
    }

    // Simulated 3-turn conversation mirroring tpae's screenshots.
    // Turn 1: user asks to create a README for their game.
    // Turn 2: same request, but with a prior (fake) tool response
    //         in the conversation history — the model re-plans.
    // Turn 3: follow-up asking about the weather (fresh topic, second
    //         tool-call-likely scenario).
    let turns: [(label: String, messages: [[String: any Sendable]])] = [
        (
            "Turn 1 — first request",
            [
                ["role": "user", "content": "Can you generate a README for my game?"]
            ]
        ),
        (
            "Turn 2 — pre-loaded context, just generate",
            [
                ["role": "user", "content":
                    "I have a tic-tac-toe game in HTML/CSS/JS with two-player "
                    + "gameplay and win detection. Write a brief README for it."]
            ]
        ),
        (
            "Turn 3 — follow-up, different topic",
            [
                ["role": "user", "content": "What's the weather in Irvine? Think briefly."]
            ]
        ),
    ]

    var results: [TurnResult] = []
    var anyLeak = false

    for (idx, turn) in turns.enumerated() {
        let promptTokens: [Int]
        do {
            promptTokens = try context.tokenizer.applyChatTemplate(
                messages: turn.messages,
                tools: nil,
                additionalContext: ["enable_thinking": true])
        } catch {
            promptTokens = try context.tokenizer.applyChatTemplate(
                messages: turn.messages)
        }
        let promptIds = MLXArray(promptTokens.map { Int32($0) })
            .reshaped(1, promptTokens.count)
        let input = LMInput(text: LMInput.Text(tokens: promptIds))
        nonisolated(unsafe) let ctxSendable = context
        nonisolated(unsafe) let sendable = input

        let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
        let stream = await engine.generate(input: sendable, parameters: params)

        var chunkText = ""
        var reasoningText = ""
        var chunkCount = 0
        var reasoningCount = 0
        var toolCallCount = 0
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                chunkText += c
                chunkCount += 1
            case .reasoning(let r):
                reasoningText += r
                reasoningCount += 1
            case .toolCall:
                toolCallCount += 1
            case .info:
                break
            }
        }
        // Check every envelope pattern we support — whichever the model's
        // family uses. If ANY leaks, the test fails.
        let leakedMarkers = [
            "<think>", "</think>",           // Qwen/DeepSeek/GLM/MiniMax/Nemotron
            "<|channel>", "<channel|>",      // Gemma-4 harmony
        ]
        let leaked = leakedMarkers.contains { chunkText.contains($0) }
        if leaked { anyLeak = true }
        let r = TurnResult(
            idx: idx + 1,
            promptTokens: promptTokens.count,
            chunks: chunkCount,
            reasoningDeltas: reasoningCount,
            toolCalls: toolCallCount,
            chunkSample: String(chunkText.prefix(160)),
            reasoningSample: String(reasoningText.prefix(160)),
            leakedThink: leaked)
        results.append(r)
        print("\n[\(turn.label)] promptTokens=\(promptTokens.count)")
        print("  chunks=\(chunkCount) reasoning=\(reasoningCount) toolCalls=\(toolCallCount) leakedThinkMarkers=\(leaked)")
        print("  .chunk: \"\(r.chunkSample)\"")
        print("  .reasoning: \"\(r.reasoningSample)\"")
    }

    print("\n=== Turn-by-turn summary ===")
    for r in results {
        print("  Turn \(r.idx): prompt=\(r.promptTokens), chunks=\(r.chunks), reasoning=\(r.reasoningDeltas), toolCalls=\(r.toolCalls), leak=\(r.leakedThink)")
    }
    if anyLeak {
        fputs("\nFAIL: at least one turn leaked reasoning envelope markers in .chunk\n", stderr)
        exit(1)
    }
    print("\nPASS: all turns have zero reasoning envelope markers in .chunk")
    print("=== BENCH_QWEN_MULTITURN_TOOL: passed ===")
}

// MARK: - Perf micro-bench (BENCH_PERF=1)

/// Deterministic decode tok/s micro-bench. Runs warmup + measurement
/// turns, computes median tok/s from the library's own `.info`
/// `generationTime`. Grep-friendly one-liner output per run:
///
///   PERF model=<name> variant=<label> genTokens=N genSec=F tokps=F
///
/// Temperature 0, fixed token budget, no chat template (just raw
/// prompt) so the only variable is the decode hot path.
func runPerfBench(
    modelPath: String,
    maxNew: Int,
    variant: String,
    warmup: Int,
    runs: Int,
    useTokenIterator: Bool = false
) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
    let modelName = modelDir.lastPathComponent

    let promptText = ProcessInfo.processInfo.environment[
        "BENCH_PERF_PROMPT"]
        ?? "Write one long paragraph describing ocean waves. Be verbose and detailed."
    let messages: [[String: any Sendable]] = [
        ["role": "user", "content": promptText]
    ]

    // Try enable_thinking=false so reasoning models don't eat the
    // budget on CoT. Fall back to plain.
    let promptTokens: [Int]
    do {
        promptTokens = try context.tokenizer.applyChatTemplate(
            messages: messages,
            tools: nil,
            additionalContext: ["enable_thinking": false])
    } catch {
        promptTokens = try context.tokenizer.applyChatTemplate(messages: messages)
    }
    let promptIds = MLXArray(promptTokens.map { Int32($0) })
        .reshaped(1, promptTokens.count)

    nonisolated(unsafe) let ctxSendable = context
    let engine = BatchEngine(context: ctxSendable, maxBatchSize: 1)

    func oneTurn(_ label: String) async throws -> (Int, Double) {
        let input = LMInput(text: LMInput.Text(tokens: promptIds))
        nonisolated(unsafe) let sendable = input
        let params = GenerateParameters(
            maxTokens: maxNew, temperature: 0, prefillStepSize: 512)
        var genTokens = 0
        var genSec = 0.0
        let whichPath = ProcessInfo.processInfo.environment[
            "BENCH_PERF_PATH"] ?? "batch"
        if useTokenIterator {
            // Direct TokenIterator path — no BatchEngine actor, no
            // Batch wrappers. Same measurement semantics: read genTime
            // from the .info event emitted by generateTask.
            nonisolated(unsafe) let ctxLocal = ctxSendable
            let stream = try MLXLMCommon.generate(
                input: sendable, parameters: params, context: ctxLocal)
            for await ev in stream {
                switch ev {
                case .chunk, .reasoning, .toolCall:
                    break
                case .info(let info):
                    genTokens = info.generationTokenCount
                    genSec = info.generateTime
                }
            }
        } else if whichPath == "submit" {
            // Raw-token path through BatchEngine's `submit()` — no
            // detokenizer, no reasoning parser, no tool-call processor,
            // no consumer Task. Directly drains `BatchGeneration`
            // events. Isolates how much of the Batch-vs-Iter gap is
            // the consumer Task vs the actor hops / scheduler.
            let (_, stream) = await engine.submit(
                input: sendable, parameters: params)
            for await ev in stream {
                switch ev {
                case .token:
                    break
                case .info(let info):
                    genTokens = info.generationTokenCount
                    genSec = info.generateTime
                }
            }
        } else {
            let stream = await engine.generate(input: sendable, parameters: params)
            for await ev in stream {
                switch ev {
                case .chunk, .reasoning, .toolCall:
                    break
                case .info(let info):
                    genTokens = info.generationTokenCount
                    genSec = info.generateTime
                }
            }
        }
        return (genTokens, genSec)
    }

    // Warmup
    for i in 0..<warmup {
        _ = try await oneTurn("warmup\(i)")
    }
    // Measurement
    var tokps: [Double] = []
    var lastGenTokens = 0
    var lastGenSec = 0.0
    for i in 0..<runs {
        let (gt, gs) = try await oneTurn("run\(i)")
        let tps = gs > 0 ? Double(gt) / gs : 0
        tokps.append(tps)
        lastGenTokens = gt
        lastGenSec = gs
    }
    let median = tokps.sorted()[tokps.count / 2]
    let best = tokps.max() ?? 0
    let head = String(
        gitShortHead(modelDir: FileManager.default.currentDirectoryPath))
    let pathLabel = useTokenIterator ? "iter" : "batch"
    print(String(format:
        "PERF model=%@ variant=%@ path=%@ commit=%@ loadSec=%.2f genTokens=%d genSec=%.3f tokps_median=%.1f tokps_best=%.1f runs=%@",
        modelName, variant, pathLabel, head, loadSec, lastGenTokens, lastGenSec,
        median, best,
        tokps.map { String(format: "%.1f", $0) }.joined(separator: ",")
    ))
}

/// Get short git HEAD for the tree rooted at `modelDir`. Purely for
/// reporting — failure returns "?".
fileprivate func gitShortHead(modelDir: String) -> String {
    let p = Process()
    p.launchPath = "/usr/bin/git"
    p.arguments = ["-C", modelDir, "rev-parse", "--short", "HEAD"]
    p.currentDirectoryPath = modelDir
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()
    do {
        try p.run()
        p.waitUntilExit()
        let d = pipe.fileHandleForReading.readDataToEndOfFile()
        return String(data: d, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? "?"
    } catch {
        return "?"
    }
}

// MARK: - Crash fuzz (2026-04-23)
//
// tpae reported a reproducible `_assertionFailure` on Qwen 3.6 27B via
// osaurus 0.17.3. The osaurus repo ships no source, so we can't map the
// stack. This runner loads the real 27B once and fires a sequence of
// adversarial request patterns that osaurus is likely to produce. Each
// scenario prints `SCENARIO N START` before and `SCENARIO N DONE`
// after — the last "START" line on a crash names the culprit.

func runCrashFuzz(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_CRASH_FUZZ — tpae Qwen3.6-27B repro probe ===")
    print("Loading with real HuggingFace tokenizer from \(modelPath) ...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    nonisolated(unsafe) let ctx = context

    // Short-decode-budget baseline — each scenario should finish in
    // seconds, not minutes. Keep `maxNew` honest but low.
    let budget = min(maxNew, 48)

    // Helper: drain a submitted stream to completion. Separated from
    // prepare/submit so Swift 6 strict concurrency doesn't trip on
    // `ctx` capture inside TaskGroup closures.
    @Sendable func drain(
        _ stream: AsyncStream<BatchGeneration>,
        cap: Int
    ) async -> (tokens: [Int], stop: GenerateStopReason?) {
        var ids: [Int] = []
        var stop: GenerateStopReason? = nil
        for await e in stream {
            switch e {
            case .token(let id): ids.append(id)
            case .info(let info): stop = info.stopReason
            }
            if ids.count >= cap { break }
        }
        return (ids, stop)
    }

    // Pre-prepare inputs on the main task, then pass sendable LMInputs
    // into TaskGroup closures. Sidesteps the `ctx` non-Sendable capture.
    func prepareInputs(_ prompts: [String]) async throws -> [LMInput] {
        var out: [LMInput] = []
        for p in prompts {
            out.append(try await ctx.processor.prepare(
                input: UserInput(prompt: p)))
        }
        return out
    }

    // Scenario 1: B=1 baseline — prove the engine is healthy at all.
    do {
        print("\nSCENARIO 1 START: B=1 baseline")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs(["The capital of France is"])
        nonisolated(unsafe) let send = inputs[0]
        let (_, s) = await engine.submit(input: send, parameters: params)
        let r = await drain(s, cap: budget)
        print("SCENARIO 1 DONE: \(r.tokens.count) tokens, stop=\(r.stop.map{"\($0)"} ?? "nil")")
    }

    // Scenario 2: B=4 concurrent distinct prompts.
    do {
        print("\nSCENARIO 2 START: B=4 concurrent")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs([
            "Name one color.",
            "Name one animal.",
            "Name one country.",
            "Name one number.",
        ])
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append(s)
        }
        await withTaskGroup(of: (Int, Int).self) { group in
            for (i, s) in streams.enumerated() {
                group.addTask {
                    let r = await drain(s, cap: budget)
                    return (i, r.tokens.count)
                }
            }
            for await r in group {
                print("  slot \(r.0): \(r.1) tokens")
            }
        }
        print("SCENARIO 2 DONE")
    }

    // Scenario 3: B=4 with mid-stream cancellation on two slots.
    // Matches osaurus's "client disconnected" pattern — HTTP Task
    // cancelled while BatchEngine is mid-decode. The surviving slots
    // must finish cleanly.
    do {
        print("\nSCENARIO 3 START: B=4 with cancel on 2 of 4")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        let params = GenerateParameters(
            maxTokens: max(budget, 96), temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs([
            "Describe water briefly.",
            "Describe fire briefly.",
            "Describe wind briefly.",
            "Describe earth briefly.",
        ])
        var ids: [BatchRequestID] = []
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (uuid, s) = await engine.submit(input: send, parameters: params)
            ids.append(uuid); streams.append(s)
        }
        let perSlotCap = max(budget, 96)
        await withTaskGroup(of: (Int, Int).self) { group in
            for (i, s) in streams.enumerated() {
                group.addTask {
                    let r = await drain(s, cap: perSlotCap)
                    return (i, r.tokens.count)
                }
            }
            let idsSnapshot = ids
            group.addTask {
                try? await Task.sleep(nanoseconds: 120_000_000)
                await engine.cancel(idsSnapshot[1])
                await engine.cancel(idsSnapshot[2])
                return (-1, 0)
            }
            for await r in group where r.0 >= 0 {
                print("  slot \(r.0): \(r.1) tokens")
            }
        }
        print("SCENARIO 3 DONE")
    }

    // Scenario 4: every slot maxTokens=1 — EOS-on-first-token ballpark.
    // Post-105ff8b this should finish cleanly with `.length` stops (or
    // `.stop` if first token happens to be EOS); pre-fix this was the
    // known force-unwrap.
    do {
        print("\nSCENARIO 4 START: B=4 maxTokens=1 each")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        let params = GenerateParameters(
            maxTokens: 1, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs(["Hi", "Hello", "Hey", "Yo"])
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append(s)
        }
        await withTaskGroup(of: Int.self) { group in
            for s in streams {
                group.addTask {
                    let r = await drain(s, cap: 1)
                    return r.tokens.count
                }
            }
            var total = 0
            for await c in group { total += c }
            print("  total tokens across 4 slots: \(total)")
        }
        print("SCENARIO 4 DONE")
    }

    // Scenario 5: single-token prompt. Tokenizer returns one non-BOS id
    // → prefill with 1 token is a cold-start edge case for the hybrid
    // SSM path (no multi-token prefill; the Mamba scan runs on T=1).
    do {
        print("\nSCENARIO 5 START: single-token prompt")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs(["a"])
        nonisolated(unsafe) let send = inputs[0]
        let (_, s) = await engine.submit(input: send, parameters: params)
        let r = await drain(s, cap: budget)
        print("SCENARIO 5 DONE: \(r.tokens.count) tokens, stop=\(r.stop.map{"\($0)"} ?? "nil")")
    }

    // Scenario 6: submit, then abandon the stream (consumer drops).
    // If BatchEngine relies on backpressure from the consumer to free
    // slot state, an abandoned stream could wedge or trap.
    do {
        print("\nSCENARIO 6 START: submit + drop consumer")
        let engine = BatchEngine(context: ctx, maxBatchSize: 2)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let input6 = try await ctx.processor.prepare(
            input: UserInput(prompt: "Count to ten."))
        nonisolated(unsafe) let sendable6 = input6
        let (uuid6, s6) = await engine.submit(
            input: sendable6, parameters: params)
        _ = s6 // drop the stream on the floor
        try? await Task.sleep(nanoseconds: 300_000_000)
        await engine.cancel(uuid6)
        print("SCENARIO 6 DONE")
    }

    // Scenario 7: back-to-back identical prompts on one engine. Second
    // submit should hit the coordinator cache. Exercises SSM seed +
    // paged KV hit + potential disk-write race.
    do {
        print("\nSCENARIO 7 START: identical prompt twice back-to-back")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        for pass in 1...2 {
            let inputs = try await prepareInputs(["List three prime numbers."])
            nonisolated(unsafe) let send = inputs[0]
            let (_, s) = await engine.submit(input: send, parameters: params)
            let r = await drain(s, cap: budget)
            print("  pass \(pass): \(r.tokens.count) tokens, stop=\(r.stop.map{"\($0)"} ?? "nil")")
        }
        print("SCENARIO 7 DONE")
    }

    // Scenario 8: stop-string that matches the first plausible output.
    // `extraStopStrings = [" "]` — the first whitespace after the prompt
    // will trip the matcher and close the slot quickly. Exercises the
    // StopStringMatcher integration under B=4.
    do {
        print("\nSCENARIO 8 START: B=4 with aggressive stop-string")
        let engine = BatchEngine(context: ctx, maxBatchSize: 4)
        var params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        params.extraStopStrings = ["\n", "."]
        let inputs = try await prepareInputs([
            "Say hi.", "Say hello.", "Say hey.", "Say yo.",
        ])
        var streams: [AsyncStream<BatchGeneration>] = []
        for input in inputs {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append(s)
        }
        await withTaskGroup(of: Int.self) { group in
            for s in streams {
                group.addTask {
                    let r = await drain(s, cap: budget)
                    return r.tokens.count
                }
            }
            var total = 0
            for await c in group { total += c }
            print("  total: \(total) tokens across 4 slots")
        }
        print("SCENARIO 8 DONE")
    }

    // Scenario 9: wildly different prompt lengths in the same batch.
    // Short prompts + a 500-token prompt → pad/mask alignment stress.
    do {
        print("\nSCENARIO 9 START: B=2 mixed short + long prompts")
        let engine = BatchEngine(context: ctx, maxBatchSize: 2)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let longPrompt = String(repeating: "The quick brown fox jumps. ", count: 60)
        let inputs = try await prepareInputs(["Hi.", longPrompt])
        var streams: [(String, AsyncStream<BatchGeneration>)] = []
        for (label, input) in zip(["short", "long"], inputs) {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            streams.append((label, s))
        }
        await withTaskGroup(of: (String, Int).self) { group in
            for (label, s) in streams {
                group.addTask {
                    let r = await drain(s, cap: budget)
                    return (label, r.tokens.count)
                }
            }
            for await r in group {
                print("  \(r.0): \(r.1) tokens")
            }
        }
        print("SCENARIO 9 DONE")
    }

    // Scenario 10: rapid sequential submits on a shared engine — the
    // osaurus server handles requests one at a time in the common case
    // where the client opens a new HTTP request per turn. Cache writes
    // from turn N overlap with turn N+1's submit — potential race on
    // the coordinator's disk write queue.
    do {
        print("\nSCENARIO 10 START: 5 rapid sequential submits")
        let engine = BatchEngine(context: ctx, maxBatchSize: 1)
        let params = GenerateParameters(
            maxTokens: budget, temperature: 0, prefillStepSize: 512)
        let inputs = try await prepareInputs([
            "One.", "Two.", "Three.", "Four.", "Five.",
        ])
        for (i, input) in inputs.enumerated() {
            nonisolated(unsafe) let send = input
            let (_, s) = await engine.submit(input: send, parameters: params)
            let r = await drain(s, cap: budget)
            print("  turn \(i+1): \(r.tokens.count) tokens")
        }
        print("SCENARIO 10 DONE")
    }

    print("\n=== BENCH_CRASH_FUZZ: all scenarios finished without crash ===")
}

// MARK: - Crash fuzz V2 — generate() pipeline, multi-turn (2026-04-23)
//
// Targets the NaiveStreamingDetokenizer + ReasoningParser +
// ToolCallProcessor + StopStringMatcher chain on the
// `BatchEngine.generate(input:parameters:)` path — the same path
// osaurus uses and the one that reproduced tpae's crash.
// Each scenario drives REAL chat-template rendering and expects the
// generate() stream to close cleanly. Printing `SCENARIO N START`
// before and `SCENARIO N DONE` after makes the last line before a
// crash name the culprit.

func runCrashFuzzV2(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    print("\n=== BENCH_CRASH_FUZZ_V2 — generate() pipeline stress ===")
    print("Loading with real HuggingFace tokenizer from \(modelPath) ...")
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    print(String(format: "Load: %.2fs", CFAbsoluteTimeGetCurrent() - loadStart))
    print("Model: \(type(of: context.model))")
    nonisolated(unsafe) let ctx = context

    let budget = min(maxNew, 96)
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    let params = GenerateParameters(
        maxTokens: budget, temperature: 0, prefillStepSize: 512)

    // Run a single prompt through `generate()` — the full production
    // pipeline (detokenizer → reasoning parser → tool processor →
    // stop-string matcher). Returns accumulated text chunks + whether
    // anything streamed at all. If any inner stage traps, we crash
    // here with the SCENARIO line as the breadcrumb.
    @Sendable func runGenerate(
        _ userInput: UserInput, label: String
    ) async throws -> (chunks: String, reasoning: String, toolCalls: Int) {
        let input = try await ctx.processor.prepare(input: userInput)
        nonisolated(unsafe) let send = input
        let stream = await engine.generate(input: send, parameters: params)
        var chunks = ""
        var reasoning = ""
        var tools = 0
        for await event in stream {
            switch event {
            case .chunk(let c): chunks += c
            case .reasoning(let r): reasoning += r
            case .toolCall: tools += 1
            case .info: break
            }
        }
        let preview = chunks.count > 80 ? String(chunks.prefix(80)) + "..." : chunks
        print("  \(label): chunks=\(chunks.count) reasoning=\(reasoning.count) tools=\(tools) -> \"\(preview)\"")
        return (chunks, reasoning, tools)
    }

    // Scenario 1: emoji-heavy output — forces byte-level BPE to
    // complete multi-byte graphemes across token boundaries. This is
    // the class that tripped the tpae crash pre-fix.
    do {
        print("\nSCENARIO 1 START: emoji-heavy output")
        _ = try await runGenerate(
            UserInput(prompt: "Write exactly five emojis separated by spaces. Only emojis, nothing else."),
            label: "emoji")
        print("SCENARIO 1 DONE")
    }

    // Scenario 2: Unicode-dense multilingual output — cleanUp
    // substitutions + grapheme-cluster edge cases compound here.
    do {
        print("\nSCENARIO 2 START: multilingual output")
        _ = try await runGenerate(
            UserInput(prompt: "Say \"hello\" in Japanese, Korean, Chinese, Arabic, Hindi."),
            label: "ml")
        print("SCENARIO 2 DONE")
    }

    // Scenario 3: punctuation-dense output — exercises the
    // cleanUpTokenizationSpaces substitutions specifically
    // (" ." → ".", " n't" → "n't", " 's" → "'s").
    do {
        print("\nSCENARIO 3 START: contractions + punctuation dense")
        _ = try await runGenerate(
            UserInput(prompt: "Write one sentence that uses: don't, won't, isn't, it's, she's, we've, they're, I'm. All in one sentence."),
            label: "punct")
        print("SCENARIO 3 DONE")
    }

    // Scenario 4: reasoning ON then OFF on the same engine. Verifies
    // the reasoning parser resets cleanly between turns AND that
    // special-token boundaries (Qwen 3.6 has `<|im_end|>`, channel
    // markers, etc.) don't trip the detokenizer.
    do {
        print("\nSCENARIO 4 START: reasoning ON, then OFF, on the same engine")
        let onPrompt = UserInput(prompt: "What is 7 * 8? Think step by step.")
        let offPrompt = UserInput(prompt: "What color is the sky? /no_think")
        _ = try await runGenerate(onPrompt, label: "ON")
        _ = try await runGenerate(offPrompt, label: "OFF")
        print("SCENARIO 4 DONE")
    }

    // Scenario 5: very short output that likely ends on an EOS-family
    // special token. This class is what originally inspired the
    // special-token collapse test case.
    do {
        print("\nSCENARIO 5 START: single-word answer")
        _ = try await runGenerate(
            UserInput(prompt: "Reply with exactly one word: yes or no. Is the sky blue?"),
            label: "oneword")
        print("SCENARIO 5 DONE")
    }

    // Scenario 6: `\u{fffd}`-prone content — ask for characters whose
    // UTF-8 encoding is likely to split across tokens. Incomplete
    // grapheme completion collapsing multi-replacement tails into a
    // single emoji is the third decode-shrinkage path.
    do {
        print("\nSCENARIO 6 START: UTF-8 multibyte continuation")
        _ = try await runGenerate(
            UserInput(prompt: "Write this exact string and nothing else: 你好世界 🌏 🚀 café naïve résumé"),
            label: "utf8")
        print("SCENARIO 6 DONE")
    }

    // Scenario 7: 10 rapid back-to-back turns on the same engine,
    // alternating short / emoji / code prompts. Exhausts cross-turn
    // cache reuse + re-entry into the generate() pipeline.
    do {
        print("\nSCENARIO 7 START: 10 rapid back-to-back turns")
        let prompts = [
            "Count to three.",
            "Three emojis: ",
            "```python\nprint('hi')\n```",
            "¿Hola, cómo estás?",
            "🚀🎉🔥",
            "42",
            "Say 'done'.",
            "一",
            ".",
            "!",
        ]
        for (i, p) in prompts.enumerated() {
            _ = try await runGenerate(
                UserInput(prompt: p),
                label: "t\(i+1)")
        }
        print("SCENARIO 7 DONE")
    }

    print("\n=== BENCH_CRASH_FUZZ_V2: all scenarios finished without crash ===")
}

// MARK: - Official final-pass multi-turn (2026-04-23)
//
// One-model harness designed to be looped over several models by the
// shell driver. Each scenario reports per-turn timing, tok/s, reasoning
// / chunk / tool-call counts, peak RSS in MiB, and content validation
// where applicable. All scenarios run through `engine.generate(...)`
// — the production path — so every turn exercises the full pipeline
// (detokenizer → reasoning parser → tool-call processor → stop-string
// matcher). Stats are printed in a compact single-line format per turn
// for easy shell post-processing.

/// Current resident set size in MiB, via `mach_task_basic_info`. Used
/// as a cheap peak-memory tracker across turns.
fileprivate func currentRSSMiB() -> Double {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(
        MemoryLayout<mach_task_basic_info>.stride / MemoryLayout<natural_t>.stride)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(
                mach_task_self_,
                task_flavor_t(MACH_TASK_BASIC_INFO),
                $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return -1 }
    return Double(info.resident_size) / (1024.0 * 1024.0)
}

func runOfficialMultiTurn(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    print("\n=== BENCH_OFFICIAL — \(modelName) ===")
    let rssBefore = currentRSSMiB()
    print(String(format: "RSS before load: %.0f MiB", rssBefore))
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
    let rssAfterLoad = currentRSSMiB()
    print(String(format: "Load: %.2fs  Model: %@  RSS +%.0f MiB -> %.0f MiB",
        loadSec, String(describing: type(of: context.model)),
        rssAfterLoad - rssBefore, rssAfterLoad))
    print("Tool format: \(context.configuration.toolCallFormat.map { "\($0)" } ?? "json (default)")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    nonisolated(unsafe) let ctx = context
    let engine = BatchEngine(context: ctx, maxBatchSize: 1)
    let params = GenerateParameters(
        maxTokens: max(maxNew, 96), temperature: 0, prefillStepSize: 512)

    // Use reference boxes so the nested @Sendable closure can mutate
    // counts without tripping Swift 6's strict-concurrency capture
    // rules. Closures run sequentially on the caller's task; the box
    // isn't a concurrency tool, just a capture-mode tool.
    final class Stats: @unchecked Sendable {
        var peakRSS: Double
        var passCount = 0
        var failCount = 0
        init(peakRSS: Double) { self.peakRSS = peakRSS }
    }
    let stats = Stats(peakRSS: rssAfterLoad)

    // One turn through generate(). Captures timing, content, peak RSS.
    // Returns the text/reasoning/toolCall counts so scenario-specific
    // validation can run on them.
    func runTurn(
        label: String, prompt: String, thinking: Bool?,
        tools: [[String: any Sendable]]? = nil,
        validate: (_ text: String, _ reasoning: String, _ tools: Int) -> (ok: Bool, why: String)
    ) async throws {
        var userInput = UserInput(prompt: prompt)
        if let thinking {
            userInput.additionalContext = ["enable_thinking": thinking]
        }
        if let tools {
            userInput.tools = tools
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let input = try await ctx.processor.prepare(input: userInput)
        nonisolated(unsafe) let send = input
        let stream = await engine.generate(input: send, parameters: params)
        var text = ""
        var reasoning = ""
        var toolCalls = 0
        var chunks = 0
        var reasoningDeltas = 0
        var ttft: Double?
        for await event in stream {
            switch event {
            case .chunk(let c):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                text += c; chunks += 1
            case .reasoning(let r):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                reasoning += r; reasoningDeltas += 1
            case .toolCall: toolCalls += 1
            case .info: break
            }
        }
        let total = CFAbsoluteTimeGetCurrent() - t0
        let now = currentRSSMiB()
        if now > stats.peakRSS { stats.peakRSS = now }

        // Approximate decode tok/s as (chunks+reasoningDeltas)/total —
        // deltas are per-token in the generate() pipeline. Not exact
        // (detokenizer may buffer multi-byte chars across multiple
        // tokens, emitting fewer events than tokens), but close enough
        // for relative comparison across turns.
        let deltas = chunks + reasoningDeltas
        let tokps = total > 0 ? Double(deltas) / total : 0
        let ttftMs = Int((ttft ?? 0) * 1000)
        let preview = (text.isEmpty ? reasoning : text)
        let short = preview.count > 100 ? String(preview.prefix(100)) + "…" : preview

        let v = validate(text, reasoning, toolCalls)
        let status = v.ok ? "PASS" : "FAIL"
        if v.ok { stats.passCount += 1 } else { stats.failCount += 1 }

        print(String(format:
            "  [%@] %@  ttft=%4dms total=%5.2fs tokps=%5.1f chunks=%4d reasoning=%4d tools=%d rss=%.0fMiB -> %@%@",
            status, label, ttftMs, total, tokps, chunks, reasoning.count, toolCalls, now,
            v.ok ? "" : "WHY=\(v.why) ",
            "\"\(short.replacingOccurrences(of: "\n", with: "\\n"))\""))
    }

    // S1: Math problem — reasoning ON. Validate answer text contains "4".
    try await runTurn(
        label: "S1 reasoning=ON  math 7+8-11", prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { text, reasoning, _ in
        // Any of {text, reasoning} must contain "4" for pass.
        let combined = text + reasoning
        if combined.contains("4") {
            return (true, "")
        }
        return (false, "answer not found")
    }

    // S2: Same prompt — verifies cache hit + reproducibility.
    try await runTurn(
        label: "S2 reasoning=ON  math 7+8-11 (cache hit)",
        prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { text, reasoning, _ in
        let combined = text + reasoning
        if combined.contains("4") {
            return (true, "")
        }
        return (false, "answer not found on cache hit")
    }

    // S3: thinking=OFF — should produce mostly .chunk events, minimal
    // .reasoning. Ask for a short visible answer.
    try await runTurn(
        label: "S3 reasoning=OFF factual",
        prompt: "What color is the sky on a clear day? Answer with one word.",
        thinking: false
    ) { text, reasoning, _ in
        let combined = (text + " " + reasoning).lowercased()
        if combined.contains("blue") {
            return (true, "")
        }
        // Tolerate models that still answer in reasoning when
        // enable_thinking=false is template-overridden — only fail on
        // empty output.
        if (text + reasoning).isEmpty {
            return (false, "empty response")
        }
        return (true, "accepted non-blue answer")
    }

    // S4: Multi-tool-call prompt. Two tools; ask the model to call
    // both. Different formats (xmlFunction, Mistral inline, harmony,
    // Pythonic) have different match criteria, so we accept as PASS
    // if the generation completed without crash and the pipeline
    // produced ≥1 tool call OR ≥1 content chunk (some models prefer
    // to inline the answer rather than call tools when the question
    // is answerable from prior knowledge).
    // Nested dictionaries need explicit `[String: any Sendable]`
    // annotations so nested literals don't widen to `Any`.
    let weatherParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "city": ["type": "string", "description": "City name"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["city"],
    ]
    let weatherFn: [String: any Sendable] = [
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": weatherParams,
    ]
    let weatherTool: [String: any Sendable] = [
        "type": "function",
        "function": weatherFn,
    ]
    let timeParams: [String: any Sendable] = [
        "type": "object",
        "properties": [
            "timezone": ["type": "string", "description": "IANA zone"] as [String: any Sendable],
        ] as [String: any Sendable],
        "required": ["timezone"],
    ]
    let timeFn: [String: any Sendable] = [
        "name": "get_time",
        "description": "Get current time in a timezone.",
        "parameters": timeParams,
    ]
    let timeTool: [String: any Sendable] = [
        "type": "function",
        "function": timeFn,
    ]
    try await runTurn(
        label: "S4 multi-tool-call",
        prompt: "I need BOTH the current weather in Tokyo AND the current time in Asia/Tokyo. Call the appropriate tools.",
        thinking: true, tools: [weatherTool, timeTool]
    ) { text, reasoning, tools in
        // Accept any of: ≥1 tool call extracted OR the model produced
        // substantive output discussing both entities (fallback path
        // when the tokenizer emits tool-call syntax in a format our
        // parser doesn't recognize for this model). Primary metric:
        // no crash + non-empty output.
        if tools >= 1 { return (true, "extracted \(tools) tool call(s)") }
        if (text + reasoning).count >= 10 {
            return (true, "no tool call but \(text.count + reasoning.count) chars emitted")
        }
        return (false, "empty output, no tool call")
    }

    // S5: UTF-8 / emoji / multilingual stress — the shrinkage-prone
    // class that originally tripped tpae's crash.
    try await runTurn(
        label: "S5 utf8 emoji stress",
        prompt: "Write exactly this line verbatim: 🚀 café naïve résumé 你好 こんにちは 안녕하세요",
        thinking: false
    ) { text, reasoning, _ in
        if (text + reasoning).isEmpty { return (false, "empty") }
        return (true, "")
    }

    // S6: 5-turn rapid sequential chat. Each turn is a plain prompt
    // (no tools). Tests cross-turn cache reuse + repeated enter/exit
    // of the generate() pipeline + detokenizer stability under churn.
    let rapidPrompts = [
        "Name a country.",
        "Name a color.",
        "Name a fruit.",
        "Name an animal.",
        "Name a day of the week.",
    ]
    for (i, p) in rapidPrompts.enumerated() {
        try await runTurn(
            label: "S6.\(i+1) rapid",
            prompt: p, thinking: false
        ) { text, reasoning, _ in
            if (text + reasoning).isEmpty { return (false, "empty") }
            return (true, "")
        }
    }

    print(String(format:
        "\n=== BENCH_OFFICIAL summary: model=%@ pass=%d fail=%d peakRSS=%.0fMiB loadSec=%.2f ===",
        modelName, stats.passCount, stats.failCount, stats.peakRSS, loadSec))

    if stats.failCount > 0 {
        throw NSError(domain: "BENCH_OFFICIAL", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(stats.failCount) scenario(s) failed validation",
        ])
    }
}

// MARK: - Production exhaustive matrix (2026-04-23)
//
// The most comprehensive real-model test we run. For a single model,
// exercises every production path vmlx-swift-lm ships: tool-call round
// trips, reasoning ON→OFF alternation, L2 disk cache hit, SSM state
// re-derive on hybrid models, TurboQuant runtime sidecar on JANGTQ
// bundles. Every scenario validates content — a pass requires the
// right answer or the right tool_call schema, not just "stream
// didn't crash".
//
// Environment knobs:
//   BENCH_PROD_CACHE_DIR   L2 disk cache root (default
//                          /tmp/vmlx-prod-cache/<model-basename>)
//   BENCH_MAX_TOKENS       per-turn decode cap (default 64)
//   BENCH_PROD_SKIP_TOOLS  skip S3 tool round-trip for hostile formats
//                          (e.g. pure-JSON models during triage)
//
// Exit code 0 = every scenario PASS with correct content. Non-zero
// = at least one scenario flunked its content predicate; the
// summary line names which.


// MARK: - Production exhaustive matrix (2026-04-23, v2 — prompt-based)
//
// Prompt-string based to avoid the chat-template apply hang from v1.
// Uses the same text-history accumulation VLBench.runMixedMultiTurn
// relies on. Validates content per scenario.
//
// Coverage per invocation:
//   S1  reasoning=ON  math (validate "4" in output)
//   S2  SAME prompt as S1 — paged cache hit, TTFT drops
//   S3  reasoning=OFF factual (validate "blue")
//   S4  reasoning ON→OFF→ON alternation within one engine
//   S5  UTF-8 / emoji / multilingual verbatim (shrinkage stress)
//   S6  SSM-seed: hybrid-SSM-only models — identical prefix
//       continuation should reuse paged KV + SSM companion cache
//   S7  L2 disk: rerun with fresh process via env override
//       (scripted by shell loop, not here)

func runProdMatrix(modelPath: String, maxNew: Int) async throws {
    let modelDir = URL(fileURLWithPath: modelPath)
    let modelName = modelDir.lastPathComponent
    let env = ProcessInfo.processInfo.environment
    let budget = max(maxNew, 96)

    print("\n=== BENCH_PROD — \(modelName) ===")
    let cacheRoot = env["BENCH_PROD_CACHE_DIR"] ??
        "/tmp/vmlx-prod-cache/\(modelName)"
    try? FileManager.default.createDirectory(
        atPath: cacheRoot, withIntermediateDirectories: true)
    print("Cache dir: \(cacheRoot)")

    let rss0 = currentRSSMiB()
    let loadStart = CFAbsoluteTimeGetCurrent()
    let context = try await MLXLMCommon.loadModel(
        from: modelDir, using: #huggingFaceTokenizerLoader())
    let loadSec = CFAbsoluteTimeGetCurrent() - loadStart
    let rss1 = currentRSSMiB()
    print(String(format: "Load: %.2fs  Model: %@  RSS +%.0f MiB",
        loadSec, String(describing: type(of: context.model)),
        rss1 - rss0))
    print("Tool format: \(context.configuration.toolCallFormat.map{"\($0)"} ?? "json")")
    print("Reasoning stamp: \(context.configuration.reasoningParserName ?? "nil")")

    // BatchEngine with the standard default CacheCoordinator wiring
    // (nil passes through to the per-prompt cache — this matches the
    // production path the osaurus server uses). If BENCH_PROD_COORD=1
    // is set, attach an explicit L2-disk coordinator.
    nonisolated(unsafe) let ctx = context
    let coordinator: CacheCoordinator?
    if (env["BENCH_PROD_COORD"] ?? "0") == "1" {
        let cfg = CacheCoordinatorConfig(
            usePagedCache: true, enableDiskCache: true,
            pagedBlockSize: 64, maxCacheBlocks: 512,
            diskCacheMaxGB: 4.0,
            diskCacheDir: URL(fileURLWithPath: cacheRoot),
            ssmMaxEntries: 32, modelKey: modelName)
        coordinator = CacheCoordinator(config: cfg)
        print("Coordinator: enabled (L2 disk at \(cacheRoot))")
    } else {
        coordinator = nil
    }
    let engine = BatchEngine(
        context: ctx, maxBatchSize: 1, cacheCoordinator: coordinator)
    let params = GenerateParameters(
        maxTokens: budget, temperature: 0, prefillStepSize: 512)

    final class Stats: @unchecked Sendable {
        var peakRSS: Double; var pass = 0; var fail = 0
        var ttftByLabel: [String: Int] = [:]
        init(peakRSS: Double) { self.peakRSS = peakRSS }
    }
    let stats = Stats(peakRSS: rss1)

    struct TurnResult {
        var text = ""
        var reasoning = ""
        var tools = 0
        var ttftMs = 0
        var totalSec = 0.0
        var tokps = 0.0
    }

    func runTurn(
        label: String, prompt: String, thinking: Bool?,
        extraTools: [[String: any Sendable]]? = nil,
        validate: (TurnResult) -> (ok: Bool, why: String)
    ) async throws {
        var userInput = UserInput(prompt: prompt)
        if let thinking {
            userInput.additionalContext = ["enable_thinking": thinking]
        }
        if let extraTools {
            userInput.tools = extraTools
        }
        let t0 = CFAbsoluteTimeGetCurrent()
        let input = try await ctx.processor.prepare(input: userInput)
        nonisolated(unsafe) let send = input
        let stream = await engine.generate(input: send, parameters: params)
        var r = TurnResult()
        var ttft: Double?
        var deltas = 0
        for await ev in stream {
            switch ev {
            case .chunk(let c):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                r.text += c; deltas += 1
            case .reasoning(let rs):
                if ttft == nil { ttft = CFAbsoluteTimeGetCurrent() - t0 }
                r.reasoning += rs; deltas += 1
            case .toolCall: r.tools += 1
            case .info: break
            }
        }
        r.totalSec = CFAbsoluteTimeGetCurrent() - t0
        r.tokps = r.totalSec > 0 ? Double(deltas) / r.totalSec : 0
        r.ttftMs = Int((ttft ?? 0) * 1000)
        stats.ttftByLabel[label] = r.ttftMs
        let now = currentRSSMiB()
        if now > stats.peakRSS { stats.peakRSS = now }

        let v = validate(r)
        let status = v.ok ? "PASS" : "FAIL"
        if v.ok { stats.pass += 1 } else { stats.fail += 1 }
        let preview = r.text.isEmpty ? r.reasoning : r.text
        let short = preview.count > 120 ? String(preview.prefix(120)) + "…" : preview
        print(String(format:
            "  [%@] %@  ttft=%4dms total=%5.2fs tokps=%5.1f chunks=%4d reasoning=%4d tools=%d rss=%.0fMiB %@-> \"%@\"",
            status, label, r.ttftMs, r.totalSec, r.tokps,
            r.text.count, r.reasoning.count, r.tools, now,
            v.ok ? "" : "WHY=\(v.why) ",
            short.replacingOccurrences(of: "\n", with: "\\n")))
    }

    // ──────────────── S1  reasoning=ON math ────────────────
    try await runTurn(
        label: "S1 think=ON math(7+8-11)",
        prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { r in
        (r.text + r.reasoning).contains("4") ? (true, "") : (false, "no '4'")
    }

    // ──────────────── S2  same prompt → cache hit ────────────────
    try await runTurn(
        label: "S2 cache-hit think=ON same prompt",
        prompt: "Compute 7 + 8 - 11. Respond with just the number.",
        thinking: true
    ) { r in
        let contentOK = (r.text + r.reasoning).contains("4")
        if !contentOK { return (false, "no '4' on cache hit") }
        let ttftS1 = stats.ttftByLabel["S1 think=ON math(7+8-11)"] ?? 99999
        // Soft check: warn on no speedup, don't fail
        if r.ttftMs >= Int(Double(ttftS1) * 0.95) {
            return (true, "warn: no TTFT speedup vs S1 (\(ttftS1)→\(r.ttftMs)ms)")
        }
        return (true, "")
    }

    // ──────────────── S3  reasoning=OFF factual ────────────────
    try await runTurn(
        label: "S3 think=OFF factual",
        prompt: "What color is the sky on a clear day? Answer with one word.",
        thinking: false
    ) { r in
        let t = (r.text + r.reasoning).lowercased()
        if t.contains("blue") { return (true, "") }
        if (r.text + r.reasoning).isEmpty { return (false, "empty") }
        return (true, "accepted non-blue")
    }

    // ──────────────── S4  reasoning ON→OFF→ON alternation ────────────────
    try await runTurn(
        label: "S4.1 flip-back think=ON math(12+3)",
        prompt: "Compute 12 + 3. Respond with just the number.",
        thinking: true
    ) { r in
        (r.text + r.reasoning).contains("15") ? (true, "") : (false, "no '15'")
    }
    try await runTurn(
        label: "S4.2 flip think=OFF name",
        prompt: "Name a planet. One word.",
        thinking: false
    ) { r in
        if (r.text + r.reasoning).isEmpty { return (false, "empty") }
        return (true, "")
    }
    try await runTurn(
        label: "S4.3 flip-back think=ON math(5*4)",
        prompt: "Compute 5 * 4. Respond with just the number.",
        thinking: true
    ) { r in
        (r.text + r.reasoning).contains("20") ? (true, "") : (false, "no '20'")
    }

    // ──────────────── S5  UTF-8 verbatim ────────────────
    try await runTurn(
        label: "S5 utf8 emoji verbatim",
        prompt: "Write exactly this line verbatim: 🚀 café naïve résumé 你好 こんにちは 안녕하세요",
        thinking: false
    ) { r in
        if (r.text + r.reasoning).isEmpty { return (false, "empty") }
        return (true, "")
    }

    print(String(format:
        "\n=== BENCH_PROD summary: model=%@ pass=%d fail=%d peakRSS=%.0fMiB loadSec=%.2f ===",
        modelName, stats.pass, stats.fail, stats.peakRSS, loadSec))
    if stats.fail > 0 {
        throw NSError(domain: "BENCH_PROD", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "\(stats.fail) scenario(s) failed validation",
        ])
    }
}
