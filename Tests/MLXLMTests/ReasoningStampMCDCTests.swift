// Copyright © 2026 osaurus.
//
// MC/DC tests for `reasoningStampFromModelType` (ReasoningParser.swift:467).
//
// Decision tree:
//
//   D1: modelType != nil ∧ !modelType.isEmpty
//     ↓ false → return "none"
//     ↓ true
//   D2: t.hasPrefix("gemma4")
//     ↓ true  → return "harmony"
//     ↓ false
//   D3: thinkXmlPrefixes.contains(where: t.hasPrefix)
//        — disjunction of 9 prefix checks (one per family)
//     ↓ true  → return "think_xml"
//     ↓ false → return "none"
//
// MC/DC requirements:
//
//  - D1 (∧): need (T,T)→T, (F,T), (T,F) — 3 cases.
//  - D2 (single condition): 2 cases, T and F.
//  - D3 (∨ over 9 prefixes): each prefix must independently flip the
//    decision. Need 1 all-false case + 9 cases each with exactly one
//    matching prefix and all 8 others non-matching = 10 cases.
//
// Plus a master "all-default" case where t doesn't match any branch.

import Foundation
import MLXLMCommon
import Testing

@Suite("reasoningStampFromModelType — MC/DC coverage")
struct ReasoningStampMCDCTests {

    // MARK: - D1: nil/empty guard (∧)

    @Test("D1 a=F (nil) → 'none'")
    func d1_nilModelType() {
        #expect(reasoningStampFromModelType(nil) == "none")
    }

    @Test("D1 b=F (empty) → 'none'")
    func d1_emptyModelType() {
        #expect(reasoningStampFromModelType("") == "none")
    }

    @Test("D1 a∧b=T then proceeds (case-folded match too)")
    func d1_bothTrueProceeds() {
        // Non-nil, non-empty → falls through to D2/D3. Use a known
        // think_xml family so we observe the proceed path concretely.
        #expect(reasoningStampFromModelType("qwen3") == "think_xml")
    }

    // MARK: - D2: gemma4 prefix branch

    @Test("D2 T (gemma4) → 'harmony'")
    func d2_gemma4_true() {
        #expect(reasoningStampFromModelType("gemma4") == "harmony")
    }

    @Test("D2 T (gemma4_27b minor variant) → 'harmony' via prefix")
    func d2_gemma4Variant_true() {
        #expect(reasoningStampFromModelType("gemma4_27b") == "harmony")
    }

    @Test("D2 F (gemma3 — not gemma4) → falls through")
    func d2_gemma3_false() {
        // Gemma 3 has no harmony channel → not "harmony".
        // Also not in think_xml list → "none".
        #expect(reasoningStampFromModelType("gemma3") == "none")
    }

    // MARK: - D3: think_xml prefix disjunction
    //
    // Each test demonstrates one prefix independently producing
    // think_xml, with the rest non-matching. The "all-false" case at
    // the bottom of this section pairs to show D3 false → "none".

    @Test("D3.qwen3 prefix flips decision")
    func d3_qwen3() {
        #expect(reasoningStampFromModelType("qwen3") == "think_xml")
        #expect(reasoningStampFromModelType("qwen3_5") == "think_xml")
        #expect(reasoningStampFromModelType("qwen3_next_moe") == "think_xml")
    }

    @Test("D3.deepseek prefix flips decision")
    func d3_deepseek() {
        #expect(reasoningStampFromModelType("deepseek") == "think_xml")
        #expect(reasoningStampFromModelType("deepseek_v3") == "think_xml")
        #expect(reasoningStampFromModelType("deepseek_v4") == "think_xml")
    }

    @Test("D3.glm4_moe prefix flips decision (and glm4 alone does NOT)")
    func d3_glm4Moe() {
        #expect(reasoningStampFromModelType("glm4_moe") == "think_xml")
        #expect(reasoningStampFromModelType("glm4_moe_lite") == "think_xml")
        // glm4 alone is NOT in the prefix list — it's only glm4_moe and glm5.
        // Locks the boundary: a future drift that adds bare "glm4" would
        // accidentally promote dense Gemma-style GLM4 models to think_xml.
        #expect(reasoningStampFromModelType("glm4") == "none",
            "bare 'glm4' must NOT match — only glm4_moe and glm5")
    }

    @Test("D3.glm5 prefix flips decision")
    func d3_glm5() {
        #expect(reasoningStampFromModelType("glm5") == "think_xml")
        #expect(reasoningStampFromModelType("glm5_air") == "think_xml")
    }

    @Test("D3.minimax prefix flips decision")
    func d3_minimax() {
        #expect(reasoningStampFromModelType("minimax") == "think_xml")
        #expect(reasoningStampFromModelType("minimax_m2") == "think_xml")
    }

    @Test("D3.kimi prefix flips decision")
    func d3_kimi() {
        #expect(reasoningStampFromModelType("kimi") == "think_xml")
        #expect(reasoningStampFromModelType("kimi_k25") == "think_xml")
    }

    @Test("D3.nemotron_h prefix flips decision (bare 'nemotron' does NOT)")
    func d3_nemotronH() {
        #expect(reasoningStampFromModelType("nemotron_h") == "think_xml")
        // Bare "nemotron" is NOT in the prefix list — only nemotron_h.
        // Locks the boundary against drift that would over-accept old
        // Nemotron-2 / NeMo dense bundles that don't emit <think>.
        #expect(reasoningStampFromModelType("nemotron") == "none",
            "bare 'nemotron' must NOT match — only nemotron_h")
    }

    @Test("D3.holo prefix flips decision")
    func d3_holo() {
        #expect(reasoningStampFromModelType("holo") == "think_xml")
        #expect(reasoningStampFromModelType("holo3") == "think_xml")
    }

    @Test("D3.laguna prefix flips decision (Bug-3a sibling pre-registration)")
    func d3_laguna() {
        #expect(reasoningStampFromModelType("laguna") == "think_xml")
        #expect(reasoningStampFromModelType("laguna_xs") == "think_xml")
        #expect(reasoningStampFromModelType("Laguna") == "think_xml",
            "case-insensitive match")
    }

    /// All 9 prefixes FALSE simultaneously → D3 disjunction is FALSE → "none".
    /// Pair-completes the MC/DC table: shows that with no prefix matching,
    /// the decision flips to FALSE regardless of which prefix is "tested".
    @Test("D3 all-false → 'none' (LFM2, LLaMA, Phi, etc.)")
    func d3_allFalse() {
        let nonReasoningTypes = [
            "lfm2",          // Liquid LFM-2
            "llama",         // LLaMA dense
            "phi",           // Microsoft Phi
            "starcoder2",
            "openelm",
            "internlm2",
            "gpt_oss",
            "nanochat",
            "mistral",       // Mistral 3 / 4 — no <think>
            "ministral3",    // Mistral 3.5 inner — no <think>
        ]
        for t in nonReasoningTypes {
            #expect(reasoningStampFromModelType(t) == "none",
                "\(t) must resolve to 'none' (no prefix match)")
        }
    }

    // MARK: - Case-folding (the .lowercased() pre-pass)

    @Test("Case-folding works for all branches")
    func caseFolding_allBranches() {
        #expect(reasoningStampFromModelType("GEMMA4") == "harmony")
        #expect(reasoningStampFromModelType("Qwen3") == "think_xml")
        #expect(reasoningStampFromModelType("DEEPSEEK_V4") == "think_xml")
        #expect(reasoningStampFromModelType("LFM2") == "none")
    }
}
