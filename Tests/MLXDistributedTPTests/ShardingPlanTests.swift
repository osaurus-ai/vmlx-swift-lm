// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Tests for `ShardingPlan` — the declarative TP sharding directive
// surface that walks an MLXNN module tree and replaces matching
// `Linear`s with `AllToShardedLinear` / `ShardedToAllLinear` variants.
//
// These tests use a synthetic Llama-shaped module tree so they don't
// depend on a real bundle, run in milliseconds, and don't touch the
// MLX-distributed C symbols (group is size-1 in the size-1 path; the
// multi-rank walker is exercised by injecting a stub `Group` via the
// loopback bit-identity test in a separate file).

import Foundation
import MLX
import MLXNN
import Testing
@testable import MLXDistributedTP

@Suite("ShardingPlan")
struct ShardingPlanTests {

    // MARK: - Synthetic Llama-shaped module tree

    /// Mirrors `LlamaAttention` minus the RoPE/scale machinery — just
    /// enough `@ModuleInfo`-keyed Linears to exercise the walker.
    final class FakeAttention: Module {
        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int) {
            self._wq.wrappedValue = Linear(hidden, heads * headDim, bias: false)
            self._wk.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            self._wv.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
            self._wo.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        }
    }

    final class FakeMLP: Module {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(hidden: Int, intermediate: Int) {
            self._gate.wrappedValue = Linear(hidden, intermediate, bias: false)
            self._down.wrappedValue = Linear(intermediate, hidden, bias: false)
            self._up.wrappedValue = Linear(hidden, intermediate, bias: false)
        }
    }

    final class FakeBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: FakeAttention
        @ModuleInfo(key: "mlp") var mlp: FakeMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm

        init(hidden: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int) {
            self._attention.wrappedValue = FakeAttention(
                hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim)
            self._mlp.wrappedValue = FakeMLP(hidden: hidden, intermediate: intermediate)
            self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: hidden)
            self._postNorm.wrappedValue = RMSNorm(dimensions: hidden)
        }
    }

    final class FakeInner: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [FakeBlock]
        let norm: RMSNorm

        init(hidden: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int,
            vocab: Int, nLayers: Int)
        {
            self._embedTokens.wrappedValue = Embedding(
                embeddingCount: vocab, dimensions: hidden)
            self.layers = (0..<nLayers).map { _ in
                FakeBlock(hidden: hidden, intermediate: intermediate,
                    heads: heads, kvHeads: kvHeads, headDim: headDim)
            }
            self.norm = RMSNorm(dimensions: hidden)
        }
    }

    final class FakeLlamaModel: Module {
        let model: FakeInner
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(hidden: Int = 64, intermediate: Int = 128, heads: Int = 4,
            kvHeads: Int = 2, headDim: Int = 16, vocab: Int = 256, nLayers: Int = 2)
        {
            self.model = FakeInner(
                hidden: hidden, intermediate: intermediate,
                heads: heads, kvHeads: kvHeads, headDim: headDim,
                vocab: vocab, nLayers: nLayers)
            self._lmHead.wrappedValue = Linear(hidden, vocab, bias: false)
        }
    }

    // MARK: - Tests

    @Test("size-1 group: apply is a no-op (returns empty set)")
    func size1NoOp() {
        let model = FakeLlamaModel()
        let group = Group(strict: false)  // size-1 group when no env
        #expect(group.size == 1)
        let replaced = ShardingPlan.llama.apply(to: model, group: group)
        #expect(replaced.isEmpty)
    }

    @Test("Llama plan declares all 7 expected directives per layer")
    func llamaPlanDirectivesShape() {
        let plan = ShardingPlan.llama
        // q/k/v/o + gate/up/down = 7 keys
        #expect(plan.directiveCount == 7)
        #expect(plan.directives["self_attn.q_proj"] == .allToSharded(segments: 1))
        #expect(plan.directives["self_attn.k_proj"] == .allToSharded(segments: 1))
        #expect(plan.directives["self_attn.v_proj"] == .allToSharded(segments: 1))
        #expect(plan.directives["self_attn.o_proj"] == .shardedToAll(segments: 1))
        #expect(plan.directives["mlp.gate_proj"] == .allToSharded(segments: 1))
        #expect(plan.directives["mlp.up_proj"] == .allToSharded(segments: 1))
        #expect(plan.directives["mlp.down_proj"] == .shardedToAll(segments: 1))
    }

    @Test("flatten() of leafModules surfaces the expected dot paths")
    func flattenedDotPaths() {
        let model = FakeLlamaModel(nLayers: 2)
        let leaves = model.leafModules().flattened()
        let paths = leaves.map { $0.0 }.sorted()
        // Expected paths include q/k/v/o + gate/up/down per layer + lm_head
        // + embed_tokens + norm. This proves our suffix-match key shape
        // ("self_attn.q_proj", "mlp.gate_proj", …) is correct.
        #expect(paths.contains("model.layers.0.self_attn.q_proj"))
        #expect(paths.contains("model.layers.0.self_attn.k_proj"))
        #expect(paths.contains("model.layers.0.self_attn.v_proj"))
        #expect(paths.contains("model.layers.0.self_attn.o_proj"))
        #expect(paths.contains("model.layers.0.mlp.gate_proj"))
        #expect(paths.contains("model.layers.0.mlp.up_proj"))
        #expect(paths.contains("model.layers.0.mlp.down_proj"))
        #expect(paths.contains("model.layers.1.self_attn.q_proj"))
        #expect(paths.contains("lm_head"))
    }

    // (Multi-rank walker behavior is exercised by the loopback
    //  bit-identity test in `LlamaTPBitIdentityTests.swift`, which gates
    //  on a real `mlx_distributed_init` returning size > 1. We can't
    //  fake a multi-rank Group inside an XCTest process because env
    //  vars are read at process start.)
}
