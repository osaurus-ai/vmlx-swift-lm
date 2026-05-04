// Copyright © 2026 Jinho Jang. All rights reserved.
//
// Llama-family sharding plan. Covers the Llama and Mistral model classes
// in `MLXLLM/Models/Llama.swift` (which share the same `LlamaAttention`
// + `LlamaMLP` building blocks with `q_proj` / `k_proj` / `v_proj` /
// `o_proj` and `gate_proj` / `down_proj` / `up_proj` keys).
//
// Layer-uniform plan, segments=1 everywhere (Llama has no fused
// projections — q/k/v are separate Linears, gate/up are separate too).
// All transformer-block-internal Linears are sharded; norms, embeddings,
// and lm_head stay replicated.

import Foundation

extension ShardingPlan {

    /// Sharding plan for the Llama / Mistral family.
    ///
    /// Per-layer Linears:
    /// - `self_attn.q_proj` / `self_attn.k_proj` / `self_attn.v_proj`
    ///   → `AllToShardedLinear` (output-sharded along head axis).
    /// - `self_attn.o_proj` → `ShardedToAllLinear` (input-sharded with
    ///   all-reduce of partial outputs back to full hidden size).
    /// - `mlp.gate_proj` / `mlp.up_proj`
    ///   → `AllToShardedLinear` (output-sharded along intermediate axis).
    /// - `mlp.down_proj` → `ShardedToAllLinear` (all-reduce).
    ///
    /// Replicated (not in the directive map, so default `.replicated`):
    /// `model.embed_tokens`, every `RMSNorm`, `model.norm`, `lm_head`.
    /// Both ranks compute the lm_head matmul on the full final hidden
    /// state — output is identical and sampling can happen on rank 0
    /// alone.
    ///
    /// Divisibility constraints (validated at apply-time by the
    /// `AllToShardedLinear.from` / `ShardedToAllLinear.from`
    /// preconditions):
    /// - `attentionHeads % world_size == 0`
    /// - `kvHeads % world_size == 0`
    /// - `intermediateSize % world_size == 0`
    public static let llama = ShardingPlan(directives: [
        // Attention
        "self_attn.q_proj": .allToSharded(segments: 1),
        "self_attn.k_proj": .allToSharded(segments: 1),
        "self_attn.v_proj": .allToSharded(segments: 1),
        "self_attn.o_proj": .shardedToAll(segments: 1),
        // MLP
        "mlp.gate_proj":    .allToSharded(segments: 1),
        "mlp.up_proj":      .allToSharded(segments: 1),
        "mlp.down_proj":    .shardedToAll(segments: 1),
    ])
}
