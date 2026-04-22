// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT

import Foundation
import MLX
import os

// MARK: - Stage-local model shape

/// Adapter protocol a model conforms to so it can run **one stage's
/// worth** of transformer layers in isolation.
///
/// Pipeline-parallel inference requires splitting a model's forward
/// pass into slices that can be executed on different devices. This
/// protocol is the contract between the generic ``PipelineEngine`` and
/// the concrete per-model stage-forward code.
///
/// Implementing it for a new model family is a mechanical exercise:
/// for each of the three call sites below, call the already-existing
/// per-layer functions the model exposes internally. The model doesn't
/// need to expose its full layer array — it only needs to know how to
/// execute a sub-range.
public protocol PipelineStageModel: AnyObject {

    /// Embed a `[B, T]` int32 token tensor to a `[B, T, H]` hidden-state
    /// tensor. Called only on stage 0 (`ModelPartition.isFirstStage`).
    /// Any pre-layer scaling (e.g. Gemma's `sqrt(hiddenSize)` factor)
    /// belongs here.
    func embedTokens(_ tokens: MLXArray) -> MLXArray

    /// Run the transformer layers in `partition.layerRange` over the
    /// hidden state. Mutates the KV cache in place just like a normal
    /// forward pass — the caller owns the cache array for that layer
    /// range.
    ///
    /// `cache` covers ALL layers in the model (allocated by
    /// ``newCache(parameters:)`` below). The implementation indexes
    /// into it with `partition.layerRange`.
    func runLayers(
        hidden: MLXArray,
        partition: ModelPartition,
        cache: [KVCache]?
    ) -> MLXArray

    /// Apply the final RMSNorm + lm_head to produce `[B, T, V]` logits.
    /// Called only on the last stage (`ModelPartition.isLastStage`).
    func finalizeLogits(_ hidden: MLXArray) -> MLXArray

    /// Allocate a fresh full-model KV cache array. `PipelineEngine`
    /// slices into this per-stage by layer range — each stage's layers
    /// own the matching subset, everything outside the range is allocated
    /// but never written.
    ///
    /// Pre-allocating the whole array (instead of only the stage's
    /// slice) keeps the per-layer index stable across stages, which
    /// matches how ``runLayers(hidden:partition:cache:)`` already
    /// expects to index. Phase 2 will tighten this to allocate only
    /// the stage's slice when we formalize cache-sizing per stage.
    func newCache(parameters: GenerateParameters) -> [KVCache]

    /// Total number of transformer layers the model has. Used by
    /// ``ModelPartition/uniform(totalLayers:stageCount:stageIndex:)`` to
    /// compute the layer range for this rank.
    var totalLayerCount: Int { get }
}

// MARK: - PipelineEngine

/// Orchestrator for a single forward pass across multiple pipeline
/// stages. Phase 1 scope: single-request, text-only, ``KVCacheSimple``
/// only. Phase 3 will integrate with ``BatchEngine`` for continuous
/// batching; Phase 4 will add hybrid-SSM + VLM support.
///
/// ## Concurrency model
///
/// This type is NOT an actor. Each process (rank) runs its own
/// instance. Coordination between ranks happens via
/// ``MLXDistributed/send(_:dst:group:stream:)`` and ``recv(shape:...)``.
/// The ``MLXDistributed`` module handles rank discovery + group
/// construction; this engine is policy only.
///
/// ## Lifecycle
///
/// 1. Every rank calls ``MLXDistributed/initialize(strict:backend:)`` to
///    join the world group.
/// 2. Every rank constructs its own ``ModelPartition`` using its rank
///    and the total layer count.
/// 3. Every rank loads the full model (Phase 1; Phase 2 adds selective
///    load) and instantiates a ``PipelineEngine`` wrapping it.
/// 4. Rank 0 drives ``runPrompt(tokens:parameters:maxNewTokens:)`` which
///    threads the prompt through the pipeline.
/// 5. Non-zero ranks drive ``runWorker(parameters:)`` which loops
///    receiving hidden-state tensors from `prevRank` and sending to
///    `nextRank` (or emitting logits on the last rank).
///
/// ## Shape contract between stages
///
/// All hidden-state tensors exchanged between stages are `[B, T, H]`
/// in the model's native dtype (typically bf16 or fp16). Stage 0 sends
/// the post-embed, pre-first-layer output. Intermediate stages send
/// post-last-owned-layer output. The last stage produces logits locally
/// and does not send anything downstream.
public final class PipelineEngine {

    private static let logger = Logger(subsystem: "vmlx", category: "PipelineEngine")

    /// The model adapter this engine drives. Held by reference so the
    /// engine can be constructed without transferring ownership.
    public let model: PipelineStageModel

    /// This rank's partition.
    public let partition: ModelPartition

    /// World group — typically `MLXDistributed.worldGroup`. Kept on the
    /// engine so every send/recv uses the same group consistently.
    public let group: MLXDistributedGroup

    public init(
        model: PipelineStageModel,
        partition: ModelPartition,
        group: MLXDistributedGroup
    ) {
        precondition(partition.stageCount == group.size,
            "partition.stageCount (\(partition.stageCount)) must equal group.size (\(group.size))")
        precondition(partition.stageIndex == group.rank,
            "partition.stageIndex (\(partition.stageIndex)) must equal group.rank (\(group.rank))")
        self.model = model
        self.partition = partition
        self.group = group
    }

    // MARK: - Rank 0 driver

    /// Run a prompt through the pipeline to produce up to
    /// `maxNewTokens` generated token IDs. Rank 0 only. Other ranks
    /// should be running ``runWorker(parameters:)`` concurrently.
    ///
    /// Phase 1 generates greedily by default — sampling parameters from
    /// `parameters` are applied on the last stage. Streaming + per-token
    /// callbacks are Phase 3 (continuous batching integration).
    ///
    /// - Parameters:
    ///   - tokens: prompt tokens as a `[1, T]` int32 MLXArray.
    ///   - parameters: generation parameters (maxTokens / temperature /
    ///     topP / etc.). Only the sampling-related fields apply on the
    ///     last stage; cache-related fields apply on every stage.
    ///   - maxNewTokens: cap on generated tokens regardless of what
    ///     `parameters.maxTokens` says. Primarily a safety valve.
    /// - Returns: generated token IDs in order. Empty if the first
    ///   token was an EOS.
    public func runPrompt(
        tokens: MLXArray,
        parameters: GenerateParameters,
        maxNewTokens: Int,
        stopTokenIDs: Set<Int> = []
    ) -> [Int] {
        precondition(partition.isFirstStage,
            "runPrompt must be driven from stage 0 / rank 0")
        precondition(tokens.ndim == 2 && tokens.dim(0) == 1,
            "tokens must be [1, T] — Phase 1 is single-request")

        let cache = model.newCache(parameters: parameters)

        // Prefill: one stage's layers, then forward the hidden state to
        // the next stage. Rank 0 embeds tokens; middle+last ranks are
        // waiting in their runWorker loop to receive.
        var hidden = model.embedTokens(tokens)
        hidden = model.runLayers(
            hidden: hidden, partition: partition, cache: cache
        )

        if !partition.isLastStage {
            sendWithEnvelope(hidden, dst: partition.nextRank)
        }

        // Decode loop. On each step rank 0 receives the previous step's
        // sampled token ID back from the last rank (via a dedicated
        // token-broadcast send), then runs its own layers over the new
        // token's embedding, and forwards again.
        //
        // For a 2-stage pipeline with N ranks, the control flow is:
        //   rank 0: embed → layers[0..k] → send hidden → recv sampled
        //   rank N-1: recv hidden → layers[k..L] → logits → sample →
        //            broadcast sampled token id to every rank
        //
        // Broadcasting the sampled token id back to rank 0 gives every
        // rank the next input token for the NEXT forward step, so each
        // rank embeds/runs on the same token sequence.
        var generated: [Int] = []
        generated.reserveCapacity(maxNewTokens)

        // Rank 0 needs the first sampled token from the last rank.
        // Everyone receives the sampled id so they can embed the next
        // token locally. Use int32 [1] for the broadcast.
        var sampledID = Self.recvSampledToken(from: partition.stageCount - 1, group: group)

        for _ in 0 ..< maxNewTokens {
            let tokenID = Int(sampledID.item(Int32.self))
            if stopTokenIDs.contains(tokenID) { break }
            generated.append(tokenID)

            // Rank 0 embeds the just-sampled token and runs its layers.
            let nextInput = MLXArray([Int32(tokenID)])[.newAxis, 0...]
            hidden = model.embedTokens(nextInput)
            hidden = model.runLayers(
                hidden: hidden, partition: partition, cache: cache
            )
            sendWithEnvelope(hidden, dst: partition.nextRank)

            sampledID = Self.recvSampledToken(
                from: partition.stageCount - 1, group: group)
        }

        // Terminator — zero envelope tells every worker to exit its
        // loop. Sent only by rank 0 on decode completion.
        if !partition.isLastStage {
            let zero = MLXArray([Int32(0), 0, 0])
            _ = MLXDistributed.send(zero, dst: partition.nextRank, group: group)
        }

        return generated
    }

    // MARK: - Worker driver (non-zero ranks)

    /// Loop forever forwarding hidden states through this stage's
    /// layers until rank 0 signals termination by sending a zero-sized
    /// payload. Middle ranks forward to `nextRank`; the last rank
    /// produces logits, samples, and broadcasts the sampled token ID
    /// back to rank 0 for its next step.
    ///
    /// - Parameter parameters: generation parameters (uses the sampler
    ///   on the last rank; middle ranks use the `maxKVSize`/`kvMode`
    ///   fields for cache allocation).
    public func runWorker(parameters: GenerateParameters) {
        precondition(!partition.isFirstStage,
            "runWorker runs on stage > 0; stage 0 should call runPrompt")

        let cache = model.newCache(parameters: parameters)
        let sampler = parameters.sampler()
        var processor = parameters.processor()

        // Worker loop: peek at a small control message each iteration,
        // then receive the hidden payload shaped by the previous step.
        // Phase 1 uses a very simple protocol: each step sends one
        // hidden tensor whose outer time dim is either `T` (prefill)
        // or `1` (decode). On termination rank 0 broadcasts a special
        // zero-shape marker — but since mlx-core's `recv` requires the
        // shape at receive time, we instead use a fixed hidden-dim
        // probe: receive a `[1, 1, H]` scalar whose first element is
        // NaN as the terminator.
        //
        // Actual shape inference: the first receive is the prefill
        // `[1, T_prompt, H]`. After that every receive is `[1, 1, H]`
        // for decode. We look at the incoming shape on each step.
        //
        // Simpler approach for Phase 1: the worker just loops on
        // `recvLike` with a rank-0-advertised shape.
        //
        // We take the even-simpler approach: rank 0 always sends a
        // shape envelope first, then the tensor. The envelope is a
        // `[3]` int32 tensor of `[B, T, H]`. A `[0, 0, 0]` envelope
        // terminates the worker loop.
        while true {
            let envelope = MLXDistributed.recv(
                shape: [3], dtype: .int32,
                src: partition.prevRank, group: group
            )
            let dims = envelope.asArray(Int32.self).map { Int($0) }
            if dims == [0, 0, 0] {
                // Propagate terminator downstream so every stage exits,
                // then exit ourselves. Only middle stages need to
                // forward — the last stage has no downstream.
                if !partition.isLastStage {
                    let zero = MLXArray([Int32(0), 0, 0])
                    _ = MLXDistributed.send(
                        zero, dst: partition.nextRank, group: group)
                }
                break
            }

            let hiddenShape = [dims[0], dims[1], dims[2]]
            let hiddenDtype: DType = .bfloat16  // Phase 1 assumption

            var hidden = MLXDistributed.recv(
                shape: hiddenShape, dtype: hiddenDtype,
                src: partition.prevRank, group: group
            )
            hidden = model.runLayers(
                hidden: hidden, partition: partition, cache: cache
            )

            if partition.isLastStage {
                let logits = model.finalizeLogits(hidden)
                // Take logits for the LAST time step, sample one token.
                let stepLogits = logits[0 ..< 1, -1, 0...]
                var finalLogits = stepLogits
                if var proc = processor {
                    finalLogits = proc.process(logits: stepLogits)
                    processor = proc
                }
                let token = sampler.sample(logits: finalLogits)
                if var proc = processor {
                    proc.didSample(token: token)
                    processor = proc
                }
                // Broadcast the sampled token back to rank 0 so it can
                // embed + start the next step.
                _ = MLXDistributed.send(token, dst: 0, group: group)
            } else {
                // Middle stage: forward to next, and send envelope for it too.
                let outShape = hidden.shape.map { Int32($0) }
                let outEnvelope = MLXArray(outShape)
                _ = MLXDistributed.send(
                    outEnvelope, dst: partition.nextRank, group: group)
                _ = MLXDistributed.send(
                    hidden, dst: partition.nextRank, group: group)
            }
        }
    }

    // MARK: - Internals

    /// Helper: receive the one-scalar `[1]` int32 "sampled token ID"
    /// that the last rank broadcasts back after each decode step.
    private static func recvSampledToken(
        from src: Int,
        group: MLXDistributedGroup
    ) -> MLXArray {
        MLXDistributed.recv(
            shape: [1], dtype: .int32,
            src: src, group: group
        )
    }

    /// Helper driven from rank 0's `runPrompt` — augmented to send the
    /// size envelope that `runWorker` expects. Callers of `runPrompt`
    /// don't see this; it's invoked on every stage-0 `send`.
    @discardableResult
    private func sendWithEnvelope(_ hidden: MLXArray, dst: Int) -> MLXArray {
        let envelope = MLXArray(hidden.shape.map { Int32($0) })
        _ = MLXDistributed.send(envelope, dst: dst, group: group)
        return MLXDistributed.send(hidden, dst: dst, group: group)
    }
}
