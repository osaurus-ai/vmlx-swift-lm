// Copyright © 2026 Jinho Jang. All rights reserved.
//
// ShardingPlan — declarative description of which `Linear`s in a model
// graph become tensor-parallel `AllToShardedLinear` / `ShardedToAllLinear`
// variants when distributed across a `Group`.
//
// Mirrors the shape of `MLXLMCommon.BaseConfiguration.PerLayerQuantization`
// (which keys quantization decisions by module path with a global default).
// The same module-path keys are used here, so a future config-driven
// per-layer sharding override is a one-line `LinearShardingDirective` map
// addition.
//
// **Why declarative**: every model family (Llama, Mistral, Gemma, Qwen3.x,
// MoE families) needs the same conceptual mapping ("output-shard the
// q/k/v/gate/up; input-shard with all-reduce the o/down"), but with
// different module-path keys per family. Declarative lets us add new
// families with a single `ShardingPlans+<Family>.swift` file containing
// nothing but a static factory — no traversal logic per family.

import Foundation
import MLX
import MLXNN

// MARK: - Directives

/// What to do with one named `Linear` in a sharded forward pass.
public enum LinearShardingDirective: Sendable, Equatable {
    /// Replace with `AllToShardedLinear` — each rank holds a row-shard of
    /// the weight, output is sharded across the group along the output
    /// axis. Use for projections that produce sharded heads (q/k/v) or
    /// sharded MLP intermediate channels (gate/up).
    case allToSharded(segments: Int)

    /// Replace with `ShardedToAllLinear` — each rank holds a column-shard
    /// of the weight, partial outputs are summed via all-reduce. Use for
    /// projections that consume sharded inputs and produce a full-width
    /// output (o, down).
    case shardedToAll(segments: Int)

    /// Leave unsharded. Default for any path not in the directive map.
    case replicated
}

// MARK: - Plan

/// Sharding plan for one model family. Keys are MLXNN module-path
/// suffixes (the same path Module.modules() returns for nested children).
/// The walker matches a directive against any path that **ends with** the
/// directive key, so a single key like `"self_attn.q_proj"` covers every
/// transformer layer's query projection without enumerating layer indices.
///
/// To create a plan for a new family, write a static factory in a sibling
/// `ShardingPlans+<Family>.swift` file that returns a `ShardingPlan`
/// instance. The walker handles the rest.
public struct ShardingPlan: Sendable, Equatable {

    /// Map of module-path suffix → directive. The walker matches a
    /// module's full path against each key by `hasSuffix`, picking the
    /// longest match if multiple directives match. Paths not in the map
    /// receive `.replicated` (no transform).
    public let directives: [String: LinearShardingDirective]

    public init(directives: [String: LinearShardingDirective]) {
        self.directives = directives
    }

    /// Total number of directives — useful for tests asserting that
    /// `apply` actually rewrote the expected number of Linears.
    public var directiveCount: Int { directives.count }

    // MARK: - Walker

    /// Walk `root.leafModules()`, replace every `Linear` whose
    /// dot-path ends with a directive key, and call
    /// `Module.update(modules:)` to commit the replacements.
    ///
    /// Returns the set of full module paths that were actually replaced.
    /// Callers (tests / load-time mutators) can compare this against the
    /// expected count for the model family to surface plan/model drift.
    ///
    /// - Note: the walker is a no-op when `group.size == 1` because the
    ///   TP variants on a size-1 group produce bit-identical outputs to
    ///   the dense `Linear`. Skipping the rewrite avoids paying the
    ///   weight-slice + concat overhead on the unsharded path.
    ///
    /// Implementation pattern matches `QuantizedLinear.quantize(...)` in
    /// MLXNN: flatten leaves → compactMap to replacement modules →
    /// `update(modules:)`. The dotted path passed to the directive
    /// matcher is exactly the flattened key (e.g.
    /// `"model.layers.0.self_attn.q_proj"`).
    @discardableResult
    public func apply(to root: Module, group: Group) -> Set<String> {
        guard group.isMultiRank else { return [] }
        var replaced: Set<String> = []
        // We need the full dotted path for suffix matching, which
        // `compactMapValues` on NestedDictionary doesn't surface — use
        // `flattened()` to get `[(path, Module)]`, build a list of
        // replacements, then `unflattened()` back into the structure
        // `update(modules:)` expects.
        let flat: [(String, Module)] = root.leafModules().flattened()
        var rewrites: [(String, Module)] = []
        for (path, m) in flat {
            guard let linear = m as? Linear else { continue }
            guard let directive = bestDirectiveFor(path: path) else { continue }
            if let replacement = Self.makeReplacement(
                for: linear, directive: directive, group: group)
            {
                rewrites.append((path, replacement))
                replaced.insert(path)
            }
        }
        if !rewrites.isEmpty {
            root.update(modules: ModuleChildren.unflattened(rewrites))
        }
        return replaced
    }

    /// Find the directive whose key has the longest `hasSuffix` match
    /// against `path`. Longer-suffix-wins lets a per-layer override
    /// (`"model.layers.0.mlp.down_proj"`) take precedence over the
    /// family-default (`"mlp.down_proj"`).
    private func bestDirectiveFor(path: String)
        -> LinearShardingDirective?
    {
        var best: (key: String, directive: LinearShardingDirective)?
        for (key, directive) in directives {
            if path.hasSuffix(key) {
                if best == nil || key.count > best!.key.count {
                    best = (key, directive)
                }
            }
        }
        return best?.directive
    }

    /// Construct the replacement TP variant from a dense `Linear` per
    /// the directive. Returns nil for `.replicated` (caller skips).
    private static func makeReplacement(
        for linear: Linear,
        directive: LinearShardingDirective,
        group: Group
    ) -> Module? {
        switch directive {
        case .allToSharded(let segments):
            return AllToShardedLinear.from(
                linear, group: group, segments: segments)
        case .shardedToAll(let segments):
            return ShardedToAllLinear.from(
                linear, group: group, segments: segments)
        case .replicated:
            return nil
        }
    }
}
