// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Single-rank fallback implementations of the mlx-core distributed C API.
//
// WHY THIS EXISTS
// ---------------
// Upstream mlx-swift (osaurus-ai/mlx-swift @ osaurus-0.31.3) deliberately
// EXCLUDES mlx's distributed sources from its Cmlx build target. Its
// Package.swift line 193 comments: "do not build distributed support (yet)"
// and the two C-bridge sources `mlx-c/mlx/c/distributed.cpp` and
// `mlx-c/mlx/c/distributed_group.cpp` are also listed as excludes (lines 116
// and 117). Result: our Swift distributed bindings in `MLXDistributed.swift`
// reference the symbols `mlx_distributed_init`, `mlx_distributed_all_sum`,
// `mlx_distributed_send`, `mlx_distributed_recv`, etc. — which have public
// headers in Cmlx but no compiled implementation.
//
// Two ways to close that gap:
//
//   A. Patch `osaurus-ai/mlx-swift` to INCLUDE the distributed sources and
//      at least one backend (ring over TCP is the simplest Apple-friendly
//      option). That's Phase 0.5 in `DISTRIBUTED-DESIGN.md` and is the
//      eventual plan for multi-rank operation.
//
//   B. Ship this fallback stub as part of vmlx-swift-lm so the Swift API
//      LINKS and RUNS CORRECTLY on a single rank today. Every collective
//      reduces to the identity on size 1, which is provably correct.
//
// This file is option B. It's a conservative stub that lets the Swift
// binding compile against the published mlx-c headers without requiring any
// upstream change to mlx-swift. It also gives callers a working code path
// for dev-box work before a two-Mac rig is wired up.
//
// WHEN YOU MOVE TO MULTI-RANK
// ---------------------------
// After upstream is patched (Phase 0.5), the real `mlx_distributed_*`
// symbols will be provided by the Cmlx library. This file's symbols are
// all declared with `__attribute__((weak))` so the linker prefers the real
// implementations when both are visible. No Swift-side change will be
// required — the same `MLXDistributed.swift` code will route to real
// collectives automatically.
//
// CORRECTNESS ON SIZE 1
// ---------------------
// For a group of size 1:
//   - all_sum(x)       ≡ x       (the only contribution is x itself)
//   - all_gather(x)    ≡ x       (concatenating one array yields the same array)
//   - all_max(x)       ≡ x       (max over a single value is that value)
//   - all_min(x)       ≡ x       (likewise)
//   - sum_scatter(x)   ≡ x       (scattering one slice to one rank is the slice)
//   - send(x, dst=0)   — returns x; no-op since we're the destination
//   - recv(...) / recv_like(...) — these are erroneous on size 1 since
//     there is no peer to receive from; the fallback returns an empty array
//     and the caller is expected to gate via `MLXDistributed.worldGroup?.isMultiRank`.
//
// SEE ALSO
// --------
//   Libraries/MLXLMCommon/Distributed/MLXDistributed.swift
//   Libraries/MLXLMCommon/Distributed/DISTRIBUTED-DESIGN.md
//   .build/checkouts/mlx-swift/Source/Cmlx/mlx-c/mlx/c/distributed.h
//   .build/checkouts/mlx-swift/Source/Cmlx/mlx-c/mlx/c/distributed_group.h

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// Forward declarations matching `mlx-c/mlx/c/distributed_group.h` and
// `mlx-c/mlx/c/distributed.h`. We cannot include the headers directly
// here because they drag in the whole mlx-c header tree which is only
// present at the Cmlx build step. Declaring the opaque `mlx_array` as
// an opaque pointer struct is sufficient for the stubs below: we never
// inspect its contents.

typedef struct mlx_array_ {
    void* ctx;
} mlx_array;

typedef struct mlx_stream_ {
    void* ctx;
} mlx_stream;

typedef struct mlx_distributed_group_ {
    void* ctx;
} mlx_distributed_group;

// Matches mlx-c's mlx_dtype enum ordering. Unused by the fallback stubs
// but declared for signature parity so the linker accepts our weak aliases.
typedef enum mlx_dtype_ {
    VMLX_DISTRIB_STUB_DTYPE_UNUSED = 0
} mlx_dtype;

// A heap-allocated `mlx::core::distributed::Group` in the real build.
// Here we use a per-process stub object that carries rank=0 size=1. The
// `ctx` pointer itself serves as a sentinel that `init` was called.
static char vmlx_distributed_stub_group_sentinel = 0;

// ============================================================================
// distributed_group.h
// ============================================================================

__attribute__((weak))
bool mlx_distributed_is_available(const char* bk) {
    (void)bk;
    // Always report availability so `MLXDistributed.isAvailable()` returns
    // true. Callers that need to distinguish real multi-rank from single-rank
    // fallback should check `worldGroup?.isMultiRank` instead.
    return true;
}

__attribute__((weak))
mlx_distributed_group mlx_distributed_init(bool strict, const char* bk) {
    (void)strict;
    (void)bk;
    mlx_distributed_group g;
    g.ctx = &vmlx_distributed_stub_group_sentinel;
    return g;
}

__attribute__((weak))
int mlx_distributed_group_rank(mlx_distributed_group group) {
    (void)group;
    return 0;
}

__attribute__((weak))
int mlx_distributed_group_size(mlx_distributed_group group) {
    (void)group;
    return 1;
}

__attribute__((weak))
mlx_distributed_group mlx_distributed_group_split(
    mlx_distributed_group group, int color, int key)
{
    (void)color;
    (void)key;
    return group;
}

// ============================================================================
// distributed.h — collective ops
// ============================================================================
//
// Each op signature mirrors mlx-c. On a size-1 group all collectives are
// the identity, so we implement them by copying `x` into `*res` through
// the mlx-c `mlx_array_set` symbol. That symbol IS exposed by Cmlx in the
// shipped build because `array.cpp` is not on the exclude list. We declare
// it here with an extern prototype.

extern int mlx_array_set(mlx_array* dst, const mlx_array src);

static int vmlx_distributed_stub_identity(mlx_array* res, const mlx_array x) {
    // Mirror mlx-c's other op error code: 0 == ok.
    return mlx_array_set(res, x);
}

__attribute__((weak))
int mlx_distributed_all_sum(
    mlx_array* res, const mlx_array x,
    const mlx_distributed_group group, const mlx_stream s)
{
    (void)group; (void)s;
    return vmlx_distributed_stub_identity(res, x);
}

__attribute__((weak))
int mlx_distributed_all_gather(
    mlx_array* res, const mlx_array x,
    const mlx_distributed_group group, const mlx_stream s)
{
    (void)group; (void)s;
    return vmlx_distributed_stub_identity(res, x);
}

__attribute__((weak))
int mlx_distributed_all_max(
    mlx_array* res, const mlx_array x,
    const mlx_distributed_group group, const mlx_stream s)
{
    (void)group; (void)s;
    return vmlx_distributed_stub_identity(res, x);
}

__attribute__((weak))
int mlx_distributed_all_min(
    mlx_array* res, const mlx_array x,
    const mlx_distributed_group group, const mlx_stream s)
{
    (void)group; (void)s;
    return vmlx_distributed_stub_identity(res, x);
}

__attribute__((weak))
int mlx_distributed_sum_scatter(
    mlx_array* res, const mlx_array x,
    const mlx_distributed_group group, const mlx_stream s)
{
    (void)group; (void)s;
    return vmlx_distributed_stub_identity(res, x);
}

__attribute__((weak))
int mlx_distributed_send(
    mlx_array* res, const mlx_array x,
    int dst, const mlx_distributed_group group, const mlx_stream s)
{
    (void)dst; (void)group; (void)s;
    // On a single-rank fallback there is no real send. Return x as the
    // placeholder so the eval graph stays well-formed — same semantics the
    // mlx-core send returns in the multi-rank case.
    return vmlx_distributed_stub_identity(res, x);
}

__attribute__((weak))
int mlx_distributed_recv(
    mlx_array* res,
    const int* shape, size_t shape_num, mlx_dtype dtype,
    int src, const mlx_distributed_group group, const mlx_stream s)
{
    (void)res; (void)shape; (void)shape_num; (void)dtype;
    (void)src; (void)group; (void)s;
    // `recv` on a single-rank group is a caller error — there's no peer to
    // receive from. Log once and return a non-zero error code so Swift can
    // surface it. We intentionally do not mutate `*res`.
    static int logged = 0;
    if (!logged) {
        fprintf(stderr,
            "[vmlx][distributed] mlx_distributed_recv called on size-1 fallback "
            "group — no peer exists; patch osaurus-ai/mlx-swift to enable a "
            "real distributed backend for multi-rank operation.\n");
        logged = 1;
    }
    return 1;  // non-zero = error in the mlx-c ABI
}

__attribute__((weak))
int mlx_distributed_recv_like(
    mlx_array* res, const mlx_array x,
    int src, const mlx_distributed_group group, const mlx_stream s)
{
    (void)res; (void)x; (void)src; (void)group; (void)s;
    static int logged = 0;
    if (!logged) {
        fprintf(stderr,
            "[vmlx][distributed] mlx_distributed_recv_like called on size-1 "
            "fallback group — no peer exists.\n");
        logged = 1;
    }
    return 1;
}
