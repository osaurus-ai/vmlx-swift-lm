// Copyright © 2026 Osaurus AI. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Empty module header for the MLXDistributedCFallback SwiftPM target.
//
// The fallback target exists solely to compile MLXDistributedFallback.c,
// which provides weak-alias implementations of the `mlx_distributed_*`
// symbols referenced by the Swift bindings in
// `Libraries/MLXLMCommon/Distributed/MLXDistributedFallback.swift`.
// See the top of MLXDistributedFallback.c for the full rationale.
//
// Nothing in this header needs to be imported from Swift — the fallback is
// consumed at link time, not at the module-import level. This header exists
// only because SwiftPM requires C targets to declare a `publicHeadersPath`.

#ifndef VMLX_DISTRIBUTED_FALLBACK_H
#define VMLX_DISTRIBUTED_FALLBACK_H

#endif
