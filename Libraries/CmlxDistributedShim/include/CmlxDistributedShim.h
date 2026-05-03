// Phase 5 C bridge: exposes mlx-c's distributed collectives + group
// lifecycle as Swift-friendly opaque-pointer functions, hiding the
// fact that mlx_array and mlx_distributed_group are single-pointer
// structs passed by value (which @_silgen_name handles fine in
// theory but is fragile in practice).
//
// Each collective takes/returns an `MLXArrayPtr` — opaque pointer
// to the underlying mlx_array's ctx. Callers turn that into a Swift
// MLXArray via the bridging shim in Collectives.swift.

#ifndef VMLX_CMLX_DISTRIBUTED_SHIM_H
#define VMLX_CMLX_DISTRIBUTED_SHIM_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types.
typedef void* VMLXArray;
typedef void* VMLXGroup;
typedef void* VMLXStream;

// Lifecycle for arrays we synthesize.
void vmlx_array_free(VMLXArray a);

// Group helpers (delegated to mlx_distributed_group_*).
int  vmlx_group_rank(VMLXGroup g);
int  vmlx_group_size(VMLXGroup g);
VMLXGroup vmlx_group_init(bool strict, const char* backend);
VMLXGroup vmlx_group_split(VMLXGroup g, int color, int key);
bool vmlx_distributed_is_available(const char* backend);

// Collectives. Each returns 0 on success, non-zero on error. The result
// pointer must be freed by the caller via vmlx_array_free.
int vmlx_all_sum(VMLXArray* res, VMLXArray x, VMLXGroup g, VMLXStream s);
int vmlx_all_gather(VMLXArray* res, VMLXArray x, VMLXGroup g, VMLXStream s);
int vmlx_send(VMLXArray* res, VMLXArray x, int dst, VMLXGroup g, VMLXStream s);
int vmlx_recv_like(VMLXArray* res, VMLXArray like, int src, VMLXGroup g, VMLXStream s);
int vmlx_sum_scatter(VMLXArray* res, VMLXArray x, VMLXGroup g, VMLXStream s);

#ifdef __cplusplus
}
#endif

#endif
