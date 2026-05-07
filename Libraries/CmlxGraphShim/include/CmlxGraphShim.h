// Lightweight graph inspection bridge for RunBench.

#ifndef VMLX_CMLX_GRAPH_SHIM_H
#define VMLX_CMLX_GRAPH_SHIM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* VMLXGraphArray;

int vmlx_graph_stats(VMLXGraphArray array, int32_t* node_count, int32_t* astype_count);

#ifdef __cplusplus
}
#endif

#endif
