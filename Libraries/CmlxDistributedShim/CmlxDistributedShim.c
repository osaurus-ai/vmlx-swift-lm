// Phase 5 C bridge — see CmlxDistributedShim.h.
//
// Wraps mlx-c's distributed collectives so the Swift side can use
// opaque void* handles without worrying about by-value struct passing
// or differences in how SwiftPM exposes Cmlx symbols.

#include "CmlxDistributedShim.h"

#include "mlx/c/array.h"
#include "mlx/c/distributed.h"
#include "mlx/c/distributed_group.h"
#include "mlx/c/stream.h"

// Helpers: convert between our opaque pointers and the underlying
// single-pointer mlx_* struct types.
static inline mlx_array vmlx_arr(VMLXArray a) {
    mlx_array out;
    out.ctx = a;
    return out;
}
static inline mlx_distributed_group vmlx_grp(VMLXGroup g) {
    mlx_distributed_group out;
    out.ctx = g;
    return out;
}
static inline mlx_stream vmlx_stm(VMLXStream s) {
    mlx_stream out;
    out.ctx = s;
    return out;
}

void vmlx_array_free(VMLXArray a) {
    mlx_array_free(vmlx_arr(a));
}

int vmlx_group_rank(VMLXGroup g) {
    return mlx_distributed_group_rank(vmlx_grp(g));
}
int vmlx_group_size(VMLXGroup g) {
    return mlx_distributed_group_size(vmlx_grp(g));
}
VMLXGroup vmlx_group_init(bool strict, const char* backend) {
    mlx_distributed_group g = mlx_distributed_init(strict, backend);
    return g.ctx;
}
VMLXGroup vmlx_group_split(VMLXGroup g, int color, int key) {
    mlx_distributed_group out = mlx_distributed_group_split(vmlx_grp(g), color, key);
    return out.ctx;
}
bool vmlx_distributed_is_available(const char* backend) {
    return mlx_distributed_is_available(backend);
}

int vmlx_all_sum(VMLXArray* res, VMLXArray x, VMLXGroup g, VMLXStream s) {
    mlx_array out;
    out.ctx = NULL;
    int rc = mlx_distributed_all_sum(&out, vmlx_arr(x), vmlx_grp(g), vmlx_stm(s));
    *res = out.ctx;
    return rc;
}
int vmlx_all_gather(VMLXArray* res, VMLXArray x, VMLXGroup g, VMLXStream s) {
    mlx_array out;
    out.ctx = NULL;
    int rc = mlx_distributed_all_gather(&out, vmlx_arr(x), vmlx_grp(g), vmlx_stm(s));
    *res = out.ctx;
    return rc;
}
int vmlx_send(VMLXArray* res, VMLXArray x, int dst, VMLXGroup g, VMLXStream s) {
    mlx_array out;
    out.ctx = NULL;
    int rc = mlx_distributed_send(&out, vmlx_arr(x), dst, vmlx_grp(g), vmlx_stm(s));
    *res = out.ctx;
    return rc;
}
int vmlx_recv_like(VMLXArray* res, VMLXArray like, int src, VMLXGroup g, VMLXStream s) {
    mlx_array out;
    out.ctx = NULL;
    int rc = mlx_distributed_recv_like(&out, vmlx_arr(like), src, vmlx_grp(g), vmlx_stm(s));
    *res = out.ctx;
    return rc;
}
int vmlx_sum_scatter(VMLXArray* res, VMLXArray x, VMLXGroup g, VMLXStream s) {
    mlx_array out;
    out.ctx = NULL;
    int rc = mlx_distributed_sum_scatter(&out, vmlx_arr(x), vmlx_grp(g), vmlx_stm(s));
    *res = out.ctx;
    return rc;
}
