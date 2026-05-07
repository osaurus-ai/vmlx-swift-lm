#include "CmlxGraphShim.h"

#include <exception>
#include <cstdint>
#include <sstream>
#include <string>

#include "mlx/c/array.h"
#include "mlx/c/error.h"
#include "mlx/c/private/array.h"
#include "mlx/graph_utils.h"

static inline mlx_array vmlx_graph_arr(VMLXGraphArray a) {
    mlx_array out;
    out.ctx = a;
    return out;
}

static int count_substring(const std::string& haystack, const std::string& needle) {
    if (needle.empty()) { return 0; }
    int count = 0;
    std::size_t pos = 0;
    while ((pos = haystack.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

extern "C" int vmlx_graph_stats(
    VMLXGraphArray array,
    int32_t* node_count,
    int32_t* astype_count
) {
    if (node_count) { *node_count = -1; }
    if (astype_count) { *astype_count = -1; }
    try {
        std::ostringstream dot;
        mlx::core::export_to_dot(dot, mlx_array_get_(vmlx_graph_arr(array)));
        const std::string text = dot.str();
        if (node_count) {
            *node_count = static_cast<int32_t>(
                count_substring(text, "[label=") + count_substring(text, "[label ="));
        }
        if (astype_count) {
            *astype_count = static_cast<int32_t>(count_substring(text, "AsType"));
        }
        return 0;
    } catch (const std::exception& e) {
        mlx_error(e.what());
        return 1;
    }
}
