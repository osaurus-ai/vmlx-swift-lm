import MLX
import MLXFast
import Foundation

let source = """
    auto n = thread_position_in_grid.z;
    y[0] = q[0];
"""

let kernel = MLXFast.metalKernel(
    name: "test",
    inputNames: ["q"],
    outputNames: ["y"],
    source: source
)
print("Kernel loaded: \(kernel != nil)")
