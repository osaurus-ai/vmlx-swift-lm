import Foundation
import MLX
import MLXNN

let x = MLXArray(1.0, dtype: .float16)
let scale = Float(2.0)
let scalarArray = MLXArray(scale)
let y = x * scalarArray
print("x.dtype = \(x.dtype)")
print("scalarArray.dtype = \(scalarArray.dtype)")
print("y.dtype = \(y.dtype)")
