import Foundation
import MLX
import MLXLMCommon

@main
struct TestSlice {
    static func main() {
        let a = MLXArray.zeros([1, 23, 10])
        let b = a[0..., (-3)...]
        let c = a[0..., (-3)..., 0...]
        print("b.shape = \(b.shape)")
        print("c.shape = \(c.shape)")
    }
}
