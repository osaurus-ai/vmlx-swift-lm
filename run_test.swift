import MLX
MLX.GPU.set(cacheLimit: 10 * 1024 * 1024)
let a = MLXArray.zeros([1, 23, 10])
let b = a[0..., (-3)..., 0...]
print("b.shape = \(b.shape)")
