import mlx.core as mx

# Emulate the Swift MLX array slicing behavior or check Python
a = mx.zeros((1, 23, 10))
b = a[:, -3:, :]
print(b.shape)
