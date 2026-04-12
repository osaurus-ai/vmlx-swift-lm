import mlx.core as mx
x = mx.array([1.0, 2.0, 3.0], dtype=mx.float16)
y = mx.softmax(x, precise=True)
print("Python softmax dtype:", y.dtype)
