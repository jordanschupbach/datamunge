"""Shared catalog and independent NumPy references for matched CPU workloads."""
from pathlib import Path
import re

CATALOG = Path(__file__).resolve().parents[1] / 'benchmarks/tensorflow_cases.def'
CASES = {}
for line in CATALOG.read_text().splitlines():
    match = re.fullmatch(r'(BASE|EXTRA)\((\w+), ([\d, ]+)\)', line.strip())
    if match:
        kind, name, values = match.groups()
        sizes = tuple(map(int, values.split(',')))
        if name in CASES:
            raise RuntimeError(f'Duplicate comparison operation: {name}')
        CASES[name] = sizes[1:] if kind == 'BASE' else sizes
OPERATIONS = tuple(CASES)


def fixture(op, rows, f64, np, tf):
    dtype = np.float64 if f64 else np.float32
    def array(shape):
        return ((np.arange(np.prod(shape), dtype=np.int64) % 251 - 125) / 128.0).reshape(shape).astype(dtype)
    shape = (rows, rows) if op in ('Matmul', 'MatmulTranspose', 'Solve', 'Cholesky') else (rows, 128)
    if op in ('Conv2d', 'ConvInputGradient', 'MaxPool', 'AvgPool'): shape = (rows, 16, 16, 4)
    if op == 'Attention': shape = (2, rows, 32)
    if op == 'BatchMatmul': shape = (4, rows, rows)
    if op == 'FftPower': shape = (8, rows)
    if op == 'MatmulGradient': shape = (rows, 64)
    a = array(shape)
    b = array((shape[-1],))
    if op in ('Conv2d', 'ConvInputGradient'): b = array((3, 3, 4, 8))
    if op == 'MatmulGradient': b = array((64, 32))
    if op == 'GatherRows': b = np.arange(rows // 4, dtype=np.int64) * 3
    if op in ('Solve', 'Cholesky'):
        a = (a.astype(np.float64) @ a.astype(np.float64).T + rows * np.eye(rows)).astype(dtype)
        b = array((rows, 4))
    x, w = tf.constant(a), tf.constant(b)
    ref = a.astype(np.float64)
    weight = b.astype(np.float64)
    if op in ('Add', 'SquareGradient'): expected = 2 * ref
    elif op == 'BroadcastAdd': expected = ref + weight
    elif op == 'Multiply': expected = ref * ref
    elif op == 'SumLast': expected = ref.sum(axis=-1)
    elif op == 'SumFirst': expected = ref.sum(axis=0)
    elif op == 'SumAll': expected = ref.sum()
    elif op == 'Mean': expected = ref.mean()
    elif op == 'MaxLast': expected = ref.max(axis=-1)
    elif op == 'Variance': expected = ref.var(axis=-1)
    elif op == 'Relu': expected = np.maximum(ref, 0)
    elif op == 'Exp': expected = np.exp(ref)
    elif op == 'Sigmoid': expected = 1 / (1 + np.exp(-ref))
    elif op == 'Tanh': expected = np.tanh(ref)
    elif op == 'Gelu':
        import math
        expected = .5 * ref * (1 + np.vectorize(math.erf)(ref / np.sqrt(2)))
    elif op == 'GatherRows': expected = ref[np.arange(rows // 4) * 3]
    elif op == 'TransposeAdd': expected = 2 * ref.T
    elif op in ('Matmul', 'BatchMatmul'): expected = ref @ ref
    elif op == 'MatmulTranspose': expected = ref.T @ ref
    elif op == 'MatmulGradient': expected = ref.T @ (2 * (ref @ weight))
    elif op in ('Softmax', 'LogSoftmax', 'LogSumExp', 'SoftmaxGradient', 'Attention'):
        logits = ref @ ref.swapaxes(-1, -2) / np.sqrt(32) if op == 'Attention' else ref
        largest = logits.max(axis=-1, keepdims=True)
        shifted = logits - largest
        exp = np.exp(shifted)
        total = exp.sum(axis=-1, keepdims=True)
        softmax = exp / total
        if op == 'Softmax': expected = softmax
        elif op == 'LogSoftmax': expected = shifted - np.log(total)
        elif op == 'LogSumExp': expected = (largest + np.log(total)).squeeze(-1)
        elif op == 'Attention': expected = softmax @ ref
        else: expected = 2 * softmax * (softmax - (softmax * softmax).sum(axis=-1, keepdims=True))
    elif op in ('LayerNorm', 'BatchNorm'):
        axis = -1 if op == 'LayerNorm' else 0
        expected = (ref - ref.mean(axis=axis, keepdims=True)) / np.sqrt(ref.var(axis=axis, keepdims=True) + 1e-5)
    elif op == 'FftPower': expected = np.abs(np.fft.rfft(ref)) ** 2
    elif op == 'Solve': expected = np.linalg.solve(ref, weight)
    elif op == 'Cholesky': expected = np.linalg.cholesky(ref)
    elif op in ('MaxPool', 'AvgPool'):
        windows = ref.reshape(rows, 8, 2, 8, 2, 4)
        expected = windows.max(axis=(2, 4)) if op == 'MaxPool' else windows.mean(axis=(2, 4))
    elif op in ('Conv2d', 'ConvInputGradient'):
        if op == 'Conv2d':
            expected = np.zeros((rows, 14, 14, 8))
            for dy in range(3):
                for dx in range(3): expected += ref[:, dy:dy+14, dx:dx+14] @ weight[dy, dx]
        else:
            expected = np.zeros_like(ref)
            for dy in range(3):
                for dx in range(3): expected[:, dy:dy+14, dx:dx+14] += weight[dy, dx].sum(axis=-1)
    else: raise RuntimeError(f'No independent reference for {op}')

    def operation(a, b):
        if op == 'Add': return tf.add(a, a)
        if op == 'BroadcastAdd': return tf.add(a, b)
        if op == 'Multiply': return tf.multiply(a, a)
        if op == 'SumLast': return tf.reduce_sum(a, axis=-1)
        if op == 'SumFirst': return tf.reduce_sum(a, axis=0)
        if op == 'SumAll': return tf.reduce_sum(a)
        if op == 'Mean': return tf.reduce_mean(a)
        if op == 'MaxLast': return tf.reduce_max(a, axis=-1)
        if op == 'Variance': return tf.math.reduce_variance(a, axis=-1)
        if op == 'LogSumExp': return tf.reduce_logsumexp(a, axis=-1)
        if op == 'Relu': return tf.nn.relu(a)
        if op == 'Exp': return tf.exp(a)
        if op == 'Sigmoid': return tf.math.sigmoid(a)
        if op == 'Tanh': return tf.math.tanh(a)
        if op == 'Gelu': return tf.nn.gelu(a, approximate=False)
        if op == 'GatherRows': return tf.gather(a, b)
        if op == 'TransposeAdd':
            t = tf.transpose(a)
            return t + t
        if op == 'Softmax': return tf.nn.softmax(a)
        if op == 'LogSoftmax': return tf.nn.log_softmax(a)
        if op in ('Matmul', 'BatchMatmul'): return tf.matmul(a, a)
        if op == 'MatmulTranspose': return tf.matmul(a, a, transpose_a=True)
        if op == 'Conv2d': return tf.nn.conv2d(a, b, 1, 'VALID')
        if op == 'MaxPool': return tf.nn.max_pool2d(a, 2, 2, 'VALID')
        if op == 'AvgPool': return tf.nn.avg_pool2d(a, 2, 2, 'VALID')
        if op in ('LayerNorm', 'BatchNorm'):
            mean, variance = tf.nn.moments(a, axes=[1] if op == 'LayerNorm' else [0], keepdims=True)
            return (a - mean) * tf.math.rsqrt(variance + tf.cast(1e-5, a.dtype))
        if op == 'Attention':
            weights = tf.nn.softmax(tf.matmul(a, a, transpose_b=True) / tf.sqrt(tf.cast(32, a.dtype)))
            return tf.matmul(weights, a)
        if op == 'FftPower':
            z = tf.signal.rfft(a)
            return tf.square(tf.math.real(z)) + tf.square(tf.math.imag(z))
        if op == 'Solve': return tf.linalg.solve(a, b)
        if op == 'Cholesky': return tf.linalg.cholesky(a)
        with tf.GradientTape() as tape:
            variable = b if op == 'MatmulGradient' else a
            tape.watch(variable)
            if op == 'MatmulGradient': loss = tf.reduce_sum(tf.square(tf.matmul(a, b)))
            elif op == 'ConvInputGradient': loss = tf.reduce_sum(tf.nn.conv2d(a, b, 1, 'VALID'))
            elif op == 'SoftmaxGradient': loss = tf.reduce_sum(tf.square(tf.nn.softmax(a)))
            else: loss = tf.reduce_sum(a * a)
        return tape.gradient(loss, variable)
    tolerance = (1e-8 if op == 'Gelu' else 1e-10) if f64 else 2e-4 if op in ('MatmulGradient', 'FftPower') else 2e-5
    return (x, w), operation, expected, tolerance
