# Native Python NN example. Direct solvers and optimization are implemented here.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.6_classical_machine_learning/linear_regression.py
"""Fourier-feature regression: Cholesky, QR, SGD and an ill-conditioned design."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def least_squares_cholesky(x, y):
    lower = nn.cholesky(nn.matmul(x, x, True, False))
    rhs = nn.matmul(x, y, True, False)
    z = nn.triangular_solve(lower, rhs)
    return nn.triangular_solve(lower, z, True, True)


def least_squares_qr(x, y):
    factor = nn.qr(x)
    return nn.triangular_solve(factor.r, nn.matmul(factor.q, y, True, False), False)


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    n, truth = 64, [0.7, -1.2, 2.5]
    design, target = [], []
    for i in range(n):
        t = 2 * math.pi * i / n
        row = [1.0, math.sqrt(2) * math.cos(t), math.sqrt(2) * math.sin(t)]
        design.extend(row)
        value = 0.1 * math.cos(2 * t)
        for a, b in zip(row, truth):
            value += a * b
        target.append(value)
    x, y = nn.from_vector_double(design, [n, 3]), nn.from_vector_double(target, [n, 1])
    chol, qr = least_squares_cholesky(x, y), least_squares_qr(x, y)
    residual = nn.sub(nn.matmul(x, qr), y)
    normal = nn.matmul(x, residual, True, False)
    assert max(abs(a - b) for a, b in zip(chol.values(), truth)) < 1e-12
    assert max(abs(a - b) for a, b in zip(qr.values(), truth)) < 1e-12
    assert max(abs(v) for v in normal.values()) < 1e-11
    print("Cholesky / QR coefficients:", list(chol.values()), list(qr.values()))

    xf, yf = nn.cast(x, nn.DType.Float32), nn.cast(y, nn.DType.Float32)
    w = nn.zeros([3, 1]).requires_grad_()
    for epoch in range(1, args["epochs"] + 1):
        loss = nn.mul(nn.mean(nn.square(nn.sub(nn.matmul(xf, w), yf)), [0, 1]), 0.5)
        gradient = nn.grad(loss, [w])[0]
        with nn.no_grad():
            w.sub_(gradient, 0.1)
        print(f"epoch {epoch}: pre-update half-MSE={loss.item():.9g}")
    mse = nn.mean(nn.square(nn.sub(nn.matmul(xf, w), yf))).item()
    values = list(w.values())
    expected = [v * (1 - 0.9 ** args["epochs"]) for v in truth]
    assert max(abs(a - b) for a, b in zip(values, expected)) < 2e-6
    assert (
        abs(mse - (0.005 + sum(v * v for v in truth) * 0.9 ** (2 * args["epochs"])))
        < 3e-6
    )
    test_mse = 0.0
    for i in range(n):
        t = 2 * math.pi * (i + 0.5) / n
        row = [1.0, math.sqrt(2) * math.cos(t), math.sqrt(2) * math.sin(t)]
        error = -0.1 * math.cos(2 * t)
        for a, b, c in zip(row, values, truth):
            error += a * (b - c)
        test_mse += error * error / n
    assert (
        abs(test_mse - (0.005 + sum((a - b) ** 2 for a, b in zip(values, truth))))
        < 1e-12
    )
    print(f'{args["epochs"]} SGD steps: train MSE={mse:.9g}; test MSE={test_mse:.9g}')

    collinear, near_target = [], []
    for i in range(n):
        a, b = design[3 * i + 1 : 3 * i + 3]
        last = a + 1e-5 * b
        collinear.extend([1.0, a, last])
        near_target.append(1 + 2 * a - 2 * last)
    cx, cy = nn.from_vector_double(collinear, [n, 3]), nn.from_vector_double(
        near_target, [n, 1]
    )
    ch, q = least_squares_cholesky(cx, cy), least_squares_qr(cx, cy)
    error_chol = max(abs(a - b) for a, b in zip(ch.values(), [1, 2, -2]))
    error_qr = max(abs(a - b) for a, b in zip(q.values(), [1, 2, -2]))
    assert error_qr < 1e-8 and error_chol < 0.01
    print(
        f"Collinear max coefficient error: Cholesky={error_chol:.9g}, QR={error_qr:.9g}"
    )
    print("All linear regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
