# Native Python NN example. PCA and kernel projection are implemented here.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.6_classical_machine_learning/principal_components.py
"""Thin-SVD PCA and centered RBF kernel PCA, including held-out checks."""
import math
from pydatamunge import nn


def maximum_error(tensor):
    return max(abs(v) for v in tensor.values())


def linear():
    n, a, b = 128, 0.6, 0.4
    rotation = [
        [math.cos(a), -math.sin(a) * math.cos(b), math.sin(a) * math.sin(b)],
        [math.sin(a), math.cos(a) * math.cos(b), -math.cos(a) * math.sin(b)],
        [0, math.sin(b), math.cos(b)],
    ]
    offset = [1, -2, 0.5]

    def data(shift):
        values = []
        for i in range(n):
            t = 2 * math.pi * (i + shift) / n
            latent = [
                math.sqrt(18) * math.cos(t),
                math.sqrt(2) * math.sin(t),
                math.sqrt(0.08) * math.cos(2 * t),
            ]
            for d in range(3):
                value = offset[d]
                for j in range(3):
                    value += rotation[d][j] * latent[j]
                values.append(value)
        return nn.from_vector_double(values, [n, 3])

    training, held_out = data(0), data(0.5)
    mean = nn.mean(training, [0], True)
    factor = nn.svd(nn.sub(training, mean))
    components = nn.slice(factor.v, [0, 0], [3, 2])
    variances = nn.div(nn.square(nn.slice(factor.s, [0], [2])), n - 1)

    def transform(x):
        return nn.matmul(nn.sub(x, mean), components)

    def reconstruct(scores):
        return nn.add(mean, nn.matmul(scores, components, False, True))

    scores = transform(training)
    whitened = nn.div(scores, nn.sqrt(variances))
    covariance = nn.div(nn.matmul(whitened, whitened, True, False), n - 1)
    assert max(abs(x - y) for x, y in zip(mean.values(), offset)) < 1e-12
    for c, expected in enumerate([9 * n / (n - 1), n / (n - 1)]):
        dot = sum(components.at([d, c]) * rotation[d][c] for d in range(3))
        assert abs(abs(dot) - 1) < 1e-12
        assert abs(variances.at([c]) - expected) < 1e-11
    assert maximum_error(nn.sub(covariance, nn.eye(2, nn.DType.Float64))) < 1e-12
    errors = []
    for x in (training, held_out):
        error = nn.mean(nn.square(nn.sub(x, reconstruct(transform(x))))).item()
        assert abs(error - 0.04 / 3) < 1e-12
        errors.append(error)
    print(
        "PCA variances:",
        list(variances.values()),
        "explained fraction:",
        10 / 10.04,
        "train / held-out MSE:",
        errors,
    )


def kernel():
    n, retained = 80, 3
    points, queries = [], []
    for i in range(n):
        angle, radius = 2 * math.pi * (i // 2) / (n // 2), 2 if i % 2 else 0.5
        points.extend([radius * math.cos(angle), radius * math.sin(angle)])
    for i in range(6):
        angle, radius = 0.13 + 0.7 * i, 2 if i % 2 else 0.5
        queries.extend([radius * math.cos(angle), radius * math.sin(angle)])
    x, query = nn.from_vector(points, [n, 2]), nn.from_vector(queries, [6, 2])

    def rbf(a, b):
        delta = nn.sub(nn.expand_dims(a, 1), nn.expand_dims(b, 0))
        return nn.exp(nn.mul(nn.sum(nn.square(delta), [2]), -0.5))

    gram = rbf(x, x)
    column_mean, grand_mean = nn.mean(gram, [0], True), nn.mean(gram, [0, 1])

    def center(cross):
        return nn.add(
            nn.sub(nn.sub(cross, column_mean), nn.mean(cross, [1], True)), grand_mean
        )

    centered = center(gram)
    eigen = nn.eigh(centered)
    values = nn.reverse(nn.slice(eigen.e, [n - retained], [retained]), [0])
    vectors = nn.reverse(nn.slice(eigen.v, [0, n - retained], [n, retained]), [1])
    root = nn.expand_dims(nn.sqrt(values), 0)
    scaled, training_scores = nn.div(vectors, root), nn.mul(vectors, root)

    def project(q):
        return nn.matmul(center(rbf(q, x)), scaled)

    assert maximum_error(nn.sum(centered, [1])) < 2e-5
    assert eigen.e.at([0]) > -1e-5 and values.at([retained - 1]) > 0.1
    residual = nn.sub(
        nn.matmul(centered, vectors), nn.mul(vectors, nn.expand_dims(values, 0))
    )
    assert maximum_error(residual) < 2e-5
    assert maximum_error(nn.sub(training_scores, project(x))) < 2e-5
    projected = project(query)
    for i in range(6):
        single = project(nn.slice_axis(query, 0, i, i + 1))
        assert (
            maximum_error(nn.sub(single, nn.slice_axis(projected, 0, i, i + 1))) < 2e-6
        )
    print("Kernel PCA retained eigenvalues:", list(values.values()))
    print(
        "Centered Gram, eigenproblem, training reprojection and six independent query projections passed."
    )


def main(arguments=None):
    linear()
    kernel()
    print("All PCA and kernel-PCA checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
