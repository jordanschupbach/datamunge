# Native Python NN example. Legendre features, QR fits and diagnostics run here.
"""Polynomial regression and the finite-ensemble bias/variance identity."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


def polynomial_features(x, degree):
    if degree < 0:
        raise ValueError("polynomial degree must be nonnegative")
    columns = [nn.ones_like(x)]
    if degree:
        columns.append(x)
    for n in range(2, degree + 1):
        columns.append(
            nn.div(
                nn.sub(
                    nn.mul(nn.mul(x, columns[n - 1]), 2 * n - 1),
                    nn.mul(columns[n - 2], n - 1),
                ),
                n,
            )
        )
    return nn.concat(columns, 1)


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    n, replicas, grid, sigma = 20, 400, 201, 0.3
    rng = nn.Mt19937(113)
    xs = [-1 + 2.0 * i / (n - 1) for i in range(n)]
    qs = [-1 + 2.0 * i / (grid - 1) for i in range(grid)]
    ys = [
        math.sin(math.pi * x) + rng.normal64(0, sigma)
        for x in xs
        for _ in range(replicas)
    ]
    x, y = nn.from_vector_double(xs, [n, 1]), nn.from_vector_double(ys, [n, replicas])
    query = nn.from_vector_double(qs, [grid, 1])
    directory = output_root(args["full"]) / "metrics/polynomial_regression"
    directory.mkdir(parents=True, exist_ok=True)
    errors = []
    biases = []
    variances = []
    risks = []
    with (directory / "curves.tsv").open("w") as curves, (
        directory / "summary.tsv"
    ).open("w") as summary:
        curves.write("degree\tx\ttruth\tmean\tstddev\n")
        summary.write(
            "degree\ttrain_mse\tbias_squared\tvariance\tnoise_variance\texpected_test_mse\n"
        )
        for degree in [1, 3, 15]:
            design, query_design = polynomial_features(x, degree), polynomial_features(
                query, degree
            )
            factor = nn.qr(design)
            weights = nn.triangular_solve(
                factor.r, nn.matmul(factor.q, y, True, False), False
            )
            prediction = nn.matmul(query_design, weights).values()
            fit = nn.matmul(design, weights).values()
            train_error = sum((a - b) ** 2 for a, b in zip(fit, ys)) / (n * replicas)
            bias = variance = direct_mse = 0.0
            for i in range(grid):
                truth = math.sin(math.pi * qs[i])
                row = prediction[i * replicas : (i + 1) * replicas]
                mean = sum(v / replicas for v in row)
                var = sum((v - mean) ** 2 / replicas for v in row)
                direct_mse += sum((v - truth) ** 2 / (replicas * grid) for v in row)
                bias += (mean - truth) ** 2 / grid
                variance += var / grid
                curves.write(
                    f"{degree}\t{qs[i]:.17g}\t{truth:.17g}\t{mean:.17g}\t{math.sqrt(var):.17g}\n"
                )
            assert abs(direct_mse - bias - variance) < 1e-11
            risk = direct_mse + sigma * sigma
            summary.write(
                f"{degree}\t{train_error:.17g}\t{bias:.17g}\t{variance:.17g}\t{sigma*sigma:.17g}\t{risk:.17g}\n"
            )
            errors.append(train_error)
            biases.append(bias)
            variances.append(variance)
            risks.append(risk)
            print(
                f"degree {degree}: train MSE={train_error:.9g}; bias^2={bias:.9g}; variance={variance:.9g}; noisy-test risk={risk:.9g}"
            )
    assert errors[0] > errors[1] > errors[2]
    assert biases[0] > biases[1] * 10 and variances[2] > variances[1] * 5
    assert risks[1] < risks[0] and risks[1] < risks[2]
    print("All polynomial regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
