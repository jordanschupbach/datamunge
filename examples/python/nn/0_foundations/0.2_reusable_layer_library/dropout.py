# Native Python NN example. Inverted dropout, regularization and Monte Carlo uncertainty.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/dropout.py
"""Train explicit MLPs with/without dropout and measure stochastic predictions."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def statistics():
    generator = nn.Generator(7)
    n = 100000
    x = nn.ones([1, n])
    for p in [0, 0.1, 0.25, 0.5, 0.8]:
        p = nn.scalar(p).item()
        keep = nn.scalar(1 - p).item()
        values = list(nn.dropout(x, p, True, generator).values())
        zero = sum(v == 0 for v in values) / n
        mean = sum(values) / n
        variance = sum(v * v for v in values) / n - mean * mean
        assert (
            abs(zero - p) < 0.006
            and abs(mean - 1) < 0.02
            and abs(variance - p / keep) < 0.03 * (1 + p / keep)
        )
        assert abs(max(values) - 1 / keep) < 1e-5
        print(
            f"p={p:.6g}: zero fraction={zero:.9g}, mean={mean:.9g}, variance={variance:.9g}, expected variance={p/keep:.9g}, survivor={max(values):.9g}"
        )
    rng = nn.Mt19937_64(0)
    random = nn.from_vector([rng.normal32() for _ in range(n)], [1, n])
    assert list(nn.dropout(random, 0.5, False, generator).values()) == list(
        random.values()
    )
    print("Inference dropout is exactly the identity for all", n, "entries.")


def train_mlp(p, seed, x_train, y_train, x_test, y_test, steps, mc=False):
    parameters = nn.ParameterStore(seed)
    drop1, drop2 = nn.Generator(seed * 10 + 1), nn.Generator(seed * 10 + 2)
    hidden = nn.DenseOptions()
    hidden.activation = nn.Activation.Relu
    hidden.init = nn.init.he_uniform()

    def forward(x, training):
        h = nn.dense(parameters, "h1", x, 256, hidden)
        h = nn.dropout(h, p, training, drop1)
        h = nn.dense(parameters, "h2", h, 256, hidden)
        h = nn.dropout(h, p, training, drop2)
        return nn.dense(parameters, "out", h, 1)

    x, y = nn.from_vector(x_train, [len(x_train), 1]), nn.from_vector(
        y_train, [len(y_train), 1]
    )
    xt, yt = nn.from_vector(x_test, [len(x_test), 1]), nn.from_vector(
        y_test, [len(y_test), 1]
    )
    with nn.no_grad():
        forward(x, False)
    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    adam = nn.Optimizer(parameters, 1e-3, cfg)
    for _ in range(steps):
        adam.minimize(nn.mse(forward(x, True), y))
    assert adam.iterations() == steps
    with nn.no_grad():
        train = nn.mse(forward(x, False), y).item()
        test = nn.mse(forward(xt, False), yt).item()
        std = []
        assert math.isfinite(train) and math.isfinite(test)
        if mc:
            samples = 100
            total = [0.0] * len(x_test)
            square = [0.0] * len(x_test)
            for _ in range(samples):
                values = list(forward(xt, True).values())
                for i, v in enumerate(values):
                    total[i] += v
                    square[i] += v * v
            std = [
                math.sqrt(max(0, square[i] / samples - (total[i] / samples) ** 2))
                for i in range(len(x_test))
            ]
    return train, test, std


def regularization(steps):
    rng = nn.Mt19937_64(42)
    x_train = []
    y_train = []
    for _ in range(40):
        x = rng.uniform32(-1, 1)
        value = nn.sin(nn.mul(nn.scalar(x), 3)).item()
        y = nn.scalar(value + rng.normal32(0, 0.3)).item()
        x_train.append(x)
        y_train.append(y)
    x_test = []
    y_test = []
    for i in range(201):
        x = nn.add(nn.div(nn.mul(nn.scalar(i), 3), 200), -1.5).item()
        x_test.append(x)
        y_test.append(nn.sin(nn.mul(nn.scalar(x), 3)).item())
    inside = [i for i, x in enumerate(x_test) if abs(x) <= 1]
    xin, yin = [x_test[i] for i in inside], [y_test[i] for i in inside]
    means = []
    print(f"MLP 1-256-256-1, 40 noisy sin(3x) points; {steps} Adam updates per fit.")
    for p in [0, 0.5]:
        train_sum = test_sum = 0
        for seed in range(1, 4):
            train, test, _ = train_mlp(p, seed, x_train, y_train, xin, yin, steps)
            train_sum += train / 3
            test_sum += test / 3
            print(
                f"p={p}, seed={seed}: train MSE={train:.9g}, in-range test MSE={test:.9g}"
            )
        means.append((train_sum, test_sum))
        print("p=", p, "three-seed mean train/test:", train_sum, test_sum)
    _, _, std = train_mlp(0.5, 1, x_train, y_train, x_test, y_test, steps, True)
    assert all(math.isfinite(v) for v in std)
    print("Monte Carlo dropout, 100 stochastic forward passes:")
    for i in range(0, 201, 20):
        print(f"x={x_test[i]:.6g}, predictive std={std[i]:.9g}")
    if steps >= 3000:
        assert (
            means[0][0] < means[1][0]
            and means[1][1] < means[0][1]
            and means[1][1] < 0.1
        )
    assert std[0] > std[100] and std[200] > std[100]


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    nn.set_num_threads(2)
    statistics()
    regularization(args["epochs"])
    print("All dropout checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
