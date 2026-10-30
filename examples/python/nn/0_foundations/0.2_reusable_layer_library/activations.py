# Native Python NN example. Activation values, derivatives, statistics and deep stacks.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/activations.py
"""Compare thirteen nonlinearities with independent closed-form expressions."""
import math
from pydatamunge import nn

LAMBDA = 1.0507009873554804934193349852946
ALPHA = 1.6732632423543772848170429916717
LEAKY = nn.scalar(0.2).item()
NAMES = [
    "Relu",
    "Relu6",
    "LeakyRelu",
    "Elu",
    "Selu",
    "Gelu",
    "GeluTanh",
    "Silu",
    "Mish",
    "Softplus",
    "Softsign",
    "Sigmoid",
    "Tanh",
]


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softplus(x):
    return x if x > 30 else math.log1p(math.exp(x))


def cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def value_and_derivative(name, x):
    if name == "Relu":
        return max(x, 0), float(x > 0)
    if name == "Relu6":
        return min(max(x, 0), 6), float(0 < x < 6)
    if name == "LeakyRelu":
        return (x if x > 0 else LEAKY * x), (1 if x > 0 else LEAKY)
    if name == "Elu":
        return (x if x > 0 else math.expm1(x)), (1 if x > 0 else math.exp(x))
    if name == "Selu":
        return LAMBDA * (x if x > 0 else ALPHA * math.expm1(x)), LAMBDA * (
            1 if x >= 0 else ALPHA * math.exp(x)
        )
    if name == "Gelu":
        return x * cdf(x), cdf(x) + x * math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    if name == "GeluTanh":
        u = math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x)
        du = math.sqrt(2 / math.pi) * (1 + 3 * 0.044715 * x * x)
        t = math.tanh(u)
        return 0.5 * x * (1 + t), 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du
    if name == "Silu":
        s = sigmoid(x)
        return x * s, s * (1 + x * (1 - s))
    if name == "Mish":
        t = math.tanh(softplus(x))
        return x * t, t + x * (1 - t * t) * sigmoid(x)
    if name == "Softplus":
        return softplus(x), sigmoid(x)
    if name == "Softsign":
        return x / (1 + abs(x)), 1 / (1 + abs(x)) ** 2
    if name == "Sigmoid":
        s = sigmoid(x)
        return s, s * (1 - s)
    if name == "Tanh":
        t = math.tanh(x)
        return t, 1 - t * t
    raise ValueError("unknown activation " + name)


def table():
    grid = [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]
    print("Activation grid:", grid)
    for name in NAMES:
        x = nn.from_vector(grid).requires_grad_()
        y = nn.activate(x, getattr(nn.Activation, name), LEAKY)
        dy = nn.grad(nn.sum(y), [x])[0]
        expected = [value_and_derivative(name, v) for v in grid]
        value_error = max(abs(a - e[0]) for a, e in zip(y.values(), expected))
        gradient_error = max(abs(a - e[1]) for a, e in zip(dy.values(), expected))
        assert value_error < 2e-6 and gradient_error < 2e-6
        print(
            name,
            "values:",
            list(y.values()),
            "derivatives:",
            list(dy.values()),
            "max errors:",
            value_error,
            gradient_error,
        )


def properties():
    rng = nn.Mt19937_64(1)
    samples = [rng.normal64() for _ in range(200000)]
    print(
        "Properties on 200000 standard-normal samples: min, max, mean, std, dead fraction, max derivative"
    )
    for name in NAMES:
        low, high, total, square, dead, max_derivative = 1e9, -1e9, 0.0, 0.0, 0, 0.0
        for x in samples:
            value, derivative = value_and_derivative(name, x)
            low = min(low, value)
            high = max(high, value)
            total += value
            square += value * value
            dead += derivative == 0
            max_derivative = max(max_derivative, derivative)
        mean = total / len(samples)
        std = math.sqrt(square / len(samples) - mean * mean)
        fraction = dead / len(samples)
        if name == "Selu":
            assert abs(mean) < 0.02 and abs(std - 1) < 0.02
        if name == "Relu":
            assert abs(fraction - 0.5) < 0.01
        print(name, low, high, mean, std, fraction, max_derivative)


def selu_depth():
    depth, width, batch = 30, 256, 256
    rng = nn.Mt19937_64(2)
    inputs = nn.from_vector(
        [rng.normal32() for _ in range(batch * width)], [batch, width]
    )
    with nn.no_grad():
        for name in ["Selu", "Elu", "Tanh", "Relu"]:
            parameters = nn.ParameterStore(5)
            h = inputs
            means = []
            stds = []
            for layer in range(depth):
                option = nn.DenseOptions()
                option.init = nn.init.lecun_normal()
                option.bias = False
                option.activation = getattr(nn.Activation, name)
                h = nn.dense(parameters, "l" + str(layer), h, width, option)
                double = nn.cast(h, nn.DType.Float64)
                mean = nn.mean(double).item()
                means.append(mean)
                stds.append(math.sqrt(nn.mean(nn.square(double)).item() - mean * mean))
            if name == "Selu":
                assert all(abs(v) < 0.1 for v in means) and all(
                    0.85 < v < 1.15 for v in stds
                )
            else:
                assert stds[-1] < 0.25
            print(
                name,
                "layer, mean, std:",
                [(i, means[i - 1], stds[i - 1]) for i in [1, 2, 5, 10, 20, 30]],
            )


def main(arguments=None):
    nn.set_num_threads(2)
    table()
    properties()
    selu_depth()
    print("All activation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
