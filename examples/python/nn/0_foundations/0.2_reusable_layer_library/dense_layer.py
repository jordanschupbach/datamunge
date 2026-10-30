# Native Python NN example. Dense forward passes, sharing, initializer scales and gradients.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/dense_layer.py
"""Check the bound dense layer against independent arithmetic and derivatives."""
import math
from pydatamunge import nn


def forward_pass():
    p = nn.ParameterStore(1)
    linear = nn.DenseOptions()
    no_bias = nn.DenseOptions()
    no_bias.bias = False
    relu = nn.DenseOptions()
    relu.activation = nn.Activation.Relu
    relu.bias_value = 0.1
    x = nn.from_vector([1, 2, 3, 0, -1, 1, 2, 2, 2, -1, 0, 1], [4, 3])
    outputs = [
        nn.dense(p, "lin", x, 2, linear),
        nn.dense(p, "nob", x, 2, no_bias),
        nn.dense(p, "relu", x, 2, relu),
    ]
    for label, tensor in [("x", x), ("W (lin)", p.at("lin/W"))] + list(
        zip(["linear", "no bias", "relu"], outputs)
    ):
        print(label, tensor.str())
    w, b = p.at("relu/W"), p.at("relu/b")
    error = 0
    for i in range(4):
        for j in range(2):
            expected = max(
                0, b.at([j]) + sum(x.at([i, k]) * w.at([k, j]) for k in range(3))
            )
            error = max(error, abs(expected - outputs[2].at([i, j])))
    assert error < 1e-5 and p.count() == 22 and abs(b.at([0]) - 0.1) < 1e-7
    print("Dense forward max hand-computation error:", error, "parameters:", p.count())


def sharing_and_sequences():
    p = nn.ParameterStore(2)
    same = nn.from_vector([1, 2, 3, 0, -1, 1], [2, 3])
    values = nn.sub(nn.mul(nn.from_vector([i % 7 for i in range(30)]), 0.1), 0.3)
    sequence = nn.reshape(values, [2, 5, 3])
    a = nn.dense(p, "shared", same, 2)
    b = nn.dense(p, "shared", same.clone(), 2)
    output = nn.dense(p, "shared", sequence, 2)
    assert list(a.values()) == list(b.values()) and p.count() == 8 and p.size() == 2
    assert tuple(output.shape()) == (2, 5, 2)
    row = nn.reshape(nn.select(nn.select(sequence, 0, 1), 0, 3), [1, 3])
    expected = nn.select(nn.select(output, 0, 1), 0, 3)
    error = max(
        abs(v) for v in nn.sub(nn.dense(p, "shared", row, 2), expected).values()
    )
    assert error < 1e-6
    print(
        "Three shared applications use eight parameters; sequence shape:",
        tuple(output.shape()),
        "row error:",
        error,
    )


def initializer_scales():
    cases = [
        ("glorot_uniform + tanh", nn.init.glorot_uniform(), nn.Activation.Tanh),
        ("glorot_uniform + relu", nn.init.glorot_uniform(), nn.Activation.Relu),
        ("he_normal + relu", nn.init.he_normal(), nn.Activation.Relu),
        ("lecun_normal + selu", nn.init.lecun_normal(), nn.Activation.Selu),
        ("lecun_normal + relu", nn.init.lecun_normal(), nn.Activation.Relu),
    ]
    depth, width, batch = 24, 256, 512
    rng = nn.Mt19937_64(0)
    inputs = nn.from_vector(
        [rng.normal32() for _ in range(batch * width)], [batch, width]
    )
    deviations = []
    with nn.no_grad():
        for label, initializer, activation in cases:
            p = nn.ParameterStore(3)
            h = inputs
            row = []
            for layer in range(depth):
                option = nn.DenseOptions()
                option.init = initializer
                option.bias = False
                z = nn.dense(p, "l" + str(layer), h, width, option)
                h = nn.activate(z, activation)
                row.append(
                    math.sqrt(nn.mean(nn.square(nn.cast(z, nn.DType.Float64))).item())
                )
            deviations.append(row)
            print(
                label,
                "pre-activation RMS at layers 1,4,8,12,16,20,24:",
                [row[i - 1] for i in [1, 4, 8, 12, 16, 20, 24]],
            )
    assert abs(deviations[0][0] - 1) < 0.1 and 0.08 < deviations[0][-1] < 0.3
    assert deviations[1][-1] < 0.01 and 0.6 < deviations[2][-1] < 2.5
    assert all(0.85 < v < 1.15 for v in deviations[3]) and deviations[4][-1] < 0.01


def gradients():
    p = nn.ParameterStore(4)
    x = nn.from_vector([1, 2, 3, 0, -1, 1, 2, 2, 2, -1, 0, 1], [4, 3])
    target = nn.from_vector([1, 0, 0, 1, 1, 1, 0, 0], [4, 2])

    def loss():
        y = nn.dense(p, "d", x, 2)
        return nn.mean(nn.square(nn.sub(y, target)), [0, 1]), y

    value, y = loss()
    w, b = p.at("d/W"), p.at("d/b")
    gw, gb = nn.grad(value, [w, b])
    error = 0
    for k in range(3):
        for j in range(2):
            expected = 0.25 * sum(
                x.at([i, k]) * (y.at([i, j]) - target.at([i, j])) for i in range(4)
            )
            error = max(error, abs(expected - gw.at([k, j])))
    for j in range(2):
        expected = 0.25 * sum(y.at([i, j]) - target.at([i, j]) for i in range(4))
        error = max(error, abs(expected - gb.at([j])))
    finite_error = 0
    eps = nn.scalar(1e-3).item()
    with nn.no_grad():
        for parameter, gradient in [(w, gw), (b, gb)]:
            base = parameter.clone()
            original = list(base.values())
            expected = list(gradient.values())
            flat = nn.reshape(parameter, [-1])
            for i in range(parameter.numel()):
                cell = nn.select(flat, 0, i)
                cell.fill_(original[i] + eps)
                plus = loss()[0].item()
                parameter.copy_(base)
                cell.fill_(original[i] - eps)
                minus = loss()[0].item()
                parameter.copy_(base)
                finite_error = max(
                    finite_error, abs((plus - minus) / (2 * eps) - expected[i])
                )
    gmax = max(abs(v) for v in gw.values())
    assert error < 1e-5 and finite_error / gmax < 5e-3
    print("Loss:", value.item(), "dL/dW:", gw.str(), "dL/db:", gb.str())
    print(
        "Gradient max closed-form error:",
        error,
        "finite-difference error:",
        finite_error,
        "relative:",
        finite_error / gmax,
    )


def main(arguments=None):
    nn.set_num_threads(2)
    forward_pass()
    sharing_and_sequences()
    initializer_scales()
    gradients()
    print("All dense layer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
