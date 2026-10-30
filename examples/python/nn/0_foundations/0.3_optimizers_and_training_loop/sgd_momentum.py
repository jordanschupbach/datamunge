# Native Python NN example. SGD, momentum and Nesterov equations and minibatches.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.3_optimizers_and_training_loop/sgd_momentum.py
"""Independent kernel formulas, synchronous gradients and shuffled regression."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options

KINDS = ["Sgd", "Momentum", "Nesterov"]


def configuration(name):
    config = nn.OptimizerConfig()
    config.kind = getattr(nn.OptimizerKind, name)
    return config


def near(actual, expected, tolerance=2e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def slot_floats(opt):
    return sum(
        value.numel()
        for name, value in opt.state()
        if name not in ("step", "nadam_product")
    )


def f(value):
    return nn.scalar(value).item()


def kernel_agreement(name):
    initial = [1, -2, 0.5]
    w = nn.from_vector(initial).requires_grad_()
    config = configuration(name)
    opt = nn.Optimizer([w], 0.1, config)
    fresh = opt.state()
    reference = list(initial)
    velocity = [0.0] * 3
    mu = f(config.momentum)
    worst = 0
    for t in range(20):
        gradient = nn.from_vector([-0.7 if t % 3 == 0 else 0.3, f(f(0.1) * (t - 8)), 0])
        rate = f(0.1 if t < 10 else 0.03)
        opt.set_learning_rate(rate)
        opt.step([gradient])
        for j, g in enumerate(gradient.values()):
            if name == "Sgd":
                reference[j] = f(reference[j] - f(rate * g))
            else:
                velocity[j] = f(f(velocity[j] * mu) + g)
                update = (
                    f(f(g * rate) + f(f(velocity[j] * mu) * rate))
                    if name == "Nesterov"
                    else f(rate * velocity[j])
                )
                reference[j] = f(reference[j] - update)
        for got, want in zip(w.values(), reference):
            near(got, want)
            worst = max(worst, abs(got - want))
        assert opt.iterations() == t + 1
    print(
        f"{name}: 20 changing-gradient diagnostic updates, max kernel error={worst:.9g}, slots={slot_floats(opt)} floats"
    )
    with nn.no_grad():
        w.copy_(nn.from_vector(initial))
    opt.load_state(fresh)
    assert (
        list(w.values()) == initial and opt.iterations() == 0 and slot_floats(opt) == 0
    )


def coupled_gradients():
    x, y = nn.scalar(1).requires_grad_(), nn.scalar(2).requires_grad_()
    opt = nn.Optimizer([x, y], 0.1)
    expected_x, expected_y = 1.0, 2.0
    for _ in range(10):
        next_x = expected_x - 0.1 * expected_y
        expected_y -= 0.1 * expected_x
        expected_x = next_x
        opt.minimize(nn.mul(x, y))
        near(x.item(), expected_x)
        near(y.item(), expected_y)
    print("Coupled gradients use the same old parameter state.")


def quadratic(name, steps):
    w = nn.from_vector([4, 4]).requires_grad_()
    scale = nn.from_vector([0.5, 50])
    opt = nn.Optimizer([w], 0.01, configuration(name))
    values = []
    for t in range(steps + 1):
        loss = nn.sum(nn.mul(scale, nn.square(w)))
        if t in [0, 10, 50, steps]:
            value = loss.item()
            assert math.isfinite(value)
            values.append((t, value))
            if t == steps and steps >= 200:
                assert value < 0.15
        if t < steps:
            opt.minimize(loss)
    print(name, "quadratic loss by step:", values)


def minibatches(name, epochs):
    w, b = nn.scalar(0).requires_grad_(), nn.scalar(0).requires_grad_()
    opt = nn.Optimizer([w, b], 0.03, configuration(name))
    ordering = list(range(64))
    rng = nn.Mt19937(7)
    for epoch in range(epochs):
        ordering = list(rng.shuffle(ordering))
        assert sorted(ordering) == list(range(64))
        for start in range(0, 64, 8):
            x = nn.sub(
                nn.div(nn.mul(nn.from_vector(ordering[start : start + 8]), 2), 63), 1
            )
            y = nn.sub(nn.mul(x, 3), 0.5)
            prediction = nn.add(nn.mul(x, w), b)
            opt.minimize(nn.mean(nn.square(nn.sub(prediction, y))))
    assert (
        math.isfinite(w.item())
        and math.isfinite(b.item())
        and opt.iterations() == 8 * epochs
    )
    if epochs >= 40:
        near(w.item(), 3, 0.005)
        near(b.item(), -0.5, 0.005)
    print(
        f"{name}: {epochs} minibatch epochs, slope={w.item():.9g}, intercept={b.item():.9g}, updates={opt.iterations()}"
    )


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    for name in KINDS:
        kernel_agreement(name)
    coupled_gradients()
    for name in KINDS:
        quadratic(name, args["epochs"])
    for name in KINDS:
        minibatches(name, args["epochs"])
    print("All SGD/momentum/Nesterov checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
