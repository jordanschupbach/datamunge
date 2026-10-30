# Native Python NN example. LAMB/LARS trust ratios, Lion signs and Adafactor moments.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.3_optimizers_and_training_loop/advanced_optimizers.py
"""Optimizer mechanics, state memory and direct matrix fitting."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options

KINDS = ["Lamb", "Lars", "Lion", "Adafactor"]


def config(name):
    cfg = nn.OptimizerConfig()
    cfg.kind = getattr(nn.OptimizerKind, name)
    return cfg


def slot_floats(opt):
    return sum(
        value.numel()
        for name, value in opt.state()
        if name not in ("step", "nadam_product")
    )


def near(actual, expected, tolerance=1e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def trust_ratio(name):
    w = nn.from_vector([3, 4]).requires_grad_()
    cfg = config(name)
    cfg.momentum = 0
    cfg.trust_coefficient = 0.1
    opt = nn.Optimizer([w], 0.1, cfg)
    opt.step([nn.from_vector([1, -2])])
    values = list(w.values())
    relative = math.hypot(values[0] - 3, values[1] - 4) / 5
    near(relative, 0.1 if name == "Lamb" else 0.01, 1e-7)
    print(name, "relative update norm:", relative)


def lion_sign():
    w = nn.scalar(1).requires_grad_()
    cfg = config("Lion")
    cfg.beta2 = 0.99
    opt = nn.Optimizer([w], 0.1, cfg)
    for gradient in [10, -0.85]:
        opt.step([nn.scalar(gradient)])
    near(w.item(), 0.8)
    near(dict(opt.state())["m/0"].item(), 0.0905)
    print("Lion uses old momentum for signs; gradients 10 then -.85 produce w=.8.")


def factorization():
    w = nn.ones([2, 3]).requires_grad_()
    cfg = config("Adafactor")
    cfg.scale_parameter = False
    cfg.relative_step = False
    cfg.update_clip = 0.5
    opt = nn.Optimizer([w], 0.1, cfg)
    opt.step([nn.from_vector([1, 2, 3, 2, 4, 6], [2, 3])])
    state = dict(opt.state())
    assert max(abs(v - 0.95) for v in w.values()) < 1e-6
    row = list(state["row/0"].values())
    col = list(state["col/0"].values())
    near(row[0], 14 / 3)
    near(row[1], 56 / 3, 2e-6)
    for j, value in enumerate(col):
        near(value, 2.5 * (j + 1) ** 2)
    assert slot_floats(opt) == 5
    print(
        "Adafactor rank-one preconditioner, clipped RMS update .5, five slots for 2x3 matrix passed."
    )


def zeros_and_memory():
    for name in KINDS:
        for shape in [[], [3], [2, 3, 4]]:
            w = nn.zeros(shape).requires_grad_()
            opt = nn.Optimizer([w], 0.01, config(name))
            for _ in range(3):
                opt.step([nn.zeros_like(w)])
            assert all(v == 0 for v in w.values())
    w = nn.zeros([256, 128]).requires_grad_()
    opt = nn.Optimizer([w], 0.01, config("Adafactor"))
    opt.step([nn.ones_like(w)])
    assert slot_floats(opt) == 384
    print(
        "Zero-gradient scalar/vector/tensor checks passed; 256x128 Adafactor uses 384 moment floats."
    )


def fit(name, steps):
    w = nn.ones([3, 4]).requires_grad_()
    target = nn.reshape(nn.add(nn.mul(nn.sin(nn.arange(0, 12)), 0.5), 0.3), [3, 4])

    def loss():
        return nn.mean(nn.square(nn.sub(w, target)))

    cfg = config(name)
    cfg.trust_coefficient = 0.1
    if name == "Lion":
        cfg.beta2 = 0.99
    initial_lr = nn.scalar(
        0.01 if name == "Lion" else 0.1 if name == "Lars" else 0.03
    ).item()
    opt = nn.Optimizer([w], initial_lr, cfg)
    initial = loss().item()
    for t in range(steps):
        rate = nn.mul(nn.sub(1, nn.div(nn.scalar(t), 1500)), initial_lr).item()
        opt.set_learning_rate(rate)
        opt.minimize(loss())
    final = loss().item()
    assert math.isfinite(final) and final < initial and opt.iterations() == steps
    if steps >= 1500:
        assert final < 1e-5
    print(
        f"{name}: {steps} updates, initial lr={initial_lr:.6g}, final MSE={final:.9g}, persistent slots={slot_floats(opt)} floats"
    )


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if not 1 <= args["epochs"] <= 1500:
        raise ValueError("epochs must be in [1,1500]")
    nn.set_num_threads(2)
    trust_ratio("Lamb")
    trust_ratio("Lars")
    lion_sign()
    factorization()
    zeros_and_memory()
    for name in KINDS:
        fit(name, args["epochs"])
    print("All advanced optimizer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
