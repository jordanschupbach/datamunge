# Native Python NN example. Adaptive optimizer formulas, state invariants and fitting.
"""Adagrad, RMSProp, Adam, AdamW, Nadam, AMSGrad and Adamax checks."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options

KINDS = ["Adagrad", "RMSProp", "Adam", "AdamW", "Nadam", "AMSGrad", "Adamax"]


def near(actual, expected, tolerance=2e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def config(name):
    cfg = nn.OptimizerConfig()
    cfg.kind = getattr(nn.OptimizerKind, name)
    return cfg


def fused_kernel(name):
    w = nn.from_vector([1, -2, 0.5]).requires_grad_()
    cfg = config(name)
    cfg.epsilon = 1e-4
    opt = nn.Optimizer([w], 0.03, cfg)
    reference = [1.0, -2.0, 0.5]
    first = [0.0] * 3
    second = [0.1 if name == "Adagrad" else 0.0] * 3
    eps, rho, b1, b2 = cfg.epsilon, cfg.rho, cfg.beta1, cfg.beta2
    worst = 0
    for t in range(1, 41):
        gradient = nn.from_vector(
            [2 if t % 3 == 0 else -0.2, 0 if t % 4 == 0 else 0.005, 0]
        )
        rate = nn.scalar(0.03 if t < 20 else 0.01).item()
        opt.set_learning_rate(rate)
        opt.step([gradient])
        b1p, b2p = b1**t, b2**t
        for i, g in enumerate(gradient.values()):
            if name == "Adagrad":
                second[i] += g * g
                reference[i] -= rate * g / (math.sqrt(second[i]) + eps)
            elif name == "RMSProp":
                second[i] += (g * g - second[i]) * (1 - rho)
                first[i] = rate * g / math.sqrt(second[i] + eps)
                reference[i] -= first[i]
            else:
                lr_t = rate * math.sqrt(1 - b2p) / (1 - b1p)
                eps_hat = eps * math.sqrt(1 - b2p)
                first[i] += (g - first[i]) * (1 - b1)
                second[i] += (g * g - second[i]) * (1 - b2)
                reference[i] -= lr_t * first[i] / (math.sqrt(second[i]) + eps_hat)
        for got, want in zip(w.values(), reference):
            near(got, want, 3e-6)
            worst = max(worst, abs(got - want))
    print(f"{name}: max fused-kernel error over 40 diagnostic updates={worst:.9g}")


def invariants(name):
    w = nn.from_vector([1, -2]).requires_grad_()
    twin = nn.from_vector([1, -2]).requires_grad_()
    cfg = config(name)
    if name == "AdamW":
        cfg.weight_decay = 0.1
    opt = nn.Optimizer([w, twin], 0.1, cfg)
    fresh = opt.state()

    def feed(values):
        gradient = nn.from_vector(values)
        opt.step([gradient, gradient])

    feed([3, -4])
    for got, other, g, initial in zip(w.values(), twin.values(), [3, -4], [1, -2]):
        if name == "Adagrad":
            direction = g / (math.sqrt(0.1 + g * g) + cfg.epsilon)
        elif name == "RMSProp":
            direction = g / math.sqrt((1 - cfg.rho) * g * g + cfg.epsilon)
        else:
            scale = 1
            if name == "Nadam":
                mu = cfg.beta1 * (1 - 0.5 * 0.96)
                mu_next = cfg.beta1 * (1 - 0.5 * 0.96 * 0.96)
                scale += mu_next * (1 - cfg.beta1) / (1 - mu * mu_next)
            direction = scale * g / (abs(g) + cfg.epsilon) + cfg.weight_decay * initial
        near(got, initial - 0.1 * direction)
        near(got, other, 1e-7)
    if name in ["AMSGrad", "Adamax"]:
        slot = "vmax/0" if name == "AMSGrad" else "u/0"
        previous = -1
        for t in range(8):
            value = dict(opt.state())[slot].values()[0]
            if name == "AMSGrad":
                assert value >= previous
            else:
                near(value, 3 * cfg.beta2**t)
            previous = value
            feed([0, 0])
    with nn.no_grad():
        w.copy_(nn.from_vector([1, -2]))
        twin.copy_(nn.from_vector([1, -2]))
    opt.load_state(fresh)
    for _ in range(5):
        feed([0, 0])
    near(w.values()[0], (1 - 0.1 * cfg.weight_decay) ** 5)
    assert opt.iterations() == 5
    print(
        name, "first update, independent slots, reset and zero-gradient checks passed."
    )


def fit(name, steps):
    w = nn.from_vector([4, -4]).requires_grad_()
    target, weight = nn.from_vector([1.5, -0.75]), nn.from_vector([1, 30])

    def loss():
        return nn.sum(nn.mul(weight, nn.square(nn.sub(w, target))))

    cfg = config(name)
    if name == "AdamW":
        cfg.weight_decay = 0.01
    opt = nn.Optimizer([w], 0.1, cfg)
    initial = loss().item()
    for t in range(steps):
        opt.set_learning_rate(
            nn.scalar(0.1 if t < 3200 else 0.01 if t < 3600 else 0.001).item()
        )
        opt.minimize(loss())
    assert loss().item() < initial and opt.iterations() == steps
    values = list(w.values())
    if steps >= 4000:
        near(values[0], 1.5, 0.04)
        near(values[1], -0.75, 0.04)
    slots = sum(
        t.numel()
        for key, t in opt.state()
        if key != "step" and (key != "nadam_product" or name == "Nadam")
    )
    print(
        f"{name}: {steps} updates, w={values}, loss={loss().item():.9g}, slots={slots} floats + int64 step"
    )


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    for name in KINDS[:3]:
        fused_kernel(name)
    for name in KINDS:
        invariants(name)
    for name in KINDS:
        fit(name, args["epochs"])
    print("All adaptive optimizer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
