# Native Python NN example. Schedule boundaries and explicit scheduled SGD.
"""Use completed optimizer updates to index learning-rate schedules."""
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def boundaries():
    s = nn.schedule
    cases = [
        (s.step_decay(1, 0.5, 10, 9), 1),
        (s.step_decay(1, 0.5, 10, 10), 0.5),
        (s.exponential(1, 0.25, 10, 5), 0.5),
        (s.cosine(1, 0.1, 100, 0), 1),
        (s.cosine(1, 0.1, 100, 50), 0.55),
        (s.cosine(1, 0.1, 100, 200), 0.1),
        (s.warmup(1, 10, 0), 0),
        (s.warmup(1, 0, 0), 1),
        (s.warmup_cosine(1, 0.1, 10, 100, 10), 1),
        (s.one_cycle(0.1, 1, 0.01, 30, 100, 30), 1),
        (s.one_cycle(0.1, 1, 0.01, 30, 100, 200), 0.01),
        (s.cyclical(0.1, 1, 10, 10), 1),
        (s.cyclical(0.1, 1, 10, 20), 0.1),
        (s.cyclical(0.1, 1, 10, 30, 0.5), 0.55),
    ]
    assert max(abs(a - b) for a, b in cases) < 1e-12
    invalid = [
        lambda: s.step_decay(1, 0.5, 0, 1),
        lambda: s.cosine(1, 0, 0, 1),
        lambda: s.warmup_cosine(1, 0, 10, 10, 1),
        lambda: s.one_cycle(1, 0.1, 0, 5, 10, 0),
        lambda: s.cyclical(0.1, 1, 0, 0),
        lambda: s.warmup(1, 10, -1),
    ]
    for call in invalid:
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid schedule accepted")
    print("Endpoint, transition, clamping, cycle and invalid-input checks passed.")


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    boundaries()
    s = nn.schedule
    schedules = [
        ("step", lambda t: s.step_decay(0.1, 0.5, 25, t)),
        ("exponential", lambda t: s.exponential(0.1, 0.5, 25, t)),
        ("cosine", lambda t: s.cosine(0.1, 0.001, 100, t)),
        ("warmup+cosine", lambda t: s.warmup_cosine(0.1, 0.001, 10, 100, t)),
        ("one-cycle", lambda t: s.one_cycle(0.01, 0.1, 0.0001, 30, 100, t)),
        ("cyclical", lambda t: s.cyclical(0.01, 0.1, 20, t)),
        ("triangular2", lambda t: s.cyclical(0.01, 0.1, 20, t, 0.5)),
    ]
    print("Learning rates by completed update:", [0, 10, 20, 30, 50, 80, 100, 120])
    for name, schedule in schedules:
        print(name, [round(schedule(t), 8) for t in [0, 10, 20, 30, 50, 80, 100, 120]])
        w = nn.scalar(2).requires_grad_()
        optimizer = nn.Optimizer([w], 0)
        expected = 2.0
        for step in range(args["epochs"]):
            assert optimizer.iterations() == step
            rate = nn.scalar(schedule(step)).item()
            expected *= 1 - rate
            optimizer.set_learning_rate(rate)
            optimizer.minimize(nn.mul(nn.square(w), 0.5))
        assert optimizer.iterations() == args["epochs"]
        assert abs(w.item() - expected) < 2e-6 and abs(expected) <= 2
        if args["epochs"] >= 100:
            assert abs(expected) < 0.5
        print(
            f'{name}: {args["epochs"]} SGD updates, w={w.item():.9g}, closed form={expected:.9g}'
        )
    print("All learning-rate schedule checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
