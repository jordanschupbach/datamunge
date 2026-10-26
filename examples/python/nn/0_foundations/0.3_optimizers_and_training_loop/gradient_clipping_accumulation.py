# Native Python NN example. Gradient clipping and unequal microbatch accumulation.
"""Compare all Adam moments and weights against a true batch of seven."""
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def near(actual, expected, tolerance=1e-6):
    assert abs(actual - expected) <= tolerance, (actual, expected)


def clipping():
    gradient = nn.from_vector([3, 4])
    result = nn.clip_by_global_norm([gradient], 2)
    clipped, norm = result[0], result[1]
    near(norm, 5, 0)
    near(2 / max(norm, 2), 0.4)
    for actual, expected in zip(clipped[0].values(), [1.2, 1.6]):
        near(actual, expected)
    assert list(nn.clip_by_value([gradient], -2, 2)[0].values()) == [2, 2]
    split = nn.clip_by_global_norm([nn.scalar(3), nn.scalar(4)], 2)
    near(split[0][0].item(), 1.2)
    near(split[0][1].item(), 1.6)
    zero = nn.clip_by_global_norm([nn.zeros([2])], 2)
    assert list(zero[0][0].values()) == [0, 0] and zero[1] == 0
    assert 2 / max(zero[1], 2) == 1
    print("(3,4), limit 2: global norm -> (1.2,1.6); by value -> (2,2).")
    print("Split tensors share one norm; zero gradients remain finite.")


def slots(optimizer):
    return {
        name: tensor
        for name, tensor in optimizer.state()
        if name not in ("step", "nadam_product")
    }


def accumulation(steps):
    w, b = nn.scalar(3).requires_grad_(), nn.scalar(-1).requires_grad_()
    wr, br = nn.scalar(3).requires_grad_(), nn.scalar(-1).requires_grad_()

    def loss(weight, bias, x, y):
        return nn.mean(nn.square(nn.sub(nn.add(nn.mul(x, weight), bias), y)))

    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    micro, batch = nn.Optimizer([w, b], 0.01, config), nn.Optimizer(
        [wr, br], 0.01, config
    )
    accumulator = nn.GradientAccumulator()

    def apply():
        mean = accumulator.mean()
        clipped = nn.clip_by_global_norm(mean, 1)
        micro.step(clipped[0])
        accumulator.reset()

    try:
        apply()
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty accumulation accepted")
    assert micro.iterations() == 0 and w.item() == 3
    x = nn.from_vector([-3, -2, -1, 0, 1, 2, 3])
    y = nn.from_vector([-5, -3, -1, 1, 3, 5, 7])
    try:
        accumulator.add(nn.grad(loss(w, b, x, y), [w, b]), 0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero microbatch weight accepted")
    assert accumulator.total_weight() == 0
    worst = 0
    for update in range(steps):
        start = 0
        for size in [2, 3, 2]:
            bx, by = nn.slice_axis(x, 0, start, start + size), nn.slice_axis(
                y, 0, start, start + size
            )
            accumulator.add(nn.grad(loss(w, b, bx, by), [w, b]), size)
            start += size
        if update == 0:
            mean = accumulator.mean()
            near(mean[0].item(), 8)
            near(mean[1].item(), -4)
        apply()
        clipped = nn.clip_by_global_norm(nn.grad(loss(wr, br, x, y), [wr, br]), 1)
        batch.step(clipped[0])
        actual, reference = slots(micro), slots(batch)
        assert list(actual) == list(reference) and len(actual) == 4
        pairs = [(w.item(), wr.item()), (b.item(), br.item())]
        pairs += [(actual[name].item(), reference[name].item()) for name in actual]
        for got, want in pairs:
            near(got, want, 2e-6)
            worst = max(worst, abs(got - want))
        assert micro.iterations() == batch.iterations() == update + 1
        assert accumulator.total_weight() == 0
    print(
        f"Microbatches 2+3+2 vs batch 7: max weight/moment error {worst:.9g} over {steps} Adam updates."
    )
    print("Empty updates and zero weights rejected without changing optimizer state.")


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    clipping()
    accumulation(args["epochs"])
    print("All clipping and accumulation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
