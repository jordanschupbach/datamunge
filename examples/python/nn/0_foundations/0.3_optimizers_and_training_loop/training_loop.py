# Native Python NN example. Resumable batching, weighted logging and explicit training.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.3_optimizers_and_training_loop/training_loop.py
"""Every epoch visits 103 rows once, including its seven-row final batch."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options


def batching_checks():
    original, restored = nn.BatchOrder(11, 4, 7), nn.BatchOrder(11, 4, 999)
    original.begin_epoch()
    first = list(original.next())
    restored.restore_state(original.save_state())
    seen = set(first)
    while True:
        batch = list(original.next())
        assert batch == list(restored.next())
        if not batch:
            break
        assert not seen.intersection(batch)
        seen.update(batch)
    assert seen == set(range(11))
    for _ in range(3):
        original.begin_epoch()
        restored.begin_epoch()
        while True:
            batch = list(original.next())
            assert batch == list(restored.next())
            if not batch:
                break
    state = restored.save_state()
    try:
        restored.restore_state(state + " bad")
    except RuntimeError:
        pass
    else:
        raise AssertionError("invalid sampler state accepted")
    assert restored.save_state() == state
    drop = nn.BatchOrder(10, 4, 0, False, True)
    drop.begin_epoch()
    assert (
        list(drop.next()) == [0, 1, 2, 3]
        and list(drop.next()) == [4, 5, 6, 7]
        and not list(drop.next())
    )
    singleton = nn.BatchOrder(1, 16, 0)
    singleton.begin_epoch()
    assert len(singleton.next()) == 1 and not list(singleton.next())
    source = nn.reshape(nn.arange(0, 6, 1, nn.DType.Int32), [3, 2])
    assert list(nn.take_rows(source, [2, 0, 2]).values()) == [4, 5, 0, 1, 4, 5]
    try:
        nn.take_rows(source, [3])
    except RuntimeError:
        pass
    else:
        raise AssertionError("out-of-range row accepted")
    mean = nn.WeightedMean()
    mean.add(2, 4)
    mean.add(10, 1)
    assert abs(mean.value() - 3.6) < 1e-12 and mean.count() == 5
    mean.reset()
    assert mean.count() == 0
    print("Exact mid-epoch resume and three future shuffles passed.")
    print(
        "Coverage, drop_last, singleton, row gathering, invalid state and weighted means passed."
    )


def train(epochs):
    index = nn.arange(0, 103)
    a, b = nn.sin(nn.mul(index, 0.7)), nn.cos(nn.mul(index, 0.4))
    inputs = nn.stack([a, b], 1)
    labels = nn.reshape(nn.add(nn.sub(nn.mul(a, 1.5), nn.mul(b, 0.7)), 0.2), [103, 1])
    w, bias = nn.zeros([2, 1]).requires_grad_(), nn.zeros([1]).requires_grad_()

    def loss_of(x, y):
        return nn.mean(nn.square(nn.sub(nn.bias_add(nn.matmul(x, w), bias), y)))

    cfg = nn.OptimizerConfig()
    cfg.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer([w, bias], 0.03, cfg)
    sampler = nn.BatchOrder(103, 16, 42)
    initial = loss_of(inputs, labels).item()
    for epoch in range(1, epochs + 1):
        sampler.begin_epoch()
        epoch_loss = nn.WeightedMean()
        visited = set()
        batches = 0
        while True:
            rows = list(sampler.next())
            if not rows:
                break
            assert not visited.intersection(rows)
            visited.update(rows)
            loss = loss_of(nn.take_rows(inputs, rows), nn.take_rows(labels, rows))
            optimizer.minimize(loss)
            epoch_loss.add(loss.item(), len(rows))
            batches += 1
        assert visited == set(range(103)) and epoch_loss.count() == 103 and batches == 7
        print(
            f"epoch {epoch}: batches={batches}, samples={epoch_loss.count()}, online mean loss={epoch_loss.value():.9g}"
        )
    final = loss_of(inputs, labels).item()
    weights = list(w.values())
    assert (
        math.isfinite(final)
        and final < initial
        and optimizer.iterations() == 7 * epochs
    )
    if epochs >= 50:
        assert (
            final < 1e-9
            and max(abs(x - y) for x, y in zip(weights, [1.5, -0.7])) < 1e-4
            and abs(bias.item() - 0.2) < 1e-4
        )
    print(
        f"Final fixed-weight MSE={final:.9g}, weights={weights}, bias={bias.item():.9g}, updates={optimizer.iterations()}"
    )


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 1:
        raise ValueError("epochs must be positive")
    nn.set_num_threads(2)
    batching_checks()
    train(args["epochs"])
    print("All training-loop checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
