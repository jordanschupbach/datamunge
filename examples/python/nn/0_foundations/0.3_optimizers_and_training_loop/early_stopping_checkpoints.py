# Native Python NN example. Validation monitoring and complete model/optimizer checkpoints.
"""Best-checkpoint selection and exact resumed Adam trajectories."""
import math
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


class Model:
    def __init__(self, initial=0):
        self.store = nn.ParameterStore()
        self.w = self.store.get("w", [], nn.init.constant(initial))
        cfg = nn.OptimizerConfig()
        cfg.kind = nn.OptimizerKind.Adam
        self.optimizer = nn.Optimizer(self.store, 0.1, cfg)

    def train(self, count):
        for _ in range(count):
            self.optimizer.set_learning_rate(
                nn.schedule.step_decay(0.1, 0.5, 25, self.optimizer.iterations())
            )
            self.optimizer.minimize(nn.square(nn.sub(self.w, 4)))

    def validation(self):
        with nn.no_grad():
            return nn.square(nn.sub(self.w, 2)).item()

    def state(self):
        return [
            ("params/" + name, self.store.at(name).detach().clone())
            for name in self.store.names()
        ] + [("optimizer/" + name, value) for name, value in self.optimizer.state()]

    def save(self, path):
        nn.save_npz(str(path), self.state())

    def restore(self, path, weights_only=False):
        archive = nn.load_npz(str(path))
        params = {}
        slots = nn.NamedTensors()
        for name, value in archive:
            if name.startswith("params/"):
                params[name[7:]] = value
            elif name.startswith("optimizer/"):
                slots.push_back(nn.NamedTensor(name[10:], value))
            else:
                raise ValueError("unexpected checkpoint entry " + name)
        if set(params) != set(self.store.names()):
            raise ValueError("checkpoint parameter names do not match")
        if not weights_only and not slots:
            raise ValueError("checkpoint has no optimizer state")
        if not weights_only:
            for name, value in slots:
                if (
                    name not in ("step", "nadam_product", "m/0", "v/0")
                    or value.dim() != 0
                ):
                    raise ValueError("invalid scalar Adam checkpoint slot " + name)
        probe = Model()
        for name, value in params.items():
            probe.store.set(name, value)
        if not weights_only:
            probe.optimizer.load_state(slots)
        for name, value in params.items():
            self.store.set(name, value)
        if not weights_only:
            self.optimizer.load_state(slots)


def monitor_checks():
    stop = nn.EarlyStopping(2, 0.1)
    assert stop.observe(1, 0).improved and not stop.observe(0.95, 1).improved
    assert stop.observe(0.8, 2).improved and stop.misses() == 0
    assert not stop.observe(0.8, 3).stop and stop.observe(0.85, 4).stop
    maximize = nn.EarlyStopping(0, 0, True)
    assert not maximize.observe(0.4, 0).stop and maximize.observe(0.5, 1).improved
    assert maximize.observe(0.5, 2).stop
    try:
        maximize.observe(0.6, 2)
    except RuntimeError:
        pass
    else:
        raise AssertionError("repeated epoch accepted")
    print("Patience, min_delta, tie, maximize and epoch-order checks passed.")


def best_checkpoint(path, epochs):
    model = Model()
    monitor = nn.EarlyStopping(6, 1e-6)
    stopped = False
    for epoch in range(epochs):
        model.train(1)
        last_loss = model.validation()
        decision = monitor.observe(last_loss, epoch)
        if decision.improved:
            model.save(path)
        if decision.stop:
            stopped = True
            break
    if epochs >= 200:
        assert stopped
    model.restore(path)
    restored = model.validation()
    assert abs(restored - monitor.best()) < 1e-7 and restored <= last_loss
    if stopped:
        assert restored < 0.01 and restored < last_loss
    assert model.optimizer.iterations() == monitor.best_epoch() + 1
    print(
        f'{"Stopped" if stopped else "Budget exhausted"} after epoch {epoch}; best epoch {monitor.best_epoch()}; validation {last_loss:.9g} -> restored {restored:.9g}'
    )
    print(
        f"Restored w={model.w.item():.9g}, optimizer step={model.optimizer.iterations()}"
    )


def resume(path, prefix, suffix):
    original = Model()
    original.train(prefix)
    original.save(path)
    original.train(suffix)
    expected = original.state()
    restored = Model(math.nan)
    restored.restore(path)
    restored.train(suffix)
    actual = restored.state()
    assert len(actual) == len(expected)
    worst = 0
    for (name, a), (key, e) in zip(actual, expected):
        assert name == key and tuple(a.shape()) == tuple(e.shape())
        if name == "optimizer/step":
            assert a.item() == e.item() == prefix + suffix
        for got, want in zip(a.values(), e.values()):
            assert abs(got - want) < 1e-7
            worst = max(worst, abs(got - want))
    print(
        f"{prefix} + save/restore + {suffix} updates = {prefix+suffix} uninterrupted updates; max state error {worst:.9g}"
    )
    weights_only = Model()
    weights_only.restore(path, True)
    weights_only.train(suffix)
    correct = restored.w.item()
    different = weights_only.w.item()
    assert abs(different - correct) > 1e-7
    print(f"Weights-only resume w={different:.9g}; full-state resume w={correct:.9g}")
    try:
        restored.restore(str(path) + "_missing.npz")
    except RuntimeError:
        pass
    else:
        raise AssertionError("missing checkpoint accepted")
    assert restored.w.item() == correct
    # Corrupt optimizer state must be detected before changing model weights.
    broken = path.parent / "broken.npz"
    arrays = original.state()
    arrays = [
        (
            name,
            (
                nn.scalar(99)
                if name == "params/w"
                else nn.zeros([2]) if name == "optimizer/m/0" else value
            ),
        )
        for name, value in arrays
    ]
    nn.save_npz(str(broken), arrays)
    before = restored.state()
    try:
        restored.restore(broken)
    except (RuntimeError, ValueError):
        pass
    else:
        raise AssertionError("invalid optimizer moment shape accepted")
    for (name, a), (key, b) in zip(restored.state(), before):
        assert name == key and list(a.values()) == list(b.values())


def main(arguments=None):
    args = options(arguments, dict(epochs=5, full=False))
    if args["epochs"] < 2:
        raise ValueError("at least two epochs required for checkpoint resume")
    directory = output_root(args["full"]) / "checkpoints/early_stopping"
    directory.mkdir(parents=True, exist_ok=True)
    monitor_checks()
    best_checkpoint(directory / "best.npz", args["epochs"])
    prefix = max(1, args["epochs"] // 2)
    resume(directory / "resume.npz", prefix, args["epochs"] - prefix)
    print("All early stopping and checkpoint checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
