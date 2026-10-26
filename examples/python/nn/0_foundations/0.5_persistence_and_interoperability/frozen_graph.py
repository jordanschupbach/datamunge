# Native Python NN example. Training, dependency pruning and frozen export live here.
import math
from pathlib import Path
import os
from pydatamunge import nn


class Affine:
    def __init__(self, initial):
        self.store = nn.ParameterStore()
        self.w = self.store.get('model/w', [], nn.init.constant(initial))
        self.b = self.store.get('model/b', [], nn.init.constant(initial))

    def predict(self, x):
        return nn.add(nn.mul(x, self.w), self.b)


def freeze(store, endpoints, outputs):
    used = [False] * store.size()
    options = nn.GradOptions()
    options.allow_unused = True
    probe = nn.from_vector([1])
    for name in outputs:
        if name not in endpoints:
            raise ValueError('unknown export output: ' + name)
        gradients = nn.grad(nn.sum(endpoints[name](probe, probe)), store.parameters(), options)
        used = [old or value.defined() for old, value in zip(used, gradients)]
    return [(name, store.at(name).detach().clone()) for name, keep in zip(store.names(), used) if keep]


def main(arguments=None):
    import sys
    epochs = 500
    for argument in (sys.argv[1:] if arguments is None else arguments):
        if argument == '--quick':
            epochs = min(epochs, 5)
        elif argument == '--full':
            pass
        elif argument.startswith('epochs='):
            epochs = int(argument.split('=', 1)[1])
        else:
            raise ValueError('unknown option: ' + argument)
    # Explicit budgets take precedence over quick mode regardless of argument order.
    for argument in (sys.argv[1:] if arguments is None else arguments):
        if argument.startswith('epochs='):
            epochs = int(argument.split('=', 1)[1])
    if epochs <= 0:
        raise ValueError('positive training budget required')
    model = Affine(0)
    cfg = nn.OptimizerConfig(); cfg.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(model.store, .05, cfg)
    endpoints = {'predictions': lambda x, y: model.predict(x),
                 'loss': lambda x, y: nn.mean(nn.square(nn.sub(model.predict(x), y)))}
    x = nn.from_vector([-2, -1, 0, 1, 2]); y = nn.from_vector([-4.5, -2.5, -.5, 1.5, 3.5])
    initial_loss = endpoints['loss'](x, y).item()
    for _ in range(epochs):
        optimizer.minimize(endpoints['loss'](x, y))
    assert optimizer.iterations() == epochs and endpoints['loss'](x, y).item() < initial_loss
    query = nn.from_vector([-3, .5, 4])
    expected = list(model.predict(query).values())
    new_batch = model.predict(nn.from_vector([5])).item()
    assert all(math.isfinite(value) for value in expected)
    if epochs >= 500:
        assert max(abs(a-b) for a, b in zip(expected, [-6.5, .5, 7.5])) < 1e-5
        assert abs(new_batch - 9.5) < 1e-5
    frozen = freeze(model.store, endpoints, ['predictions'])
    training_count = model.store.size() + len(optimizer.state())
    assert len(frozen) == 2 and len(frozen) < training_count
    assert all(name.startswith('model/') and not value.requires_grad() for name, value in frozen)
    directory = Path(os.environ.get('DATAMUNGE_OUTPUT_DIR', 'output/nn/python/quick')) / 'models/frozen_affine'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'weights.npz'
    nn.save_npz(str(path), frozen)
    (directory / 'signature.txt').write_text('input=features float32 [batch]\noutput=predictions float32 [batch]\npredictions = features * model/w + model/b\n')
    with nn.no_grad():
        model.w.fill_(-100); model.b.fill_(100)
    serving = Affine(float('nan'))
    serving.store.load(str(path), True)
    assert list(serving.predict(query).values()) == expected
    assert serving.predict(nn.from_vector([5])).item() == new_batch
    try:
        freeze(model.store, endpoints, ['does_not_exist'])
    except ValueError:
        pass
    else:
        raise AssertionError('unknown output must fail')
    print('training state arrays=', training_count, 'frozen arrays=', len(frozen))
    print('predictions=', expected, 'exact after reload; new batch=', new_batch)
    print('All frozen-graph checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
