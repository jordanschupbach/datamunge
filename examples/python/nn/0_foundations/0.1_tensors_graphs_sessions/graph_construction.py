# Native Python NN example. Naming, eager ordering, placement and errors live here.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.1_tensors_graphs_sessions/graph_construction.py
from pydatamunge import nn


def main(arguments=None):
    store = nn.ParameterStore(1)
    a = store.get('a', [], nn.init.constant(1))
    w = nn.from_vector([1, 0, 0, 1], [2, 2]).requires_grad_(True)
    store.set('encoder/w', w)
    h1 = nn.add(a, nn.scalar(2))
    h2 = nn.add(h1, nn.scalar(3))
    p = store.get('block/k', [], nn.init.constant(5))
    q = store.get('block/k', [], nn.init.constant(7))
    assert p.is_same(q) and q.item() == 5 and store.size() == 3
    assert h2.item() == 6
    print('parameter names:', list(store.names()))
    print('recorded operations:', h1.grad_fn_name(), h2.grad_fn_name())
    y = nn.sum(nn.matmul(nn.mul(nn.reshape(nn.mul(h2, p), [1, 1]), nn.ones([1, 2])), w))
    ga, gw = nn.grad(y, [a, w])
    assert ga.item() == 10 and list(gw.values()) == [30, 30, 30, 30]
    print('dy/da:', ga.item(), 'dy/dw:', list(gw.values()))

    counter = nn.scalar(0, nn.DType.Int32)
    reads = [counter.clone().item() for _ in range(3)]
    for _ in range(3):
        counter.add_(nn.scalar(1, nn.DType.Int32))
        reads.append(counter.clone().item())
    reads.append(counter.clone().item())
    assert reads == [0, 0, 0, 1, 2, 3, 3]
    print('ordered counter reads:', reads)

    matrix = nn.from_vector([1, 2, 3, 4], [2, 2])
    placed = nn.add(nn.matmul(matrix, matrix), nn.matmul(matrix, matrix))
    assert list(placed.values()) == [14, 20, 30, 44]
    print('CPU placement result:', list(placed.values()))
    bad = nn.from_vector([1, 2, 3, 4, 5, 6], [2, 3])
    try:
        nn.matmul(bad, bad)
    except RuntimeError as error:
        print('Expected construction error:', error)
    else:
        raise AssertionError('incompatible matmul must fail immediately')
    later = nn.neg(bad)
    assert later.defined() and list(later.values())[-1] == -6
    print('All graph construction checks passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
