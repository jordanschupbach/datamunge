# Native Python NN example. Variable lifecycle and initializer algorithms.
"""Mutable tensors, independent stores, seeded initialization and gradients."""
import math
from pydatamunge import nn


def lifecycle():
    w = nn.Tensor()
    assert not w.defined()
    try:
        w.values()
    except RuntimeError as error:
        print("Read before assignment:", error)
    else:
        raise AssertionError("undefined read must fail")
    w = nn.from_vector([1, 2, 3, 4], [2, 2])
    alias = w
    w.add_(nn.full([2, 2], 1))
    w.copy_(nn.mul(w, 10))
    w.copy_(nn.concat([nn.slice_axis(w, 0, 0, 1), nn.from_vector([-1, -2], [1, 2])], 0))
    assert list(alias.values()) == [20, 30, -1, -2]
    store = nn.ParameterStore(1)
    store.set("w", w)
    other = store.clone()
    with nn.no_grad():
        other.at("w").add_(nn.full([2, 2], 100))
    assert list(store.at("w").values()) == [20, 30, -1, -2]
    assert list(other.at("w").values()) == [120, 130, 99, 98]
    print("Variable and cloned store:", store.at("w").str(), other.at("w").str())


def initializers():
    fan_in, fan_out = 256, 128
    shape = [fan_in, fan_out]

    def uniform(limit, seed):
        return nn.uniform(shape, -limit, limit, nn.Generator(seed))

    def normal(stddev, seed):
        return nn.normal(shape, 0, stddev, nn.Generator(seed))

    def truncated(stddev, seed):
        return nn.truncated_normal(
            shape, 0, stddev / 0.87962566103423978, nn.Generator(seed)
        )

    def orthogonal(rows, cols, seed):
        flip = rows < cols
        qr = nn.qr(
            nn.normal([max(rows, cols), min(rows, cols)], 0, 1, nn.Generator(seed))
        )
        q = nn.mul(qr.q, nn.sign(nn.matrix_diag_part(qr.r)))
        return nn.transpose(q, [1, 0]) if flip else q

    glorot, he = math.sqrt(2 / (fan_in + fan_out)), math.sqrt(2 / fan_in)
    entries = [
        ("zeros", nn.zeros(shape), 0),
        ("constant .1", nn.full(shape, 0.1), 0),
        ("uniform", uniform(0.05, 1), 0.05 / math.sqrt(3)),
        ("normal", normal(0.02, 2), 0.02),
        ("truncated normal", truncated(0.02, 3), 0.02),
        ("glorot uniform", uniform(math.sqrt(6 / (fan_in + fan_out)), 4), glorot),
        ("glorot normal", truncated(glorot, 5), glorot),
        ("he uniform", uniform(math.sqrt(6 / fan_in), 6), he),
        ("he normal", truncated(he, 7), he),
        ("orthogonal", orthogonal(fan_in, fan_out, 8), 1 / math.sqrt(fan_in)),
        ("orthogonal wide", orthogonal(16, 64, 9), 1 / math.sqrt(64)),
    ]
    for label, tensor, expected in entries:
        values = list(tensor.values())
        mean = sum(values) / len(values)
        stddev = math.sqrt(
            max(0, sum(v * v for v in values) / len(values) - mean * mean)
        )
        print(
            f"{label}: {tuple(tensor.shape())}, mean={mean:.6g}, std={stddev:.6g}, expected={expected:.6g}, range=[{min(values):.6g},{max(values):.6g}]"
        )
        if expected:
            assert abs(stddev / expected - 1) < 0.03
            assert abs(mean) < 4 * expected / math.sqrt(len(values)) + 1e-7
        else:
            assert max(values) == min(values)
    assert max(entries[4][1].values()) <= 2 * 0.02 / 0.87962566103423978 + 1e-6
    for label, q, _ in entries[-2:]:
        tall = q.size(0) >= q.size(1)
        gram = nn.matmul(q, q, tall, not tall)
        error = max(abs(v) for v in nn.sub(gram, nn.eye(gram.size(0))).values())
        assert error < 1e-5
        print(label, "orthogonality error:", error)


def main(arguments=None):
    lifecycle()
    initializers()
    seeded, unseeded = [], []
    for i in range(2):
        seeded.append(list(nn.normal([3], 0, 1, nn.Generator(123)).values()))
        unseeded.append(list(nn.normal([3]).values()))
        print("Seeded / unseeded:", seeded[-1], unseeded[-1])
    assert seeded[0] == seeded[1] and unseeded[0] != unseeded[1]
    variable = nn.from_vector([1, 2, 3]).requires_grad_()
    with nn.no_grad():
        variable.add_(nn.from_vector([10, 10, 10]))
    gradient = nn.grad(nn.sum(nn.square(variable)), [variable])[0]
    assert list(variable.values()) == [11, 12, 13] and list(gradient.values()) == [
        22,
        24,
        26,
    ]
    print("Resource variable:", variable.str(), "gradient:", gradient.str())
    print("All variable and initializer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
