# Native Python NN example. Sinusoidal, learned, rotary and ALiBi checks.
"""Position representations, independent rotations and relative-position scores."""
import math
from pydatamunge import nn


def random_tensor(shape, rng):
    rng.reset_normal()
    return nn.from_vector(rng.normal32_values(math.prod(shape)), shape)


def sinusoidal():
    length, width = 64, 16
    table = nn.sinusoidal_encoding(length, width)
    print(
        "Sinusoidal wavelengths:",
        [2 * math.pi * 10000 ** (2 * i / width) for i in range(width // 2)],
    )
    for position in [0, 1, 2, 3, 10, 50]:
        print("Position", position, [table.at([position, j]) for j in range(width)])
    norms = [
        math.sqrt(sum(table.at([t, j]) ** 2 for j in range(width)))
        for t in range(length)
    ]
    assert max(abs(v - math.sqrt(width / 2)) for v in norms) < 1e-5
    error = 0
    for offset in [1, 5, 20]:
        for t in range(length - offset):
            for i in range(width // 2):
                theta = offset / 10000 ** (2 * i / width)
                sn, cs = table.at([t, 2 * i]), table.at([t, 2 * i + 1])
                error = max(
                    error,
                    abs(
                        sn * math.cos(theta)
                        + cs * math.sin(theta)
                        - table.at([t + offset, 2 * i])
                    ),
                    abs(
                        -sn * math.sin(theta)
                        + cs * math.cos(theta)
                        - table.at([t + offset, 2 * i + 1])
                    ),
                )
    assert error < 1e-6
    print(
        "Row norm range:", min(norms), max(norms), "fixed-rotation shift error:", error
    )
    for t in [0, 7, 30]:
        dots = []
        for offset in range(9):
            dot = sum(
                table.at([t, j]) * table.at([t + offset, j]) for j in range(width)
            )
            reference = sum(
                table.at([0, j]) * table.at([offset, j]) for j in range(width)
            )
            assert abs(dot - reference) < 1e-5
            dots.append(dot)
        print("Offset dot products at position", t, dots)


def learned():
    p = nn.ParameterStore(1)
    table = nn.learned_encoding(p, "enc", 10, 4)
    used = nn.slice(table, [0, 0], [6, 4])
    rng = nn.Mt19937_64(2)
    x, r = random_tensor([3, 6, 4], rng), random_tensor([3, 6, 4], rng)
    y = nn.add(x, used)
    gradient = nn.grad(nn.sum(nn.mul(y, r)), [p.at("enc/pos")])[0]
    assert tuple(y.shape()) == (3, 6, 4)
    norms = []
    for position in range(10):
        square = sum(gradient.at([position, j]) ** 2 for j in range(4))
        assert (square > 0) if position < 6 else square == 0
        norms.append(math.sqrt(square))
    print("Learned positional gradient row norms:", norms)


def rotary():
    length, width = 12, 8
    rng = nn.Mt19937_64(3)
    q, k = random_tensor([1, length, width], rng), random_tensor(
        [1, length, width], rng
    )
    rotated = nn.apply_rotary(q)
    error = norm_error = 0
    for t in range(length):
        before = after = 0
        for j in range(width // 2):
            theta = t / 10000 ** (2 * j / width)
            a, b = q.at([0, t, 2 * j]), q.at([0, t, 2 * j + 1])
            ra = a * math.cos(theta) - b * math.sin(theta)
            rb = a * math.sin(theta) + b * math.cos(theta)
            oa, ob = rotated.at([0, t, 2 * j]), rotated.at([0, t, 2 * j + 1])
            error = max(error, abs(ra - oa), abs(rb - ob))
            before += a * a + b * b
            after += oa * oa + ob * ob
        norm_error = max(norm_error, abs(before - after))
    assert error < 1e-5 and norm_error < 1e-4
    q_same = nn.broadcast_to(nn.slice(q, [0, 0, 0], [1, 1, width]), [1, length, width])
    k_same = nn.broadcast_to(nn.slice(k, [0, 0, 0], [1, 1, width]), [1, length, width])
    scores = nn.matmul(nn.apply_rotary(q_same), nn.apply_rotary(k_same), False, True)
    print("RoPE hand-rotation error / squared-norm error:", error, norm_error)
    for offset in [0, 1, 3]:
        values = []
        for n in range(0, length - offset, 2):
            value = scores.at([0, n + offset, n])
            assert abs(value - scores.at([0, offset, 0])) < 1e-4
            values.append(value)
        print("Relative-position score offset", offset, values)


def alibi():
    heads, length = 4, 8
    slopes = list(nn.alibi_slopes(heads))
    bias = nn.alibi_bias(heads, length)
    weights = nn.softmax(nn.add(nn.zeros([heads, length, length]), bias))
    print("ALiBi slopes:", slopes, "head-0 bias:")
    for m in range(length):
        print([bias.at([0, m, n]) for n in range(length)])
    for head in range(heads):
        z = sum(math.exp(-slopes[head] * (length - 1 - n)) for n in range(length))
        row = []
        for n in range(length):
            actual = weights.at([head, length - 1, n])
            expected = math.exp(-slopes[head] * (length - 1 - n)) / z
            assert abs(actual - expected) < 1e-6
            row.append(actual)
        for m in range(length):
            assert abs(sum(weights.at([head, m, n]) for n in range(length)) - 1) < 1e-6
            assert all(weights.at([head, m, n]) == 0 for n in range(m + 1, length))
        print("Head", head, "last-query attention:", row)


def main(arguments=None):
    nn.set_num_threads(2)
    sinusoidal()
    learned()
    rotary()
    alibi()
    print("All positional encoding checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
