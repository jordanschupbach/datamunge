# Native Python NN example. Pooling windows, gradients, global/3-D ops and translation.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/pooling.py
"""Check border averages, argmax routing and the original random fixtures."""
from pydatamunge import nn


def require_values(tensor, expected):
    actual = list(tensor.values())
    assert (
        len(actual) == len(expected)
        and max(abs(a - b) for a, b in zip(actual, expected)) < 1e-6
    )


def image(height, width, values):
    return nn.from_vector(values, [1, height, width, 1])


def by_hand():
    x = image(5, 5, list(range(1, 26)))
    for name, operation, padding, expected in [
        ("max VALID", nn.max_pool2d, nn.Padding.Valid, [7, 9, 17, 19]),
        ("avg VALID", nn.avg_pool2d, nn.Padding.Valid, [4, 6, 14, 16]),
        (
            "max SAME",
            nn.max_pool2d,
            nn.Padding.Same,
            [7, 9, 10, 17, 19, 20, 22, 24, 25],
        ),
        (
            "avg SAME",
            nn.avg_pool2d,
            nn.Padding.Same,
            [4, 6, 7.5, 14, 16, 17.5, 21.5, 23.5, 25],
        ),
    ]:
        output = operation(x, 2, 2, padding)
        require_values(output, expected)
        print(name, output.str())
    print("SAME border averages exclude padding; bottom-right is 25/1.")


def gradients():
    x = image(4, 4, [9, 1, 1, 1, 1, 1, 1, 8, 1, 1, 1, 1, 7, 1, 6, 1]).requires_grad_()
    gm = nn.grad(nn.sum(nn.max_pool2d(x, 2, 2)), [x])[0]
    ga = nn.grad(nn.sum(nn.avg_pool2d(x, 2, 2)), [x])[0]
    require_values(gm, [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
    require_values(ga, [0.25] * 16)
    print("Max-pool gradient:", gm.str(), "Average-pool gradient:", ga.str())


def global_and_3d():
    rng = nn.Mt19937_64(1)
    x = nn.from_vector([rng.normal32() for _ in range(2 * 7 * 5 * 3)], [2, 7, 5, 3])
    volume = nn.from_vector(
        [rng.normal32() for _ in range(8 * 8 * 8 * 2)], [1, 8, 8, 8, 2]
    )
    average, maximum = nn.global_avg_pool2d(x), nn.global_max_pool2d(x)
    m3, a3 = nn.max_pool3d(volume, 2, 2), nn.avg_pool3d(volume, 2, 2)
    error_avg = error_max = 0
    for n in range(2):
        for c in range(3):
            values = [x.at([n, i, j, c]) for i in range(7) for j in range(5)]
            error_avg = max(error_avg, abs(sum(values) / 35 - average.at([n, c])))
            error_max = max(error_max, abs(max(values) - maximum.at([n, c])))
    window = [
        volume.at([0, 2 + a, 4 + b, 6 + c, 1])
        for a in range(2)
        for b in range(2)
        for c in range(2)
    ]
    op_m, op_a = m3.at([0, 1, 2, 3, 1]), a3.at([0, 1, 2, 3, 1])
    assert tuple(average.shape()) == (2, 3) and tuple(m3.shape()) == tuple(
        a3.shape()
    ) == (1, 4, 4, 4, 2)
    assert (
        error_avg < 1e-6
        and error_max == 0
        and abs(op_m - max(window)) < 1e-6
        and abs(op_a - sum(window) / 8) < 1e-6
    )
    print(
        "Global average / max errors:",
        error_avg,
        error_max,
        "3-D max / avg cell:",
        op_m,
        op_a,
    )


def translation():
    rng = nn.Mt19937_64(2)
    original = [rng.uniform32() for _ in range(256)]
    shifted = [
        0 if j == 0 else original[i * 16 + j - 1] for i in range(16) for j in range(16)
    ]
    a, b = image(16, 16, original), image(16, 16, shifted)

    def pools(x):
        p = nn.max_pool2d(x, 2, 2)
        return p, nn.max_pool2d(p, 2, 2), nn.global_max_pool2d(x)

    out, shift = pools(a), pools(b)

    def agreement(u, v):
        values = list(u.values())
        other = list(v.values())
        return (
            100 * sum(abs(x - y) <= 1e-6 for x, y in zip(values, other)) / len(values)
        )

    raw, one, two = (
        agreement(a, b),
        agreement(out[0], shift[0]),
        agreement(out[1], shift[1]),
    )
    global_same = abs(out[2].item() - shift[2].item()) < 1e-6
    assert round(raw) == 0 and round(one) == 28 and round(two) == 31 and global_same
    print(
        "Translation agreement percent, raw / one / two pools:",
        raw,
        one,
        two,
        "global max unchanged:",
        global_same,
    )


def main(arguments=None):
    nn.set_num_threads(2)
    by_hand()
    gradients()
    global_and_3d()
    translation()
    print("All pooling checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
