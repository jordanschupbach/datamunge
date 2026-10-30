# Native Python NN example. Layer, RMS, group and instance normalization checks.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/normalization_layers.py
"""Independent grouped moments, batch independence and affine invariances."""
import math
from pydatamunge import nn

EPS = 1e-5


def standardize(values, groups, group, center=True):
    sums = [0.0] * groups
    squares = [0.0] * groups
    counts = [0] * groups
    for i, value in enumerate(values):
        g = group(i)
        sums[g] += value
        squares[g] += value * value
        counts[g] += 1
    output = []
    for i, value in enumerate(values):
        g = group(i)
        mean = sums[g] / counts[g] if center else 0
        variance = squares[g] / counts[g] - mean * mean
        output.append((value - mean) / math.sqrt(variance + EPS))
    return output


def max_error(tensor, expected):
    actual = list(tensor.values())
    assert len(actual) == len(expected)
    return max(abs(a - b) for a, b in zip(actual, expected))


def difference(a, b):
    return max_error(a, list(b.values()))


def layer_and_rms():
    p = nn.ParameterStore()
    rng = nn.Mt19937_64(1)
    values = [rng.normal32(2, 3) for _ in range(18)]
    x = nn.from_vector(values, [3, 6])
    ln = nn.layer_norm(p, "ln", x, EPS)
    rms = nn.rms_norm(p, "rms", x, EPS)
    error_ln = max_error(ln, standardize(values, 3, lambda i: i // 6))
    error_rms = max_error(rms, standardize(values, 3, lambda i: i // 6, False))
    assert error_ln < 1e-5 and error_rms < 1e-5
    print("Layer/RMS hand errors:", error_ln, error_rms)
    print(
        "Row 0 input:",
        values[:6],
        "LN:",
        list(ln.values())[:6],
        "RMS:",
        list(rms.values())[:6],
    )
    rng.reset_normal()
    sequence = [rng.normal32(-1, 2) for _ in range(48)]
    output = nn.layer_norm(p, "ln", nn.from_vector(sequence, [2, 4, 6]), EPS)
    error = max_error(output, standardize(sequence, 8, lambda i: i // 6))
    assert tuple(output.shape()) == (2, 4, 6) and error < 1e-5
    changed = values.copy()
    changed[6:12] = [100 * (j - 2) for j in range(6)]
    altered = nn.layer_norm(p, "ln", nn.from_vector(changed, [3, 6]), EPS)
    batch_difference = difference(nn.select(ln, 0, 0), nn.select(altered, 0, 0))
    assert batch_difference == 0
    centered = nn.sub(x, nn.mean(x, [1], True))
    zero_difference = difference(
        nn.layer_norm(p, "ln", centered, EPS), nn.rms_norm(p, "rms", centered, EPS)
    )
    assert zero_difference < 1e-5
    print(
        "Sequence hand error:",
        error,
        "other-row change:",
        batch_difference,
        "zero-mean LN/RMS difference:",
        zero_difference,
    )


def group_norms():
    p = nn.ParameterStore()
    rng = nn.Mt19937_64(2)
    raw = nn.from_vector([rng.normal32() for _ in range(256)], [2, 4, 4, 8])
    channels = nn.arange(0, 8)
    x = nn.add(nn.mul(raw, nn.add(channels, 1)), nn.mul(channels, 3))
    values = list(x.values())
    outputs = []
    for groups in [1, 2, 4, 8]:
        output = nn.group_norm(p, "gn" + str(groups), x, groups, EPS)
        outputs.append(output)
        per = 8 // groups
        expected = standardize(
            values, 2 * groups, lambda i: (i // 128) * groups + (i % 8) // per
        )
        error = max_error(output, expected)
        assert error < 1e-5
        print("GroupNorm G=", groups, "hand error:", error)
    instance = nn.instance_norm(p, "in", x, EPS)
    assert difference(outputs[-1], instance) == 0
    mean1 = [
        sum(outputs[0].at([0, h, w, c]) for h in range(4) for w in range(4)) / 16
        for c in range(8)
    ]
    mean8 = [
        sum(outputs[-1].at([0, h, w, c]) for h in range(4) for w in range(4)) / 16
        for c in range(8)
    ]
    assert max(abs(v) for v in mean8) < 1e-5
    print("G=1 example-0 channel means:", mean1)
    print("G=8 example-0 channel means:", mean8, "InstanceNorm equals G=C exactly.")


def invariances():
    p = nn.ParameterStore()
    rng = nn.Mt19937_64(3)
    x = nn.from_vector([rng.normal32(0.5, 2) for _ in range(64)], [4, 16])
    baseline_ln = nn.layer_norm(p, "ln", x, EPS)
    baseline_rms = nn.rms_norm(p, "rms", x, EPS)
    for label, a, c, ln_invariant, rms_invariant in [
        ("shift", 1, 5, True, False),
        ("scale", 3, 0, True, True),
        ("negative scale", -2, 0, False, False),
        ("scale and shift", 3, 5, True, False),
    ]:
        transformed = nn.add(nn.mul(x, a), c)
        dl = difference(nn.layer_norm(p, "ln", transformed, EPS), baseline_ln)
        dr = difference(nn.rms_norm(p, "rms", transformed, EPS), baseline_rms)
        assert (dl < 1e-4) == ln_invariant and (dr < 1e-4) == rms_invariant
        print(label, "LN/RMS differences:", dl, dr)
    row = list(nn.select(baseline_ln, 0, 0).values())
    mean = sum(row) / 16
    variance = sum(v * v for v in row) / 16 - mean * mean
    assert abs(mean) < 1e-6 and abs(variance - 1) < 1e-4
    print("LN output row-0 mean / variance:", mean, variance)


def main(arguments=None):
    nn.set_num_threads(2)
    layer_and_rms()
    group_norms()
    invariances()
    print("All normalization layer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
