# Native Python NN example. Absolute, rotary, ALiBi and T5 positions with full derivatives.
"""Five position mechanisms, cache invariants and 200 independent derivative probes."""
import math
from pathlib import Path
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


def sinusoidal(length, width):
    if length < 1 or width < 2 or width % 2:
        raise ValueError("positive length and even width required")
    return nn.sinusoidal_encoding(length, width)


def rotary(x, offset=0):
    if x.dim() < 2 or x.size(-2) < 1 or x.size(-1) < 2 or x.size(-1) % 2 or offset < 0:
        raise ValueError("invalid rotary dimensions or offset")
    return nn.apply_rotary(x, offset)


def slopes(heads):
    if heads < 1:
        raise ValueError("positive head count required")
    power = 1
    while power <= heads // 2:
        power *= 2
    result = [2.0 ** (-8.0 * h / power) for h in range(1, power + 1)]
    result.extend(2.0 ** (-4.0 * h / power) for h in range(1, 2 * (heads - power), 2))
    return nn.from_vector(result).values()


def bucket(relative, bidirectional=True, buckets=32, max_distance=128):
    if buckets < 4 or (bidirectional and buckets % 2):
        raise ValueError("invalid bucket count")
    bins = buckets // 2 if bidirectional else buckets
    exact = bins // 2
    if max_distance <= exact:
        raise ValueError("maximum distance must exceed exact range")
    distance = -relative
    side = bins if bidirectional and distance < 0 else 0
    distance = abs(distance) if bidirectional else max(distance, 0)
    value = (
        distance
        if distance < exact
        else min(
            bins - 1,
            exact
            + int(
                math.log(distance / exact)
                / math.log(max_distance / exact)
                * (bins - exact)
            ),
        )
    )
    return side + value


def buckets_grid(queries, keys, bidirectional=True, offset=0):
    if queries < 1 or keys < 1 or offset < 0:
        raise ValueError("invalid lengths or offset")
    values = [
        bucket(k - q - offset, bidirectional, 16, 32)
        for q in range(queries)
        for k in range(keys)
    ]
    return nn.cast(nn.from_vector(values, [queries, keys]), nn.DType.Int64)


def exercise(args):
    rng = nn.Mt19937(45801)
    store = nn.ParameterStore(45803)

    def random(shape):
        return nn.from_vector(rng.normal32_values(math.prod(shape)), shape)

    x = random([2, 5, 8])
    probe = random([2, 5, 8])
    source = "nn"
    fixture = Path(args["tf"]) / "positions.npz"
    if args["init"] != "nn" and fixture.exists():
        reference = dict(nn.load_npz(str(fixture)))
        for key, shape, ref in [
            ("absolute/pos", [9, 8], "learned_table"),
            ("relative", [16, 3], "relative_table"),
        ]:
            parameter = store.get(key, shape, nn.init.constant(0.0))
            assert tuple(reference[ref].shape()) == tuple(shape)
            with nn.no_grad():
                parameter.copy_(reference[ref])
        source = "TF"
    elif args["init"] == "tf":
        raise FileNotFoundError(fixture)
    sin = sinusoidal(5, 8)
    table = nn.learned_encoding(store, "absolute", 9, 8, 0.02)
    reltable = store.get("relative", [16, 3], nn.init.normal(0.2))
    indices = nn.cast(nn.from_vector([0, 2, 2, 5, 7]), nn.DType.Int32)
    q, k, v, ap = (
        random([3, 5, 8]),
        random([3, 5, 8]),
        random([3, 5, 4]),
        random([3, 5, 4]),
    )
    x.requires_grad_()

    def relative(bidirectional):
        return nn.transpose(
            nn.gather(reltable, buckets_grid(5, 5, bidirectional, 3), 0), [2, 0, 1]
        )

    def forward():
        absolute = nn.add(x, sin)
        learned = nn.add(x, nn.gather(table, indices, 0))
        rotated = rotary(x, 3)
        bias = relative(True)
        causal = relative(False)
        config = nn.AttentionOptions()
        config.mask = bias
        config.scale = 1 / math.sqrt(8)
        config.need_weights = True
        config.impl = nn.AttentionImpl.Plain
        attention = nn.scaled_dot_product_attention(q, k, v, config)
        objective = nn.add(
            nn.add(nn.sum(nn.mul(absolute, probe)), nn.sum(nn.mul(learned, probe))),
            nn.sum(nn.mul(rotated, probe)),
        )
        objective = nn.add(objective, nn.sum(nn.mul(attention.output, ap)))
        return absolute, learned, rotated, bias, causal, attention, objective

    absolute, learned, rotated, bias, causal, attention, objective = forward()
    alibi = nn.from_vector(
        [
            -1e9 if n > m else -s * abs(m - n) + 0.0
            for s in slopes(3)
            for m in range(5)
            for n in range(5)
        ],
        [3, 5, 5],
    )
    gradients = list(nn.grad(objective, [x, table, reltable]))
    cached = rotary(nn.slice(x.detach(), [0, 2, 0], [2, 3, 8]), 5)

    def dots(offset):
        return nn.matmul(rotary(q, offset), rotary(k, offset), False, True)

    dot0, dot7 = dots(0), dots(7)
    arrays = [
        ("probe", probe),
        ("indices", indices),
        ("q", q),
        ("k", k),
        ("v", v),
        ("attention_probe", ap),
        ("x", x.detach().clone()),
        ("learned_table", table.detach().clone()),
        ("relative_table", reltable.detach().clone()),
        ("sinusoidal", sin),
        ("absolute", absolute.detach()),
        ("learned", learned.detach()),
        ("rotary", rotated.detach()),
        ("alibi", alibi),
        ("relative", bias.detach()),
        ("relative_causal", causal.detach()),
        ("weights", attention.weights.detach()),
        ("output", attention.output.detach()),
        ("objective", objective.detach()),
        ("x_gradient", gradients[0]),
        ("learned_gradient", gradients[1]),
        ("relative_gradient", gradients[2]),
        ("rotary_cached", cached),
        ("rotary_dot0", dot0),
        ("rotary_dot7", dot7),
    ]
    arrays.append(
        (
            "bucket_grid",
            nn.cast(
                nn.from_vector(
                    [
                        bucket(r, bool(b), 16, 32)
                        for b in range(2)
                        for r in range(-256, 257)
                    ],
                    [2, 513],
                ),
                nn.DType.Int32,
            ),
        )
    )
    for heads in [1, 2, 3, 4, 6, 8]:
        values = slopes(heads)
        arrays.append(("slopes_" + str(heads), nn.from_vector(values, [heads])))
        if heads & (heads - 1) == 0:
            assert (
                max(abs(a - b) for a, b in zip(nn.alibi_slopes(heads), values)) < 1e-7
            )

    def difference(a, b):
        return max(abs(u - v) for u, v in zip(a.values(), b.values()))

    assert difference(cached, nn.slice(rotated.detach(), [0, 2, 0], [2, 3, 8])) < 1e-6
    assert difference(dot0, dot7) < 1e-5
    for bidirectional in [True, False]:
        assert (
            difference(
                nn.relative_position_buckets(5, 5, bidirectional, 16, 32),
                buckets_grid(5, 5, bidirectional),
            )
            == 0
        )
    count = 0
    error = 0.0
    epsilon = nn.scalar(0.002).item()
    with nn.no_grad():
        for value, gradient in zip([x, table, reltable], gradients):
            flat = nn.reshape(value, [-1])
            expected = gradient.values()
            for i in range(value.numel()):
                cell = nn.select(flat, 0, i)
                old = cell.item()
                cell.fill_(old + epsilon)
                plus = forward()[-1].item()
                cell.fill_(old - epsilon)
                minus = forward()[-1].item()
                cell.fill_(old)
                error = max(
                    error, abs(nn.scalar(plus - minus).item() / 0.004 - expected[i])
                )
                count += 1
    assert count == 200 and error < 0.006
    arrays.append(("native_fd", nn.from_vector_double([count, error], [2])))
    directory = Path(args["out"])
    directory.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(directory / "positions.npz"), arrays)
    guards = [
        lambda: sinusoidal(5, 0),
        lambda: sinusoidal(5, 3),
        lambda: rotary(nn.slice(x.detach(), [0, 0, 0], [2, 5, 7])),
        lambda: rotary(x.detach(), -1),
        lambda: slopes(0),
        lambda: bucket(1, True, 5),
        lambda: bucket(1, True, 16, 4),
    ]
    for check in guards:
        try:
            check()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid positional input accepted")
    print(
        f"Five position mechanisms (tables from {source}): {count} native FD coordinates, max error {error:.9g}; seven invalid inputs rejected PASS"
    )
    fixture = Path(args["tf"]) / ("positions.npz")
    if fixture.exists():
        reference = dict(nn.load_npz(str(fixture)))
        ours = dict(arrays)
        inputs = [
            "x",
            "probe",
            "q",
            "k",
            "v",
            "attention_probe",
            "learned_table",
            "relative_table",
        ]

        def difference(a, b):
            if tuple(a.shape()) != tuple(b.shape()):
                return math.inf
            return max((abs(x - y) for x, y in zip(a.values(), b.values())), default=0)

        input_error = max(
            difference(ours[key], reference[key]) for key in inputs if key in reference
        )
        if input_error == 0:
            worst = 0
            for key, tensor in arrays:
                if key in reference and key not in inputs + ["native_fd", "metadata"]:
                    delta = difference(tensor, reference[key])
                    worst = max(worst, delta)
                    if args["verbose"]:
                        print(key, "max difference vs TF:", delta)
            assert worst <= 1e-5
            print("TF comparison with identical inputs, max error:", worst)
        else:
            print("TF comparison skipped: input random streams differ by", input_error)


def main(arguments=None):
    args = options(
        arguments,
        dict(
            out=str(output_root(False) / "transformer_positions"),
            tf=str(
                Path.home()
                / "projects/tflow_examples/output/models/transformer_positions"
            ),
            init="auto",
            verbose=0,
            epochs=5,
            full=False,
        ),
    )
    if args["init"] not in ["auto", "tf", "nn"]:
        raise ValueError("init must be auto, tf or nn")
    nn.set_num_threads(2)
    exercise(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
