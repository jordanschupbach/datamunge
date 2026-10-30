# Native Python NN example. Padding/causal masks, cache equivalence and gradient isolation.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/5_attention_and_transformers/5.1_attention_fundamentals/attention_masks.py
"""Four masked attention cases with complete derivatives and independent finite differences."""
import math
from pathlib import Path
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


def allowed_mask(kl, ql, queries, keys, causal=False, offset=0):
    if queries < 1 or keys < 1 or offset < 0:
        raise ValueError("positive attention sizes and nonnegative offset required")
    if kl.dim() != 1 or ql.dim() != 1 or kl.size(0) != ql.size(0):
        raise ValueError("lengths must be vectors of the batch size")
    flags = [
        int(t < ql.at([n]) and j < kl.at([n]) and (not causal or j <= t + offset))
        for n in range(kl.size(0))
        for t in range(queries)
        for j in range(keys)
    ]
    return nn.cast(nn.from_vector(flags, [kl.size(0), queries, keys]), nn.DType.Bool)


def exercise_case(args, name, queries, causal, offset, probe_index, seed):
    keys = 5
    if name == "padding_cross":
        klv, qlv = [4, 2, 0], [2, 3, 1]
    elif name == "causal_square":
        klv, qlv = [5, 3, 0], [5, 3, 0]
    elif queries == 1:
        klv, qlv = [5, 3, 0], [1, 1, 1]
    else:
        klv, qlv = [5, 4, 0], [2, 1, 2]
    kl = nn.cast(nn.from_vector(klv), nn.DType.Int32)
    ql = nn.cast(nn.from_vector(qlv), nn.DType.Int32)
    rng = nn.Mt19937(seed)

    def random(shape, scale=1):
        return nn.mul(
            nn.from_vector(rng.normal32_values(math.prod(shape)), shape), scale
        )

    q, k, v, probe = (
        random([3, queries, 4]),
        random([3, keys, 4]),
        random([3, keys, 3]),
        random([3, queries, 3], 0.2),
    )
    if probe_index >= 0:
        keep = nn.expand_dims(nn.equal(nn.arange(0, queries), probe_index), 1)
        probe = nn.where(keep, probe, nn.zeros_like(probe))
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    allowed = allowed_mask(kl, ql, queries, keys, causal, offset)
    config = nn.AttentionOptions()
    config.mask = allowed
    config.scale = 0.5
    config.need_weights = True
    config.impl = nn.AttentionImpl.Plain

    def forward():
        return nn.scaled_dot_product_attention(q, k, v, config)

    result = forward()
    objective = nn.sum(nn.mul(result.output, probe))
    gradients = nn.grad(objective, [q, k, v])
    arrays = [
        ("q", q.detach()),
        ("k", k.detach()),
        ("v", v.detach()),
        ("probe", probe),
        ("key_lengths", kl),
        ("query_lengths", ql),
        ("output", result.output.detach()),
        ("weights", result.weights.detach()),
        ("allowed", allowed),
        ("objective", objective.detach()),
        ("q_gradient", gradients[0]),
        ("k_gradient", gradients[1]),
        ("v_gradient", gradients[2]),
    ]
    for n in range(3):
        for j in range(keys):
            seen = any(
                allowed.at([n, t, j]) and probe.at([n, t, 0]) != 0
                for t in range(queries)
            )
            if not seen:
                assert all(gradients[1].at([n, j, c]) == 0 for c in range(4))
                assert all(gradients[2].at([n, j, c]) == 0 for c in range(3))
    if offset > 0:
        values = [0.0] * (3 * keys * 4)
        for n in range(3):
            for t in range(queries):
                for j in range(4):
                    values[(n * keys + offset + t) * 4 + j] = q.at([n, t, j])
        full_q = nn.from_vector(values, [3, keys, 4])
        full_ql = nn.cast(nn.from_vector([offset + x for x in qlv]), nn.DType.Int32)
        full_config = nn.AttentionOptions()
        full_config.mask = allowed_mask(kl, full_ql, keys, keys, True)
        full_config.scale = 0.5
        full_config.need_weights = True
        full_config.impl = nn.AttentionImpl.Plain
        full = nn.scaled_dot_product_attention(full_q, k, v, full_config).output
        arrays += [
            ("full_q", full_q),
            ("full_query_lengths", full_ql),
            ("full_output", full.detach()),
        ]
        assert (
            max(
                abs(result.output.at([n, t, j]) - full.at([n, offset + t, j]))
                for n in range(3)
                for t in range(queries)
                for j in range(3)
            )
            < 2e-6
        )
    error = 0
    count = 0
    epsilon = nn.scalar(0.002).item()
    with nn.no_grad():
        for value, gradient in zip([q, k, v], gradients):
            flat = nn.reshape(value, [-1])
            expected = list(gradient.values())
            for i in range(6):
                coordinate = i * (value.numel() - 1) // 5
                cell = nn.select(flat, 0, coordinate)
                old = cell.item()
                cell.fill_(old + epsilon)
                plus = nn.sum(nn.mul(forward().output, probe)).item()
                cell.fill_(old - epsilon)
                minus = nn.sum(nn.mul(forward().output, probe)).item()
                cell.fill_(old)
                numeric = nn.scalar(plus - minus).item() / 0.004
                error = max(error, abs(numeric - expected[coordinate]))
                count += 1
    assert error < 5e-4 and count == 18
    arrays.append(
        (
            "metadata",
            nn.from_vector_double(
                [float(causal), offset, probe_index, count, error], [5]
            ),
        )
    )
    directory = Path(args["out"])
    directory.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(directory / (name + ".npz")), arrays)
    print(
        f"{name}: queries={queries}, keys=5, offset={offset}, native FD={count}, max error={error:.9g} PASS"
    )
    fixture = Path(args["tf"]) / (name + ".npz")
    if fixture.exists():
        reference = dict(nn.load_npz(str(fixture)))
        ours = dict(arrays)
        inputs = ["q", "k", "v", "probe"]

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
            out=str(output_root(False) / "attention_masks"),
            tf=str(
                Path.home() / "projects/tflow_examples/output/models/attention_masks"
            ),
            verbose=0,
            epochs=5,
            full=False,
        ),
    )
    nn.set_num_threads(2)
    for entry in [
        ("padding_cross", 3, False, 0, -1, 45701),
        ("causal_square", 5, True, 0, 2, 45703),
        ("cached_one", 1, True, 4, -1, 45705),
        ("cached_chunk", 2, True, 3, 0, 45707),
    ]:
        exercise_case(args, *entry)
    lengths = nn.cast(nn.from_vector([1]), nn.DType.Int32)
    for queries, keys, offset in [(0, 5, 0), (3, 0, 0), (3, 5, -1)]:
        try:
            allowed_mask(lengths, lengths, queries, keys, True, offset)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid attention size or offset accepted")
    print("Padding/causal/query masks and full/cache equivalence native PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
