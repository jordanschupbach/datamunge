# Native Python NN example. Scaled attention outputs and complete Q/K/V derivative fixtures.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/5_attention_and_transformers/5.1_attention_fundamentals/scaled_dot_attention.py
"""Cross, masked-head, single-key and additive-bias attention with finite differences."""
import math
from pathlib import Path
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


def exercise_case(args, name, heads, strict, one_key, seed):
    keys = 1 if one_key else 5
    qs = [2, 2, 3, 4] if heads else [2, 3, 4]
    ks = [2, 2, keys, 4] if heads else [2, keys, 4]
    vs = [2, 2, keys, 3] if heads else [2, keys, 3]
    os = [2, 2, 3, 3] if heads else [2, 3, 3]
    rng = nn.Mt19937(seed)

    def random(shape, scale=1):
        return nn.mul(
            nn.from_vector(rng.normal32_values(math.prod(shape)), shape), scale
        )

    q, k, v, probe = random(qs), random(ks), random(vs), random(os, 0.2)
    q.requires_grad_()
    k.requires_grad_()
    v.requires_grad_()
    mask, bias, allowed = nn.Tensor(), nn.Tensor(), nn.Tensor()
    kl = nn.cast(nn.from_vector([4, 0]), nn.DType.Int32)
    ql = nn.cast(nn.from_vector([2, 3]), nn.DType.Int32)
    if name == "biased":
        bias = nn.reshape(
            nn.mul(nn.sin(nn.mul(nn.arange(0, 3 * keys), 0.7)), 0.3), [3, keys]
        )
        mask = bias
    if strict:
        flags = [
            int(t < ql.at([n]) and j < kl.at([n]) and j <= t)
            for n in range(2)
            for t in range(3)
            for j in range(keys)
        ]
        allowed = nn.cast(nn.from_vector(flags, [2, 3, keys]), nn.DType.Bool)
        if heads:
            allowed = nn.expand_dims(allowed, 1)
        mask = allowed
    config = nn.AttentionOptions()
    config.mask = mask
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
        ("output", result.output.detach()),
        ("weights", result.weights.detach()),
        ("objective", objective.detach()),
        ("q_gradient", gradients[0]),
        ("k_gradient", gradients[1]),
        ("v_gradient", gradients[2]),
    ]
    if strict:
        arrays += [("allowed", allowed), ("key_lengths", kl), ("query_lengths", ql)]
    if bias.defined():
        arrays.append(("bias", bias))
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
    if strict:
        values = list(result.weights.values())
        assert all(x == 0 for x in values[len(values) // 2 :])
        for tensor in [result.output, gradients[1], gradients[2]]:
            values = list(tensor.values())
            assert all(x == 0 for x in values[len(values) // 2 :])
    if one_key:
        assert all(abs(x - 1) < 1e-7 for x in result.weights.values())
    arrays.append(("native_fd", nn.from_vector_double([count, error], [2])))
    directory = Path(args["out"])
    directory.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(directory / (name + ".npz")), arrays)
    print(f"{name}: Q{qs} K{ks} V{vs}, native FD={count}, max error={error:.9g} PASS")
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
            out=str(output_root(False) / "scaled_dot_attention"),
            tf=str(
                Path.home()
                / "projects/tflow_examples/output/models/scaled_dot_attention"
            ),
            verbose=0,
            epochs=5,
            full=False,
        ),
    )
    nn.set_num_threads(2)
    for case in [
        ("cross", False, False, False, 45501),
        ("masked_heads", True, True, False, 45503),
        ("one_key", False, False, True, 45505),
        ("biased", False, False, False, 45507),
    ]:
        exercise_case(args, *case)
    try:
        invalid = nn.ones([1, 1, 0])
        nn.scaled_dot_product_attention(
            invalid, invalid, nn.ones([1, 1, 1]), nn.Tensor()
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("zero key width accepted")
    print("Scaled dot-product: all native derivative checks and dimension guard PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
