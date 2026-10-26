# Native Python NN example. Multi-head projections and shared-input derivatives.
"""Four multi-head cases, parameter/input finite differences and primitive equivalence."""
import math
from pathlib import Path
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


def exercise_case(args, name, heads, self_attention, strict, identity, seed):
    tq, tk = (4, 4) if self_attention else (3, 5)
    store = nn.ParameterStore(seed)
    rng = nn.Mt19937(seed)

    def random(shape, scale=1.0):
        return nn.mul(
            nn.from_vector(rng.normal32_values(math.prod(shape)), shape), scale
        )

    xq = random([2, tq, 8])
    xkv = xq if self_attention else random([2, tk, 8])
    probe = random([2, tq, 8], 0.2)
    xq.requires_grad_()
    if not self_attention:
        xkv.requires_grad_()
    projections = ["Wq", "Wk", "Wv", "Wo"]
    fixture = Path(args["tf"]) / (name + ".npz")
    source = "nn"
    reference = None
    if not identity and args["init"] != "nn":
        if fixture.exists():
            reference = dict(nn.load_npz(str(fixture)))
            source = "TF"
        elif args["init"] == "tf":
            raise FileNotFoundError(fixture)
    if identity or reference is not None:
        for key in projections:
            value = nn.eye(8) if identity else reference["mha/" + key]
            assert tuple(value.shape()) == (8, 8)
            parameter = store.get("mha/" + key, [8, 8], nn.init.constant(0.0))
            with nn.no_grad():
                parameter.copy_(value)
        if identity:
            source = "identity"
    klv, qlv = ([4, 3], [4, 3]) if self_attention else ([4, 0], [2, 3])
    kl = nn.cast(nn.from_vector(klv), nn.DType.Int32)
    ql = nn.cast(nn.from_vector(qlv), nn.DType.Int32)
    mask = nn.Tensor()
    if strict:
        flags = [
            int(t < qlv[n] and j < klv[n] and j <= t)
            for n in range(2)
            for t in range(tq)
            for j in range(tk)
        ]
        mask = nn.cast(nn.from_vector(flags, [2, 1, tq, tk]), nn.DType.Bool)
    config = nn.MultiHeadOptions()
    config.mask = mask
    config.need_weights = True
    config.impl = nn.AttentionImpl.Plain

    def forward():
        return nn.multi_head_attention(store, "mha", xq, xkv, heads, config)

    result = forward()
    objective = nn.sum(nn.mul(result.output, probe))
    leaves = (
        [store.at("mha/" + key) for key in projections]
        + [xq]
        + ([] if self_attention else [xkv])
    )
    gradients = list(nn.grad(objective, leaves))
    arrays = [
        ("xq", xq.detach()),
        ("xkv", xkv.detach()),
        ("probe", probe),
        ("output", result.output.detach()),
        ("weights", result.weights.detach()),
        ("objective", objective.detach()),
    ]
    arrays.extend(
        ("gradient_mha/" + key, gradients[i]) for i, key in enumerate(projections)
    )
    arrays.append(("xq_gradient", gradients[4]))
    if not self_attention:
        arrays.append(("xkv_gradient", gradients[5]))
    if strict:
        arrays.extend([("allowed", mask), ("key_lengths", kl), ("query_lengths", ql)])
    if identity:
        primitive_config = nn.AttentionOptions()
        primitive_config.scale = 1 / math.sqrt(8)
        primitive_config.impl = nn.AttentionImpl.Plain
        primitive = nn.scaled_dot_product_attention(
            xq, xkv, xkv, primitive_config
        ).output.detach()
        arrays.append(("primitive", primitive))
        assert (
            max(abs(a - b) for a, b in zip(result.output.values(), primitive.values()))
            < 2e-6
        )
    arrays.extend(
        ("mha/" + key, leaves[i].detach().clone()) for i, key in enumerate(projections)
    )

    def finite_differences():
        error = 0.0
        count = 0
        epsilon = nn.scalar(0.002).item()
        for leaf, (value, gradient) in enumerate(zip(leaves, gradients)):
            samples = 3 if leaf < 4 else 6
            flat = nn.reshape(value, [-1])
            expected = gradient.values()
            for i in range(samples):
                coordinate = i * (value.numel() - 1) // (samples - 1)
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
        return error, count

    with nn.no_grad():
        error, count = finite_differences()
    assert error < 2e-3 and store.count() == 256 and store.size() == 4
    arrays.append(
        (
            "metadata",
            nn.from_vector_double([heads, int(self_attention), count, error], [4]),
        )
    )
    directory = Path(args["out"])
    directory.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(directory / (name + ".npz")), arrays)
    print(
        f"{name}: heads={heads}, shared input={self_attention}, weights={source}, native FD={count}, max error={error:.9g} PASS"
    )
    fixture = Path(args["tf"]) / (name + ".npz")
    if fixture.exists():
        reference = dict(nn.load_npz(str(fixture)))
        ours = dict(arrays)
        inputs = ["xq", "xkv", "probe"] + ["mha/" + key for key in projections]

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
            out=str(output_root(False) / "multi_head_attention"),
            tf=str(
                Path.home()
                / "projects/tflow_examples/output/models/multi_head_attention"
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
    for case in [
        ("cross_two", 2, False, False, False, 45601),
        ("masked_two", 2, False, True, False, 45603),
        ("self_four", 4, True, True, False, 45605),
        ("identity_one", 1, False, False, True, 45607),
    ]:
        exercise_case(args, *case)
    for width, heads in [(8, 0), (8, 3), (0, 1)]:
        try:
            x = nn.ones([1, 1, width])
            nn.multi_head_attention(nn.ParameterStore(), "bad", x, x, heads)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid model/head dimensions accepted")
    print("Multi-head projection/input checks and dimension guards PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
