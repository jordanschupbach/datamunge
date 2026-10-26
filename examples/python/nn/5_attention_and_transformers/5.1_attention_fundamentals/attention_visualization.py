# Native Python NN example. Attention maps and measured key-removal interventions.
"""Compare attention concentration with actual effects, including identical values."""
import math
from pathlib import Path
from pydatamunge import nn
from pydatamunge.nn_workflows import options, output_root


def exercise_case(args, equal_values):
    rng = nn.Mt19937(45901)

    def random(shape):
        return nn.from_vector(rng.normal32_values(math.prod(shape)), shape)

    q, k, v = random([2, 4, 4]), random([2, 5, 4]), random([2, 5, 3])
    if equal_values:
        v = nn.from_vector([-1.0, 0.0, 1.0] * 10, [2, 5, 3])
    keep = [
        int((t < 3 and j <= t + 1) or (h == 1 and t == 3 and j == 0))
        for h in range(2)
        for t in range(4)
        for j in range(5)
    ]

    def mask(flags):
        return nn.cast(nn.from_vector(flags, [2, 4, 5]), nn.DType.Bool)

    def forward(flags):
        config = nn.AttentionOptions()
        config.mask = mask(flags)
        config.scale = 0.5
        config.need_weights = True
        config.impl = nn.AttentionImpl.Plain
        return nn.scaled_dot_product_attention(q, k, v, config)

    result = forward(keep)
    entropy = nn.neg(
        nn.sum(
            nn.mul(
                result.weights, nn.log(nn.maximum(result.weights, nn.scalar(1e-30)))
            ),
            [-1],
        )
    )
    support = nn.mul(
        nn.exp(entropy),
        nn.cast(nn.greater(nn.sum(result.weights, [-1]), 0.0), nn.DType.Float32),
    )
    maxw = nn.max(result.weights, [-1])
    arrays = [
        ("q", q),
        ("k", k),
        ("v", v),
        ("allowed", mask(keep)),
        ("weights", result.weights),
        ("output", result.output),
        ("entropy", entropy),
        ("effective_support", support),
        ("max_weight", maxw),
    ]
    effects = [0.0] * 40
    for r in range(5):
        flags = [int(flag and index % 5 != r) for index, flag in enumerate(keep)]
        without = forward(flags).output
        arrays.append(("without_" + str(r), without))
        # The canonical diagnostic rounds each difference to Float32 before its double norm.
        difference = nn.sub(result.output, without).values()
        for h in range(2):
            for t in range(4):
                start = (h * 4 + t) * 3
                effects[(h * 4 + t) * 5 + r] = math.sqrt(
                    sum(x * x for x in difference[start : start + 3])
                )
    arrays.append(("removal_effect", nn.from_vector(effects, [2, 4, 5])))
    name = "equal_values" if equal_values else "distinct_values"
    directory = Path(args["out"])
    directory.mkdir(parents=True, exist_ok=True)
    nn.save_npz(str(directory / (name + ".npz")), arrays)
    print(name + ":")
    for h in range(2):
        print("  head", h, "weights | entropy support max weight | key removal effects")
        for t in range(4):
            weights = [result.weights.at([h, t, j]) for j in range(5)]
            stats = [entropy.at([h, t]), support.at([h, t]), maxw.at([h, t])]
            removal = effects[(h * 4 + t) * 5 : (h * 4 + t + 1) * 5]
            print(
                "   q" + str(t) + ":",
                " ".join(f"{x:.3f}" for x in weights),
                "|",
                " ".join(f"{x:.3f}" for x in stats),
                "|",
                " ".join(f"{x:.3f}" for x in removal),
            )
    assert support.at([0, 3]) == 0 and abs(support.at([1, 3]) - 1) < 1e-6
    if equal_values:
        assert (
            max(
                effects[(h * 4 + t) * 5 + r]
                for h in range(2)
                for t in range(3)
                for r in range(5)
            )
            < 2e-6
        )
    print(
        name
        + ": maps, entropy, effective support and five key-removal interventions PASS"
    )
    fixture = Path(args["tf"]) / (name + ".npz")
    if fixture.exists():
        reference = dict(nn.load_npz(str(fixture)))
        ours = dict(arrays)
        inputs = ["q", "k", "v"]

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
            out=str(output_root(False) / "attention_visualization"),
            tf=str(
                Path.home()
                / "projects/tflow_examples/output/models/attention_visualization"
            ),
            verbose=0,
            epochs=5,
            full=False,
        ),
    )
    nn.set_num_threads(2)
    exercise_case(args, False)
    exercise_case(args, True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
