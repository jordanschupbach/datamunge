# Native Python NN example. Receptive-field geometry and trained CNN feature derivatives in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.1_fundamentals/receptive_fields.py
"""Receptive fields and trained CNN feature maps, with numerical validation.

Port of tflow's receptive_fields. Preparation: just python-example cnn_mnist
(writes <output root>/models/cnn_mnist/best.npz), then run this example;
tflow's python3 scripts/plot_cnn_features.py draws the PNG/PDF figures from
features.npz (not ported). MNIST comes from tflow's dataset cache (data=PATH).
Restore the exact shared MNIST CNN, verify its test accuracy, and export
intermediate activations for the first test example of every digit.
Track receptive-field width, jump and center through conv/pool layers:
width 1 -> 3 -> 4 -> 8 -> 10; final jump 4 and first center 2.
Check the full predicted support with positive convolutions/average pools,
including boundary clipping. The trained model's input gradient for one
pooled feature lies inside that box; ReLU and max pooling may reduce it.
A finite difference checks an active derivative, and an outside pixel
perturbation leaves the selected feature unchanged. This local derivative
is an illustration for one image/unit, not a dataset-level explanation.
Data, layer geometry: output/nn/visualizations/receptive_fields/.
Options: data=PATH checkpoint=PATH threads=N.
"""
import math
import os
import struct
from pathlib import Path

from pydatamunge.nn_components import dataset_assets
from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper, options, output_root

checks = helper("checks")
cnn_baselines = helper("cnn_baselines")
mnist = helper("mnist")
receptive_field = helper("receptive_field")


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def g(value):
    return f"{value:.6g}"


def geometry():
    path = checks.output_dir("visualizations/receptive_fields")
    field = receptive_field.ReceptiveField()
    with (path / "geometry.tsv").open("w") as table:
        table.write("stage\twidth\tjump\tfirst_center\n")
        table.write("input\t1\t1\t0.5\n")
        for index, name in enumerate(("conv1", "pool1", "conv2", "pool2")):
            pool = index % 2 == 1
            field = field.after(2 if pool else 3, 2 if pool else 1, 0 if pool else 1)
            table.write(f"{name}\t{field.width}\t{field.jump}\t{g(field.start)}\n")
    checks.require(field.width == 10 and field.jump == 4 and field.start == 2, "final receptive-field geometry")
    # The width of a dilated layer is an enclosing box, not a count of samples.
    checks.require(receptive_field.ReceptiveField().after(3, 1, 2, 2).width == 5, "dilation expands effective width")
    return field


def support_checks(field):
    kernel = nn.ones([3, 3, 1, 1])
    for test in range(2):
        coordinate = 0 if test == 0 else 3
        x = nn.zeros([1, 28, 28, 1])
        x.requires_grad_()
        first = nn.avg_pool2d(nn.conv2d(x, kernel, nn.conv_options(1, nn.Padding.Same)), 2, 2)
        second = nn.avg_pool2d(nn.conv2d(first, kernel, nn.conv_options(1, nn.Padding.Same)), 2, 2)
        feature = nn.sum(nn.slice(second, [0, coordinate, coordinate, 0], [1, 1, 1, 1]))
        derivative = nn.grad(feature, [x])[0].values()
        lo, hi = field.bounds(coordinate)
        pixels_in_support = 0
        for y in range(28):
            for xx in range(28):
                inside = lo <= y < hi and lo <= xx < hi
                checks.require((derivative[y * 28 + xx] > 0) == inside, "analytic and numerical support agree")
                pixels_in_support += inside
        print(f"  positive linear network: {'corner' if test == 0 else 'center'} support={pixels_in_support} pixels")


def logits_of(store, x, training):
    return cnn_baselines.mnist_cnn(store, x).logits


def main(arguments=None):
    full = os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"
    args = options(arguments, dict(data=str(data_root() / "mnist"),
                                   checkpoint=str(output_root(full) / "models/cnn_mnist/best.npz"),
                                   threads=0, full=full))
    os.environ["DATAMUNGE_EXAMPLES_FULL"] = "1" if args["full"] else "0"
    if args["threads"]:
        nn.set_num_threads(args["threads"])
    field = geometry()
    support_checks(field)
    checks.require(Path(args["checkpoint"]).exists(), "run cnn_mnist first to create trained checkpoint")
    dataset = dataset_assets.LoadMnist(args["data"], "test")
    count = int(dataset.images.shape()[0])
    pixels = nn.reshape(dataset.images, [count, 784])

    store = nn.ParameterStore(17)
    with nn.no_grad():
        logits_of(store, nn.slice_axis(pixels, 0, 0, 1), False)
    store.load(args["checkpoint"])
    evaluation = mnist.evaluate(store, logits_of, pixels, dataset.labels, list(range(count)))
    checks.training_quality(evaluation.accuracy.value() >= .98, "restored shared CNN preserves test performance")
    print(f"  restored checkpoint test accuracy={g(evaluation.accuracy.value())}")

    labels = [int(v) for v in dataset.labels.values()]
    rows = [count] * 10
    for i, label in enumerate(labels):
        if rows[label] == count:
            rows[label] = i
    for row in rows:
        checks.require(row < count, "one example of each digit")
    selected_channel = 0
    with nn.no_grad():
        x = nn.take_rows(pixels, rows)
        network = cnn_baselines.mnist_cnn(store, x)
        arrays = [("conv1", network.conv1), ("conv2", network.conv2), ("images", network.image),
                  ("labels", nn.take_rows(dataset.labels, rows)), ("pool2", network.pool2),
                  ("predictions", nn.argmax(network.logits, 1))]
        for c in range(1, 32):
            if network.pool2.at([0, 3, 3, c]) > network.pool2.at([0, 3, 3, selected_channel]):
                selected_channel = c
        checks.require(network.pool2.at([0, 3, 3, selected_channel]) > 0, "active selected feature")

    # Differentiate in Float64 so shrinking the finite-difference interval does
    # not amplify Float32 cancellation. A tiny deterministic texture breaks
    # blank-background max-pool ties in a lightly trained quick model.
    derivative_store = nn.ParameterStore(17, nn.DType.Float64)
    for name in store.names():
        derivative_store.set(name, store.at(name).to(nn.DType.Float64))
    probe = list(nn.take_rows(pixels, [rows[0]]).values())
    if not checks.full():
        # The C++ compiler contracts both multiply-adds into fused multiply-adds (clang -ffp-contract=on).
        probe = [math.fma(.0001, math.sin(math.fma(.37, i, .1)), value) for i, value in enumerate(probe)]
    single = nn.from_vector_double(probe, [1, 784])
    arrays.append(("gradient_probe", single))

    def evaluate(inputs, derivative):
        x = inputs.clone()
        if derivative:
            x.requires_grad_()
        network = cnn_baselines.mnist_cnn(derivative_store, x)
        selected = nn.sum(nn.slice(network.pool2, [0, 3, 3, selected_channel], [1, 1, 1, 1]))
        gradient = nn.grad(selected, [x])[0] if derivative else None
        return selected.item(), gradient

    reference = evaluate(single, True)
    arrays.append(("gradient", reference[1]))
    dx = reference[1].values()
    support = [0.0] * (28 * 28)
    lo, hi = field.bounds(3)
    active = strongest = 0
    for py in range(28):
        for px in range(28):
            i = py * 28 + px
            derivative = float32(dx[i])
            inside = lo <= py < hi and lo <= px < hi
            support[i] = float(inside)
            if not inside:
                checks.near(derivative, 0, 0, "gradient stays inside theoretical receptive field")
            active += derivative != 0
            if abs(derivative) > abs(dx[strongest]):
                strongest = i
    checks.require(active > 0, "nonzero feature gradient")
    with nn.no_grad():
        base = single.values()
        changed = list(base)
        changed[0] += .5
        checks.near(evaluate(nn.from_vector_double(changed, [1, 784]), False)[0], reference[0], 0,
                    "outside perturbation cannot affect feature")
        # A short-trained checkpoint can place a ReLU/max-pool boundary inside
        # the original .001 interval. Shrink the interval until both one-sided
        # slopes agree; retain the original derivative tolerance.
        difference, stable = 0.0, False
        for h in (.001, .0002, .0001, .00005, .00002, .000001, .00000001):
            plus, minus = list(base), list(base)
            plus[strongest] += h
            minus[strongest] -= h
            forward = (evaluate(nn.from_vector_double(plus, [1, 784]), False)[0] - reference[0]) / h
            backward = (reference[0] - evaluate(nn.from_vector_double(minus, [1, 784]), False)[0]) / h
            if abs(forward - backward) <= .001:
                difference, stable = (forward + backward) / 2, True
                break
        checks.require(stable, "finite-difference interval stays within one active branch")
        checks.near(difference, dx[strongest], .001, "active derivative finite difference")
    arrays.append(("support", nn.from_vector(support, [28, 28])))
    arrays.append(("unit", nn.cast(nn.from_vector_int64([3, 3, selected_channel], [3]), nn.DType.Int32)))
    out = checks.output_dir("visualizations/receptive_fields") / "features.npz"
    nn.save_npz(str(out), arrays)
    print(f"  pool2[3,3,{selected_channel}]: theoretical box=[{lo},{hi}) squared; nonzero input derivatives={active}/100")
    print(f"  finite difference={g(difference)}; analytic={g(dx[strongest])}")
    print(f"  export: {out.resolve()}")
    print("All receptive-field checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
