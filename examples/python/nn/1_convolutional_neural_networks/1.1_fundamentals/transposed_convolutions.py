# Native Python NN example. Transposed convolution, its adjoint identity and checkerboard artifacts.
"""CMD: just python-example transposed_convolutions
Transposed convolution, its adjoint identity and checkerboard artifacts.

Port of tflow's transposed_convolutions. TensorFlow implements transposed
convolution as Conv2DBackpropInput; nn.conv2d_transpose is the same
operation. It is the adjoint of a specified forward convolution, not
generally its inverse. Verify SAME/VALID output shapes against an
independent scatter-and-add loop and verify the inner-product adjoint
identity on multi-channel inputs. Check derivatives with respect to coarse
input and filter as well.

Feed a constant 8x8 image to stride-2 transposed convolution. A 3x3 all-one
kernel creates unequal interior overlap across the four stride phases.
A 4x4 all-one kernel equalizes overlap; arbitrary learned 4x4 weights can
still create phase imbalance. Nearest-neighbor resize followed by ordinary
3x3 convolution preserves constants away from padding boundaries. Measure
interior phase means/CV separately from boundary effects, and export maps.
Inputs come from the same std::mt19937(18) stream as the C++ (bit-identical).
Arrays: output/nn/python/<mode>/visualizations/transposed_convolutions/checkerboards.npz.

C++ reference: examples/cpp/nn/1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions.cpp
"""
import math
import struct
from types import SimpleNamespace

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
im2col = helper("im2col")


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def g(value):
    return f"{value:.6g}"


def random_tensor(shape, rng):
    """A fresh std::normal_distribution<float>(0, .25) over the shared std::mt19937."""
    rng.reset_normal()
    return nn.from_vector(rng.normal32_values(math.prod(shape), 0.0, 0.25), shape)


def difference(a, b):
    checks.require(list(a.shape()) == list(b.shape()), "transposed tensor shapes agree")
    return max((f32(abs(x - y)) for x, y in zip(a.contiguous().values(), b.contiguous().values())), default=0.0)


def shape_str(shape):
    return "[" + ",".join(str(int(d)) for d in shape) + "]"


def adjoint_checks():
    rng = nn.Mt19937(18)
    for padding in ("SAME", "VALID"):
        height, width = (8, 10) if padding == "SAME" else (9, 11)
        coarse = random_tensor([2, 4, 5, 2], rng)
        kernel = random_tensor([3, 3, 3, 2], rng)
        fine = random_tensor([2, height, width, 3], rng)
        geometry = im2col.PatchGeometry(fine.shape(), 3, 3, 2, 2, 1, 1, padding)
        c, k = coarse.values(), kernel.values()
        # Independent scatter-and-add reference with float32 accumulation.
        reference = [0.0] * fine.numel()
        for n in range(2):
            for y in range(4):
                for x in range(5):
                    for a in range(3):
                        for b in range(3):
                            py, px = 2 * y + a - geometry.top, 2 * x + b - geometry.left
                            if py < 0 or py >= height or px < 0 or px >= width:
                                continue
                            for out in range(3):
                                for inp in range(2):
                                    index = ((n * height + py) * width + px) * 3 + out
                                    reference[index] = f32(reference[index] + f32(
                                        c[((n * 4 + y) * 5 + x) * 2 + inp] * k[((a * 3 + b) * 3 + out) * 2 + inp]))
        conv = nn.conv_options(2, nn.Padding.Same if padding == "SAME" else nn.Padding.Valid)

        def forward_and_gradients(filter_, derivatives):
            x, w = coarse.clone(), filter_.clone()
            x.requires_grad_()
            w.requires_grad_()
            r = SimpleNamespace()
            r.transposed = nn.conv2d_transpose(x, w, fine.shape(), conv)
            r.forward = nn.conv2d(fine, w, conv)
            r.objective = nn.sum(nn.mul(r.transposed, fine))
            r.paired = nn.sum(nn.mul(r.forward, x))
            if derivatives:
                r.gradients = nn.grad(r.objective, [x, w])
            return r

        result = forward_and_gradients(kernel, True)
        scatter_error = difference(result.transposed, nn.from_vector(reference, fine.shape()))
        checks.near(scatter_error, 0, 2e-7, "transposed output equals scatter-and-add reference")
        checks.near(result.objective.item(), result.paired.item(), 2e-6, "convolution inner-product adjoint identity")
        checks.near(difference(result.forward, result.gradients[0]), 0, 1e-7,
                    "transposed input gradient is forward convolution")
        dw = result.gradients[1].values()
        for index in (0, kernel.numel() // 2, kernel.numel() - 1):
            plus, minus = list(k), list(k)
            plus[index] = f32(plus[index] + f32(0.002))
            minus[index] = f32(minus[index] - f32(0.002))
            upper = forward_and_gradients(nn.from_vector(plus, kernel.shape()), False).objective.item()
            lower = forward_and_gradients(nn.from_vector(minus, kernel.shape()), False).objective.item()
            checks.near((upper - lower) / 0.004, dw[index], 2e-4, "transposed filter finite difference")
        adjoint_error = f32(abs(result.objective.item() - result.paired.item()))
        print(f"  {padding} output={shape_str(result.transposed.shape())}; scatter error={g(scatter_error)}; "
              f"adjoint error={g(adjoint_error)}")


def checkerboards():
    coarse = nn.ones([1, 8, 8, 1])
    result = []
    for variant in range(3):
        k = 3 if variant == 0 else 4
        filter_ = [1.0] * (k * k)
        if variant == 2:
            filter_[0] = 4.0
        result.append(nn.conv2d_transpose(coarse, nn.from_vector(filter_, [k, k, 1, 1]), [1, 14 + k, 14 + k, 1],
                                          nn.conv_options(2, nn.Padding.Valid)))
    size = nn.ImageSize()
    size.height, size.width = 16, 16
    resized = nn.resize_nearest(coarse, size)
    result.append(nn.conv2d(resized, nn.ones([3, 3, 1, 1]), nn.conv_options(1, nn.Padding.Same)))
    labels = ["transpose_k3", "transpose_k4", "transpose_k4_weighted", "resize_conv"]
    arrays = {}
    for variant, value in enumerate(result):
        means, counts = [0.0] * 4, [0] * 4
        # The same 10x10 interior window in all variants, five samples of each
        # phase on each axis; it excludes every padding/overlap edge.
        values, side = value.values(), int(value.shape()[2])
        for y in range(4, 14):
            for x in range(4, 14):
                phase = 2 * (y % 2) + x % 2
                means[phase] += values[y * side + x]
                counts[phase] += 1
        average = variance = 0.0
        for i in range(4):
            means[i] /= counts[i]
            average += means[i] / 4
        for mean in means:
            variance += (mean - average) * (mean - average) / 4
        cv = math.sqrt(variance) / average
        if variant in (0, 2):
            checks.require(cv > 0.1, "unequal phases produce checkerboard response")
        else:
            checks.near(cv, 0, 0, "equal interior phases for uniform even kernel or resize-convolution")
        if variant == 0:
            for i, expected in enumerate((4, 2, 2, 1)):
                checks.near(means[i], expected, 0, "3x3 stride-2 overlap multiplicity")
        arrays[labels[variant]] = value
        print(f"  {labels[variant]}: interior phase means " + " ".join(g(m) for m in means) + f"; phase CV={g(cv)}")
    directory = checks.output_dir("visualizations/transposed_convolutions")
    nn.save_npz(str(directory / "checkerboards.npz"), sorted(arrays.items()))


def main(arguments=None):
    options(arguments, dict(full=False))
    adjoint_checks()
    checkerboards()
    print("All transposed-convolution checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
