# Native Python NN example. Depthwise separable convolution: equivalence, gradients, rank and cost.
"""CMD: just python-example depthwise_separable
Depthwise separable convolution: equivalence, gradients, rank and cost.

Port of tflow's depthwise_separable. Compose HWIM depthwise and 1x1
pointwise filters into a dense HWIO filter, with no intervening activation.
Check outputs, input gradients and both factor gradients against dense
convolution for multipliers 1/2 and strides 1/2. Check selected factor
derivatives by central finite differences.

A depth multiplier m limits each input channel's spatial-by-output kernel
matrix to rank m. SVD gives the best rank-m approximation of arbitrary
filters: quantify lost energy for m=1/2 and exact reconstruction at full
rank. This explains why fewer parameters also restrict expressiveness.
Count kernel parameters and multiply-accumulates, then time matching dense
and separable forwards on CPU after warmup (nn with two threads, as TF's
run). Reduced MACs do not guarantee a matching latency speedup; timing
depends on kernels, hardware and shape.
TF -> datamunge: DepthwiseConv2dNative is nn.depthwise_conv2d, Svd is
nn.svd, AddSymbolicGradients is nn.grad. Inputs come from the same
std::mt19937 streams as the C++ (bit-identical).

C++ reference: examples/cpp/nn/1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable.cpp
"""
import math
import struct
import time
from types import SimpleNamespace

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
separable = helper("separable")


def f32(value):
    """Rounds a double to float32, as the C++ float arithmetic does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def random_tensor(shape, rng):
    """A fresh std::normal_distribution<float>(0, .2) over the shared std::mt19937."""
    rng.reset_normal()
    return nn.from_vector(rng.normal32_values(math.prod(shape), 0.0, 0.2), shape)


def difference(a, b):
    checks.require(list(a.shape()) == list(b.shape()), "matching tensor shapes")
    return max((f32(abs(x - y)) for x, y in zip(a.contiguous().values(), b.contiguous().values())), default=0.0)


def equivalence():
    rng = nn.Mt19937(41)
    for multiplier in (1, 2):
        for stride in (1, 2):
            pixels = random_tensor([2, 9, 10, 3], rng)
            spatial = random_tensor([3, 3, 3, multiplier], rng)
            mixing = random_tensor([1, 1, 3 * multiplier, 5], rng)

            def forward_and_gradients(a, b):
                x, dw, pw = pixels.clone(), a.clone(), b.clone()
                x.requires_grad_()
                dw.requires_grad_()
                pw.requires_grad_()
                r = SimpleNamespace()
                depthwise = nn.depthwise_conv2d(x, dw, nn.conv_options(stride, nn.Padding.Same))
                r.separable = nn.conv2d(depthwise, pw, nn.conv_options(1, nn.Padding.Valid))
                r.kernel = separable.compose_separable_kernel(dw, pw, 3, 3, multiplier, 5)
                r.dense = nn.conv2d(x, r.kernel, nn.conv_options(stride, nn.Padding.Same))
                r.sep_loss = nn.mean(nn.square(r.separable))
                full_loss = nn.mean(nn.square(r.dense))
                r.sep_grad = nn.grad(r.sep_loss, [x, dw, pw])
                r.full_grad = nn.grad(full_loss, [x, dw, pw])
                return r

            result = forward_and_gradients(spatial, mixing)
            checks.near(difference(result.separable, result.dense), 0, 2e-6,
                        "separable forward equals composed dense kernel")
            gradient_error = max(difference(result.sep_grad[i], result.full_grad[i]) for i in range(3))
            checks.near(gradient_error, 0, 2e-6, "equivalent input and factor gradients")
            s, m, k = spatial.values(), mixing.values(), result.kernel.values()
            for y in range(3):
                for x in range(3):
                    for i in range(3):
                        for o in range(5):
                            expected = 0.0
                            for j in range(multiplier):
                                expected += s[((y * 3 + x) * 3 + i) * multiplier + j] * m[(i * multiplier + j) * 5 + o]
                            checks.near(k[((y * 3 + x) * 3 + i) * 5 + o], expected, 1e-7, "host filter contraction")
            for factor in (0, 1):
                value = spatial if factor == 0 else mixing
                base = list(value.values())
                for i in (0, value.numel() // 2, value.numel() - 1):
                    plus, minus = list(base), list(base)
                    plus[i] = f32(plus[i] + f32(0.001))
                    minus[i] = f32(minus[i] - f32(0.001))
                    p, q = nn.from_vector(plus, value.shape()), nn.from_vector(minus, value.shape())
                    upper = forward_and_gradients(p, mixing) if factor == 0 else forward_and_gradients(spatial, p)
                    lower = forward_and_gradients(q, mixing) if factor == 0 else forward_and_gradients(spatial, q)
                    checks.near(f32(upper.sep_loss.item() - lower.sep_loss.item()) / 0.002,
                                result.sep_grad[1 + factor].values()[i], 2e-6, "factor finite difference")
            print(f"  multiplier={multiplier}, stride={stride}: forward error="
                  f"{g(difference(result.separable, result.dense))}; gradient error={g(gradient_error)}")


def rank_restriction():
    rng = nn.Mt19937(109)
    arbitrary = random_tensor([3, 9, 5], rng)  # input channel, spatial tap, output channel
    svd = nn.svd(arbitrary)
    a, s = arbitrary.values(), svd.s.contiguous().values()
    u, v = svd.u.contiguous().values(), svd.v.contiguous().values()
    total = sum(e * e for e in a)
    previous = math.inf
    for multiplier in (1, 2, 5):
        error = tail_energy = 0.0
        for channel in range(3):
            for j in range(multiplier, 5):
                tail_energy += s[channel * 5 + j] ** 2
            for tap in range(9):
                for out in range(5):
                    approximation = 0.0
                    for m in range(multiplier):
                        # float * float * float, accumulated in double (as in the C++).
                        approximation += f32(f32(u[(channel * 9 + tap) * 5 + m] * s[channel * 5 + m])
                                             * v[(channel * 5 + out) * 5 + m])
                    error += (approximation - a[(channel * 9 + tap) * 5 + out]) ** 2
        checks.near(error, tail_energy, 2e-6, "best rank-m error equals discarded singular-value energy")
        checks.require(error < previous, "extra depth multiplier increases representational capacity")
        previous = error
        if multiplier == 1:
            checks.require(error / total > 0.3, "arbitrary dense filters are not generally rank one")
        if multiplier == 5:
            checks.require(error / total < 1e-10, "full-rank factors reconstruct any kernel")
        print(f"  multiplier={multiplier}: relative squared kernel error={g(error / total)}")


def cost_and_timing():
    height, width, inputs, outputs, kernel = 32, 32, 32, 64, 3
    dense_parameters = kernel * kernel * inputs * outputs
    separable_parameters = kernel * kernel * inputs + inputs * outputs
    rng = nn.Mt19937(204)
    pixels = random_tensor([1, height, width, inputs], rng)
    dw = random_tensor([kernel, kernel, inputs, 1], rng)
    pw = random_tensor([1, 1, inputs, outputs], rng)
    with nn.no_grad():
        composed = separable.compose_separable_kernel(dw, pw, kernel, inputs, 1, outputs)

        def full():
            return nn.conv2d(pixels, composed, nn.conv_options(1, nn.Padding.Same))

        def sep():
            return nn.conv2d(nn.depthwise_conv2d(pixels, dw, nn.conv_options(1, nn.Padding.Same)), pw,
                             nn.conv_options(1, nn.Padding.Valid))

        checks.near(difference(full(), sep()), 0, 1e-5, "benchmark functions match")
        threads = nn.get_num_threads()
        nn.set_num_threads(2)
        for variant in (0, 1):
            forward = sep if variant else full
            sink = 0.0
            for _ in range(10):
                sink += nn.mean(forward()).item()
            start = time.perf_counter()
            for _ in range(100):
                sink += nn.mean(forward()).item()
            milliseconds = (time.perf_counter() - start) * 1000 / 100
            checks.require(math.isfinite(sink), "timed forward is finite")
            parameters = separable_parameters if variant else dense_parameters
            print(f"  {'separable' if variant else 'dense'}: kernel parameters={parameters}; "
                  f"MACs={height * width * parameters}; CPU forward mean={g(milliseconds)} ms (100 runs, two threads)")
        nn.set_num_threads(threads)
    print(f"  theoretical kernel/MAC reduction={g(dense_parameters / separable_parameters)}x; "
          "measured timings are eager nn calls")


def main(arguments=None):
    options(arguments, dict(full=False))
    equivalence()
    rank_restriction()
    cost_and_timing()
    print("All depthwise-separable checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
