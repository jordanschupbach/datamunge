# Native Python NN example. Atrous convolutions: exact dilation, gridding and learning distant context.
"""CMD: just python-example dilated_convolutions
Atrous convolutions: exact dilation, gridding and learning distant context.

Port of tflow's dilated_convolutions. A 3x3 filter at rate d has effective
width 2d+1 and nine learned taps per channel pair. Compare zero-expanded
filters with native dilated nn.conv2d at rates 1,2,3 under SAME/VALID
padding and check input/filter derivatives numerically. The expansion
supplied CPU gradients where TF's native dilation backprop is unsupported;
nn's dilated backprop is native, so this port also checks that the native
and expanded-filter gradients agree. The reference path pays for a larger
dense convolution.

Compare three stacks: rates [1,1,1], [2,2,2], [1,2,4]. Positive-kernel input
gradients reveal dense 7x7, sparse 13x13 and dense 15x15 dependency patterns.
Equal rates leave gridding holes despite a larger enclosing field.
Finally fit a nine-tap, rate-2 teacher using nine-parameter rate-1/rate-2
students on independent random images. Only the matched receptive field can
recover the teacher; compare held-out MSE, never select on test examples.
Inputs come from the same std::mt19937 streams as the C++ (bit-identical).
Support arrays: output/nn/python/<mode>/visualizations/dilated_convolutions/supports.npz.

C++ reference: examples/cpp/nn/1_convolutional_neural_networks/1.1_fundamentals/dilated_convolutions.cpp
"""
import math
import struct
from types import SimpleNamespace

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
dilated = helper("dilated")
receptive_field = helper("receptive_field")


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def g(value):
    return f"{value:.6g}"


def random_tensor(shape, rng):
    """A fresh std::normal_distribution<float>(0, .5) over the shared std::mt19937."""
    rng.reset_normal()
    return nn.from_vector(rng.normal32_values(math.prod(shape), 0.0, 0.5), shape)


def max_difference(a, b):
    checks.require(list(a.shape()) == list(b.shape()), "dilated output shape")
    return max((f32(abs(x - y)) for x, y in zip(a.contiguous().values(), b.contiguous().values())), default=0.0)


def equivalence():
    rng = nn.Mt19937(7)
    pixels = random_tensor([2, 11, 12, 2], rng)
    filter_ = random_tensor([3, 3, 2, 3], rng)
    for dilation in (1, 2, 3):
        for padding in ("SAME", "VALID"):
            pad = nn.Padding.Same if padding == "SAME" else nn.Padding.Valid

            def forward_and_gradients(a, b, derivatives):
                x, w = a.clone(), b.clone()
                x.requires_grad_()
                w.requires_grad_()
                r = SimpleNamespace()
                r.native = nn.conv2d(x, w, nn.conv_options(1, pad, dilation))
                r.portable = dilated.portable_dilated_conv2d(x, w, dilation, padding)
                r.objective = nn.mean(nn.square(r.portable))
                if derivatives:
                    r.gradients = nn.grad(r.objective, [x, w])
                    r.native_gradients = nn.grad(nn.mean(nn.square(r.native)), [x, w])
                return r

            result = forward_and_gradients(pixels, filter_, True)
            maximum = max_difference(result.native, result.portable)
            checks.near(maximum, 0, 2e-6, "native and expanded-filter dilation")
            native_gradient_error = max(max_difference(result.gradients[0], result.native_gradients[0]),
                                        max_difference(result.gradients[1], result.native_gradients[1]))
            checks.near(native_gradient_error, 0, 2e-6, "native dilated backprop equals expanded-filter gradients")
            gradient_error = 0.0
            for which in (0, 1):
                source = pixels if which == 0 else filter_
                base = list(source.values())
                analytic = result.gradients[which].values()
                for i in (0, source.numel() // 2, source.numel() - 1):
                    plus, minus = list(base), list(base)
                    plus[i] = f32(plus[i] + f32(0.002))
                    minus[i] = f32(minus[i] - f32(0.002))
                    p, m = nn.from_vector(plus, source.shape()), nn.from_vector(minus, source.shape())
                    upper = forward_and_gradients(p, filter_, False) if which == 0 else forward_and_gradients(pixels, p, False)
                    lower = forward_and_gradients(m, filter_, False) if which == 0 else forward_and_gradients(pixels, m, False)
                    finite = f32(upper.objective.item() - lower.objective.item()) / 0.004
                    error = abs(finite - analytic[i])
                    gradient_error = max(gradient_error, error)
                    checks.near(error, 0, 8e-5, "dilated finite-difference derivative")
            print(f"  rate={dilation} {padding}: forward error={g(maximum)}; derivative error={g(gradient_error)}; "
                  f"native-vs-expanded gradient error={g(native_gradient_error)}")


def gridding():
    arrays = {}
    schedules = [[1, 1, 1], [2, 2, 2], [1, 2, 4]]
    kernel = nn.ones([3, 3, 1, 1])
    for schedule, rates in enumerate(schedules):
        x = nn.zeros([1, 33, 33, 1])
        x.requires_grad_()
        h = x
        field = receptive_field.ReceptiveField()
        name = ""
        for dilation in rates:
            h = dilated.portable_dilated_conv2d(h, kernel, dilation)
            field = field.after(3, 1, dilation, dilation)
            name += str(dilation)
        center = nn.sum(nn.slice(h, [0, 16, 16, 0], [1, 1, 1, 1]))
        gradient = nn.grad(center, [x])[0]
        values = gradient.values()
        lo, hi = field.bounds(16)
        count = 0
        for y in range(33):
            for xx in range(33):
                inside = lo <= y < hi and lo <= xx < hi
                connected = values[y * 33 + xx] > 0
                expected = inside and (schedule != 1 or (y % 2 == 0 and xx % 2 == 0))
                checks.require(connected == expected, "exact dense or gridded dependency pattern")
                count += connected
        arrays["rates_" + name] = gradient
        print(f"  rates {name}: bounding width={field.width}; connected pixels={count}/{field.width * field.width}")
    directory = checks.output_dir("visualizations/dilated_convolutions")
    nn.save_npz(str(directory / "supports.npz"), sorted(arrays.items()))


def learn_context():
    parameters = nn.ParameterStore(19)
    truth = nn.from_vector([0.4, -0.7, 0.2, 0.6, 1.0, -0.3, -0.5, 0.8, 0.25], [3, 3, 1, 1])
    coefficients = truth.values()
    for dilation in (1, 2):
        parameters.get("rate" + str(dilation), [3, 3, 1, 1], nn.init.zeros())

    def losses(x):
        teacher = dilated.portable_dilated_conv2d(x, truth, 2)
        result = []
        for dilation in (1, 2):
            prediction = dilated.portable_dilated_conv2d(x, parameters.at("rate" + str(dilation)), dilation)
            result.append(nn.mean(nn.squared_difference(prediction, teacher)))
        return result

    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(parameters, 0.03, config)
    rng = nn.Mt19937(101)
    training = random_tensor([32, 13, 13, 1], rng)
    test = random_tensor([64, 13, 13, 1], rng)
    for step in range(400):
        loss = losses(training)
        optimizer.minimize(nn.add(loss[0], loss[1]))
    with nn.no_grad():
        held_out = losses(test)
        rate1, rate2 = held_out[0].item(), held_out[1].item()
    print(f"  equal nine-parameter students, held-out MSE: rate1={g(rate1)}, rate2={g(rate2)}")
    checks.require(rate1 > f32(0.2), "rate-1 student lacks independent distant context")
    checks.require(rate2 < f32(1e-8), "rate-2 student recovers teacher")
    learned = parameters.at("rate2").values()
    for i in range(9):
        checks.near(learned[i], coefficients[i], 1e-4, "learned dilated coefficient")


def main(arguments=None):
    options(arguments, dict(full=False))
    equivalence()
    gridding()
    learn_context()
    print("All dilated-convolution checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
