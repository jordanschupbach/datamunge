# Native Python NN example. Convolution variants checked against explicit reference loops.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.2_reusable_layer_library/conv_layers.py
"""1-D/2-D/3-D, depthwise, separable, transpose and dilation demonstrations."""
import itertools
import math
from pydatamunge import nn


def random_tensor(shape, rng):
    rng.reset_normal()
    return nn.from_vector([rng.normal32() for _ in range(math.prod(shape))], shape)


def configuration(stride=1, dilation=1, padding=None):
    option = nn.ConvLayerOptions()
    option.stride = stride
    option.dilation = dilation
    option.padding = nn.Padding.Same if padding is None else padding
    option.bias = False
    return option


def difference(a, b):
    assert tuple(a.shape()) == tuple(b.shape()), (tuple(a.shape()), tuple(b.shape()))
    return max(abs(x - y) for x, y in zip(a.values(), b.values()))


def reference(x, weights, stride=1, dilation=1, padding=None, depthwise=False):
    shape = list(x.shape())
    kernel = list(weights.shape())
    dims = len(shape) - 2
    sizes = shape[1:-1]
    ks = kernel[:-2]
    cin = shape[-1]
    mult = kernel[-1]
    assert len(ks) == dims
    same = padding is None or padding == nn.Padding.Same
    out = [
        (
            (size + stride - 1) // stride
            if same
            else (size - dilation * (k - 1) - 1) // stride + 1
        )
        for size, k in zip(sizes, ks)
    ]
    before = [
        max(0, (o - 1) * stride + dilation * (k - 1) + 1 - size) // 2 if same else 0
        for o, k, size in zip(out, ks, sizes)
    ]
    cout = cin * mult if depthwise else mult
    values = []
    for n in range(shape[0]):
        for position in itertools.product(*(range(size) for size in out)):
            for channel in range(cout):
                total = 0
                for offset in itertools.product(*(range(k) for k in ks)):
                    source = [
                        p * stride + a * dilation - b
                        for p, a, b in zip(position, offset, before)
                    ]
                    if any(i < 0 or i >= size for i, size in zip(source, sizes)):
                        continue
                    if depthwise:
                        ci, m = divmod(channel, mult)
                        total += x.at([n, *source, ci]) * weights.at([*offset, ci, m])
                    else:
                        for ci in range(cin):
                            total += x.at([n, *source, ci]) * weights.at(
                                [*offset, ci, channel]
                            )
                values.append(total)
    return nn.from_vector(values, [shape[0], *out, cout])


def output_sizes():
    expected = [5, 3, 3, 2, 7, 7, 4, 3]
    row = 0
    for padding in [nn.Padding.Valid, nn.Padding.Same]:
        for stride in [1, 2, 3]:
            for dilation in [1, 2]:
                if stride > 1 and dilation > 1:
                    continue
                p = nn.ParameterStore(1)
                option = configuration(stride, dilation, padding)
                y = nn.conv1d(p, "conv", nn.zeros([1, 7, 1]), 1, 3, option)
                formula = nn.conv_output_size(7, 3, stride, dilation, padding)
                assert y.size(1) == formula == expected[row]
                row += 1
                print(
                    "Output size padding/stride/dilation:",
                    padding,
                    stride,
                    dilation,
                    "formula/op:",
                    formula,
                    y.size(1),
                )


def conv2d_checks():
    x = random_tensor([2, 6, 5, 3], nn.Mt19937_64(1))
    for padding in [nn.Padding.Valid, nn.Padding.Same]:
        for stride in [1, 2]:
            for dilation in [1, 2]:
                if stride > 1 and dilation > 1:
                    continue
                p = nn.ParameterStore(2)
                y = nn.conv2d(
                    p, "conv", x, 4, 3, configuration(stride, dilation, padding)
                )
                error = difference(
                    y, reference(x, p.at("conv/W"), stride, dilation, padding)
                )
                assert error < 1e-5
                print(
                    "Conv2D padding/stride/dilation:",
                    padding,
                    stride,
                    dilation,
                    "shape:",
                    tuple(y.shape()),
                    "reference error:",
                    error,
                )
    p = nn.ParameterStore(2)
    nn.conv2d(p, "conv", x, 4, 3)
    assert p.count() == 112
    print("3x3 Conv2D 3->4 with bias parameters:", p.count())


def conv1d_and_3d():
    rng = nn.Mt19937_64(3)
    p = nn.ParameterStore(3)
    option = configuration(padding=nn.Padding.Valid)
    x1 = random_tensor([2, 9, 2], rng)
    x3 = random_tensor([1, 4, 4, 4, 2], rng)
    y1 = nn.conv1d(p, "c1", x1, 3, 3, option)
    y3 = nn.conv3d(p, "c3", x3, 3, 2, option)
    w1 = nn.reshape(p.at("c1/W"), [3, 2, 3])
    w3 = p.at("c3/W")
    d1 = difference(y1, reference(x1, w1, padding=nn.Padding.Valid))
    d3 = difference(y3, reference(x3, w3, padding=nn.Padding.Valid))
    assert (
        tuple(y1.shape()) == (2, 7, 3)
        and tuple(y3.shape()) == (1, 3, 3, 3, 3)
        and max(d1, d3) < 1e-5
        and w3.numel() == 48
    )
    print(
        "Conv1D/3D shapes:",
        tuple(y1.shape()),
        tuple(y3.shape()),
        "reference errors:",
        d1,
        d3,
    )


def depthwise_separable():
    x = random_tensor([1, 6, 6, 3], nn.Mt19937_64(4))
    p = nn.ParameterStore(4)
    option = configuration()
    dw = nn.depthwise_conv2d(p, "dw", x, 2, 3, option)
    sep = nn.separable_conv2d(p, "sep", x, 8, 3, option)
    dd = difference(dw, reference(x, p.at("dw/W"), depthwise=True))
    h = reference(x, p.at("sep/depthwise/W"), depthwise=True)
    ds = difference(
        sep, reference(h, p.at("sep/pointwise/W"), padding=nn.Padding.Valid)
    )
    assert (
        tuple(dw.shape()) == (1, 6, 6, 6)
        and tuple(sep.shape()) == (1, 6, 6, 8)
        and max(dd, ds) < 1e-5
    )
    full, separable = nn.ParameterStore(0), nn.ParameterStore(0)
    z = nn.zeros([1, 4, 4, 64])
    nn.conv2d(full, "f", z, 128, 3, option)
    nn.separable_conv2d(separable, "s", z, 128, 3, option)
    assert full.count() == 73728 and separable.count() == 8768
    print(
        "Depthwise/separable reference errors:",
        dd,
        ds,
        "full/separable parameters:",
        full.count(),
        separable.count(),
    )


def transposed():
    rng = nn.Mt19937_64(5)
    p = nn.ParameterStore(5)
    option = configuration(stride=2)
    x = random_tensor([1, 6, 6, 2], rng).requires_grad_()
    r = random_tensor([1, 3, 3, 3], rng)
    y = nn.conv2d(p, "conv", x, 3, 3, option)
    gx = nn.grad(nn.sum(nn.mul(y, r)), [x])[0]
    weights = p.at("conv/W").detach()
    back = nn.conv2d_backprop_input([1, 6, 6, 2], weights, r, nn.conv_options(2))
    transpose = nn.conv2d_transpose(r, weights, [1, 6, 6, 2], nn.conv_options(2))
    db, dt = difference(gx, back), difference(gx, transpose)
    assert max(db, dt) < 1e-6
    print("Transposed convolution as input gradient errors:", db, dt)
    p = nn.ParameterStore(6)
    x = nn.ones([1, 4, 4, 1])
    same = configuration(stride=2)
    valid = configuration(stride=2, padding=nn.Padding.Valid)
    nn.conv2d_transpose(p, "same", x, 1, 3, same)
    p.set("same/W", nn.ones([3, 3, 1, 1]))
    up = nn.conv2d_transpose(p, "same", x, 1, 3, same)
    up_valid = nn.conv2d_transpose(p, "valid", x, 1, 3, valid)
    assert tuple(up.shape()) == (1, 8, 8, 1) and tuple(up_valid.shape()) == (1, 9, 9, 1)
    print("Ones-kernel checkerboard overlap counts:")
    for i in range(8):
        row = []
        for j in range(8):
            expected = (2 if i >= 2 and i % 2 == 0 else 1) * (
                2 if j >= 2 and j % 2 == 0 else 1
            )
            assert up.at([0, i, j, 0]) == expected
            row.append(expected)
        print(*row)


def dilated():
    p = nn.ParameterStore(7)
    x = random_tensor([1, 8, 8, 1], nn.Mt19937_64(7))
    y = nn.conv2d(p, "d", x, 1, 3, configuration(dilation=2))
    w3 = list(p.at("d/W").values())
    w5 = [0.0] * 25
    for a in range(3):
        for b in range(3):
            w5[2 * a * 5 + 2 * b] = w3[a * 3 + b]
    ordinary = nn.conv2d(
        x, nn.from_vector(w5, [5, 5, 1, 1]), nn.conv_options(1, nn.Padding.Same)
    )
    error = difference(y, ordinary)
    assert error < 1e-5
    print("5x5 kernel with holes:", w5, "dilated comparison error:", error)
    print(
        "Receptive fields for dilations 1,1,1 / 1,2,4 / 1,2,4,8:",
        1 + 2 * 3,
        1 + 2 * 7,
        1 + 2 * 15,
    )


def main(arguments=None):
    nn.set_num_threads(2)
    output_sizes()
    conv2d_checks()
    conv1d_and_3d()
    depthwise_separable()
    transposed()
    dilated()
    print("All convolution layer checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
