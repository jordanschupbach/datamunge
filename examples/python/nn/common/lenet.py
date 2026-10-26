"""LeNet-5 with the sparse C3 connectivity and learned subsampling of LeCun et al. 1998
(examples/nn/common/lenet.hpp; parameter names match tflow's common/nn/lenet.h).

The six trainable stages contain exactly 60,000 parameters. C3 stores only its
60 connected 5x5 kernels. Subsampling has a learned scale and bias per map; all
six stages use 1.7159*tanh(2x/3). [0,1] MNIST pixels are padded to 32x32 and
mapped to [-0.1, 1.175]. Fixed Euclidean RBF outputs use 7x12 seven-segment codes.
"""
from types import SimpleNamespace

from pydatamunge import nn

C3_CONNECTIONS = (
    (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 0), (5, 0, 1), (0, 1, 2, 3), (1, 2, 3, 4),
    (2, 3, 4, 5), (3, 4, 5, 0), (4, 5, 0, 1), (5, 0, 1, 2), (0, 1, 3, 4), (1, 2, 4, 5), (0, 2, 3, 5),
    (0, 1, 2, 3, 4, 5))


def lenet_c3_connections():
    return C3_CONNECTIONS


def lenet_tanh(x):
    return nn.mul(nn.tanh(nn.mul(x, 2.0 / 3.0)), 1.7159)


def lenet_subsample(parameters, name, x):
    channels = int(x.shape()[-1])
    total = nn.mul(nn.avg_pool2d(x, 2, 2), 4.0)
    scale = parameters.get(name + "/scale", [channels], nn.init.constant(0.25))
    bias = parameters.get(name + "/bias", [channels], nn.init.zeros())
    return lenet_tanh(nn.add(nn.mul(total, scale), bias))


def lenet_c3(parameters, name, x):
    options = nn.ConvLayerOptions()
    options.padding = nn.Padding.Valid
    maps = []
    for i, channels in enumerate(C3_CONNECTIONS):
        selected = nn.gather(x, nn.from_vector_int64(list(channels), [len(channels)]), 3)
        maps.append(nn.conv2d(parameters, name + "/map" + str(i), selected, 1, 5, options))
    return lenet_tanh(nn.concat(maps, 3))


def lenet_digit_codes():
    """[10, 84] seven-segment codes in {-1, 1} drawn on a 7x12 grid.

    Segment order: top, upper right, lower right, bottom, lower left, upper left, middle."""
    codes = [-1.0] * (10 * 84)
    segments = ("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
    for digit in range(10):
        for segment in segments[digit]:
            if segment in "adg":
                y = 1 if segment == "a" else 10 if segment == "d" else 5
                for x in range(1, 6):
                    codes[digit * 84 + y * 7 + x] = 1.0
            else:
                upper = segment in "bf"
                x = 5 if segment in "bc" else 1
                for y in range(2 if upper else 6, (4 if upper else 9) + 1):
                    codes[digit * 84 + y * 7 + x] = 1.0
    return nn.from_vector(codes, [10, 84])


def lenet_rbf_logits(features):
    """Negative squared Euclidean distances to the fixed digit codes: [N, 84] -> [N, 10]."""
    difference = nn.sub(nn.expand_dims(features, 1), lenet_digit_codes())
    return nn.neg(nn.sum(nn.square(difference), [2]))


def _zero_pad_spatial(x, amount):
    """nn::pad(x, {{0,0},{a,a},{a,a},{0,0}}) composed from concat (nn.pad's paddings vector is not bound)."""
    n, h, w, c = (int(v) for v in x.shape())
    rows = nn.zeros([n, amount, w, c], x.dtype())
    x = nn.concat([rows, x, rows], 1)
    columns = nn.zeros([n, h + 2 * amount, amount, c], x.dtype())
    return nn.concat([columns, x, columns], 2)


def lenet5(parameters, flat_pixels):
    """Flat [N, 784] pixels in [0, 1] -> every stage of LeNet-5 (image, c1, s2, c3, s4, c5, f6, logits)."""
    out = SimpleNamespace()
    padded = _zero_pad_spatial(nn.reshape(flat_pixels, [-1, 28, 28, 1]), 2)
    out.image = nn.sub(nn.mul(padded, 1.275), 0.1)
    options = nn.ConvLayerOptions()
    options.padding = nn.Padding.Valid
    out.c1 = lenet_tanh(nn.conv2d(parameters, "c1", out.image, 6, 5, options))
    out.s2 = lenet_subsample(parameters, "s2", out.c1)
    out.c3 = lenet_c3(parameters, "c3", out.s2)
    out.s4 = lenet_subsample(parameters, "s4", out.c3)
    out.c5 = lenet_tanh(nn.conv2d(parameters, "c5", out.s4, 120, 5, options))
    out.f6 = lenet_tanh(nn.dense(parameters, "f6", nn.reshape(out.c5, [-1, 120]), 84))
    out.logits = lenet_rbf_logits(out.f6)
    return out
