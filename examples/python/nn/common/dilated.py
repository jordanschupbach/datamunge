"""Differentiable dilation by inserting zeros between filter taps (examples/nn/common/dilated.hpp).

nn.conv2d supports dilation natively; this expanded-filter path is the
teaching/reference version: only the original taps carry gradient, inserted
zeros are constants. It materializes the effective kernel.
"""
from pydatamunge import nn


def _zero_pad_axis(x, axis, after):
    """Appends `after` zeros along `axis` (nn.pad's paddings vector is not bound; concat is equivalent)."""
    if after <= 0:
        return x
    shape = list(x.shape())
    shape[axis] = after
    return nn.concat([x, nn.zeros(shape, x.dtype())], axis)


def expand_dilated_kernel(kernel, dilation):
    """[h, w, in, out] -> [(h-1)d+1, (w-1)d+1, in, out] with zeros between taps."""
    if kernel.dim() != 4 or kernel.numel() <= 0 or dilation < 1:
        raise ValueError("dilation requires a nonempty HWIO filter and positive rate")
    if dilation == 1:
        return kernel
    h, w, inputs, outputs = (int(v) for v in kernel.shape())
    d = int(dilation)
    taps = nn.reshape(kernel, [h, 1, w, 1, inputs, outputs])
    padded = nn.reshape(_zero_pad_axis(_zero_pad_axis(taps, 1, d - 1), 3, d - 1), [h * d, w * d, inputs, outputs])
    return nn.slice(padded, [0, 0, 0, 0], [(h - 1) * d + 1, (w - 1) * d + 1, inputs, outputs])


def portable_dilated_conv2d(x, kernel, dilation, padding="SAME"):
    """Ordinary stride-1 convolution with the zero-expanded kernel."""
    return nn.conv2d(x, expand_dilated_kernel(kernel, dilation),
                     nn.conv_options(1, nn.Padding.Same if padding == "SAME" else nn.Padding.Valid))
