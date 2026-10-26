"""Explicit NHWC image-to-column extraction and its additive adjoint (examples/nn/common/im2col.hpp).

Each patch-matrix row is one output location; columns order kernel row, kernel
column, input channel. SAME puts an odd extra pad at the bottom/right, as
TensorFlow does. Kernels perform cross-correlation (no spatial reversal).
Rectangular kernels, separate spatial strides and dilation are supported.
"""
import math
import struct

from pydatamunge import nn


def _float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class PatchGeometry:
    """im2col::PatchGeometry: output size and SAME padding offsets of a patch extraction."""

    def __init__(self, input_shape, kernel_h, kernel_w, stride_h=1, stride_w=1, dilation_h=1, dilation_w=1,
                 padding="VALID"):
        shape = [int(d) for d in input_shape]
        if (len(shape) != 4 or math.prod(shape) < 1 or min(kernel_h, kernel_w, stride_h, stride_w, dilation_h,
                                                           dilation_w) <= 0 or padding not in ("VALID", "SAME")):
            raise ValueError("invalid patch geometry")
        self.n, self.h, self.w, self.c = shape
        self.kh, self.kw, self.sh, self.sw, self.dh, self.dw = kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w
        effective_h, effective_w = (self.kh - 1) * self.dh + 1, (self.kw - 1) * self.dw + 1
        if padding == "SAME":
            self.oh, self.ow = (self.h + self.sh - 1) // self.sh, (self.w + self.sw - 1) // self.sw
        else:
            self.oh = (self.h - effective_h) // self.sh + 1 if self.h >= effective_h else 0
            self.ow = (self.w - effective_w) // self.sw + 1 if self.w >= effective_w else 0
        if not self.oh or not self.ow:
            raise ValueError("patch geometry has no output positions")
        self.top = max(0, (self.oh - 1) * self.sh + effective_h - self.h) // 2 if padding == "SAME" else 0
        self.left = max(0, (self.ow - 1) * self.sw + effective_w - self.w) // 2 if padding == "SAME" else 0
        self._pairs = None

    def rows(self):
        return self.n * self.oh * self.ow

    def columns(self):
        return self.kh * self.kw * self.c

    def input_shape(self):
        return [self.n, self.h, self.w, self.c]

    def pairs(self):
        """(patch-matrix index, input index) for every in-bounds patch entry."""
        if self._pairs is None:
            pairs = []
            columns = self.columns()
            for n in range(self.n):
                for y in range(self.oh):
                    for xo in range(self.ow):
                        for ky in range(self.kh):
                            for kx in range(self.kw):
                                iy, ix = y * self.sh + ky * self.dh - self.top, xo * self.sw + kx * self.dw - self.left
                                if iy < 0 or iy >= self.h or ix < 0 or ix >= self.w:
                                    continue
                                for c in range(self.c):
                                    pairs.append((((n * self.oh + y) * self.ow + xo) * columns + (ky * self.kw + kx) * self.c + c,
                                                  ((n * self.h + iy) * self.w + ix) * self.c + c))
            self._pairs = pairs
        return self._pairs


def im2col(images, geometry):
    """[N,H,W,C] float32 -> [rows, columns] patch matrix."""
    if images.dtype() != nn.DType.Float32 or list(images.shape()) != geometry.input_shape():
        raise ValueError("Im2Col requires matching float32 NHWC input")
    source = images.contiguous().values()
    output = [0.0] * (geometry.rows() * geometry.columns())
    for column, pixel in geometry.pairs():
        output[column] = source[pixel]
    return nn.from_vector(output, [geometry.rows(), geometry.columns()])


def col2im(columns, geometry):
    """Adjoint of im2col: scatter-adds each patch entry back to its pixel (float32 accumulation)."""
    if columns.dtype() != nn.DType.Float32 or list(columns.shape()) != [geometry.rows(), geometry.columns()]:
        raise ValueError("Col2Im requires matching float32 patch matrix")
    source = columns.contiguous().values()
    output = [0.0] * math.prod(geometry.input_shape())
    for column, pixel in geometry.pairs():
        # Round each partial sum to float32, as the C++ std::vector<float> accumulation does.
        output[pixel] = _float32(output[pixel] + source[column])
    return nn.from_vector(output, geometry.input_shape())
