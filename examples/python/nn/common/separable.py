"""Compose depthwise + pointwise filters into an equivalent dense HWIO kernel
(examples/nn/common/separable.hpp).

W[h,w,i,o] = sum_m D[h,w,i,m] * P[0,0,i*M+m,o]; no activation may sit between
the two convolutions. Built from differentiable ops, so gradients reach both factors.
"""
from pydatamunge import nn


def compose_separable_kernel(depthwise, pointwise, kernel, inputs, multiplier, outputs):
    if kernel <= 0 or inputs <= 0 or multiplier <= 0 or outputs <= 0:
        raise ValueError("positive separable dimensions required")
    spatial = nn.transpose(nn.reshape(depthwise, [kernel * kernel, inputs, multiplier]), [1, 0, 2])
    mixing = nn.reshape(pointwise, [inputs, multiplier, outputs])
    product = nn.matmul(spatial, mixing)
    return nn.reshape(nn.transpose(product, [1, 0, 2]), [kernel, kernel, inputs, outputs])
