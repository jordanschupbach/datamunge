"""Masking and head helpers shared by the explicit-unroll recurrent cells
(examples/nn/common/rnn.hpp; tflow's common/nn/rnn.h)."""
import math
import struct

from pydatamunge import nn


def step_mask(lengths, t):
    """Float [B, 1] mask: 1 where lengths > t."""
    return nn.expand_dims(nn.cast(nn.greater(lengths, float(t)), nn.DType.Float32), 1)


def step_valid(lengths, t):
    """Bool [B, 1] mask for where()-style blends."""
    return nn.expand_dims(nn.greater(lengths, float(t)), 1)


def blend(mask, next_value, previous):
    """mask * next + (1 - mask) * previous (tflow's arithmetic masking)."""
    return nn.add(nn.mul(mask, next_value), nn.mul(nn.sub(1.0, mask), previous))


def head(store, name, x, incoming, outgoing):
    """Dense head "<name>/W" ~ N(0, 0.1^2), "<name>/b" = 0."""
    weight = store.get(name + "/W", [incoming, outgoing], nn.init.normal(0.1))
    bias = store.get(name + "/b", [outgoing], nn.init.zeros())
    return nn.bias_add(nn.matmul(x, weight), bias)


def masked_diagnostics(valid, x):
    """Zeros where the [B, 1] bool mask is false."""
    return nn.where(valid, x, nn.zeros_like(x))


def fixed_parameter(store, name, make):
    """The parameter `name`, created from the tensor make() on first use (a
    C++ lambda initializer that ignores the generator, e.g. a gate bias)."""
    if not store.contains(name):
        store.set(name, make())
    return store.at(name)


def float32(x):
    """x rounded to float32 (C++ float literals and float arithmetic such as std::sqrt(6.f / 14))."""
    return struct.unpack("f", struct.pack("f", x))[0]


def float32_sqrt_ratio(numerator, denominator):
    """std::sqrt(float(numerator) / float(denominator)) evaluated in float32."""
    return float32(math.sqrt(float32(float32(numerator) / float32(denominator))))
