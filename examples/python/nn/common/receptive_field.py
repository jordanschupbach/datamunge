"""Receptive-field geometry for a chain of spatial layers, one axis at a time
(examples/nn/common/receptive_field.hpp).

Input pixel centers are .5, 1.5, ...; jump is the spacing between outputs in
input coordinates. A kernel's effective width is dilation*(kernel-1)+1.
Padding shifts the first center; it does not enlarge the real input. Bounds
describe the enclosing interval, including holes for dilation.
"""
import math


def _lround(value):
    """std::lround: halfway cases away from zero."""
    return int(math.floor(value + 0.5)) if value >= 0 else -int(math.floor(-value + 0.5))


class ReceptiveField:
    def __init__(self, width=1, jump=1, start=0.5):
        self.width, self.jump, self.start = int(width), int(jump), float(start)

    def after(self, kernel, stride, padding_before, dilation=1):
        if kernel <= 0 or stride <= 0 or padding_before < 0 or dilation <= 0:
            raise ValueError("invalid receptive-field layer geometry")
        effective = dilation * (kernel - 1) + 1
        return ReceptiveField(self.width + (effective - 1) * self.jump, self.jump * stride,
                              self.start + ((effective - 1) / 2.0 - padding_before) * self.jump)

    def bounds(self, coordinate):
        """Half-open integer pixel bounds, before clipping to the image."""
        center = self.start + coordinate * self.jump
        return (_lround(center - (self.width - 1) / 2.0 - 0.5), _lround(center + (self.width - 1) / 2.0 - 0.5) + 1)
