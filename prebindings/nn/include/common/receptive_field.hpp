// Receptive-field geometry for a chain of spatial layers, one axis at a time
// (port of tflow's common/nn/receptive_field.h).
// Input pixel centers are .5,1.5,...; jump is the spacing between outputs in
// input coordinates. A kernel's effective width is dilation*(kernel-1)+1.
// Padding shifts the first center; it does not enlarge the real input.
// Bounds describe the enclosing interval, including holes for dilation.
#pragma once

#include <cmath>
#include <stdexcept>
#include <utility>

namespace receptive {

struct ReceptiveField {
  int    width = 1, jump = 1;
  double start = .5;
  ReceptiveField After(int kernel, int stride, int padding_before, int dilation = 1) const {
    if (kernel <= 0 || stride <= 0 || padding_before < 0 || dilation <= 0)
      throw std::invalid_argument("invalid receptive-field layer geometry");
    const int effective = dilation * (kernel - 1) + 1;
    return {width + (effective - 1) * jump, jump * stride, start + ((effective - 1) / 2. - padding_before) * jump};
  }
  // Half-open integer pixel bounds, before clipping to the image.
  std::pair<int, int> Bounds(int coordinate) const {
    const double center = start + coordinate * jump;
    return {int(std::lround(center - (width - 1) / 2. - .5)), int(std::lround(center + (width - 1) / 2. - .5)) + 1};
  }
};

}  // namespace receptive
