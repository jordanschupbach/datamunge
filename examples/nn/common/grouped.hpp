// Grouped NHWC convolution with explicit channel partitions (port of tflow's
// common/nn/grouped.h). Each group has its own HWIO filter
// "<name>/group<g>/W" and bias and reads only its contiguous input partition.
// (nn::conv2d also supports Conv2DOptions::groups with one shared filter
// tensor; the explicit form keeps tflow's parameter names and per-group
// initializer fans.) Both channel counts must divide by `groups`.
#pragma once

#include <datamunge/nn/nn.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace grouped {

namespace nn = datamunge::nn;

inline nn::Tensor GroupedConv2D(
    nn::ParameterStore&         p,
    const std::string&          name,
    const nn::Tensor&           input,
    std::int64_t                out,
    std::int64_t                kernel,
    std::int64_t                groups,
    const nn::ConvLayerOptions& options = {}) {
  const std::int64_t in = input.dim() == 4 ? input.shape()[3] : 0;
  if (groups <= 0 || in <= 0 || out <= 0 || in % groups || out % groups)
    throw std::invalid_argument(
        "grouped convolution channels must be positive multiples of groups");
  if (groups == 1)
    return nn::conv2d(p, name, input, out, kernel, options);
  const auto              partitions = nn::split(input, groups, 3);
  std::vector<nn::Tensor> outputs;
  for (std::int64_t group = 0; group < groups; ++group)
    outputs.push_back(
        nn::conv2d(
            p,
            name + "/group" + std::to_string(group),
            partitions[static_cast<std::size_t>(group)],
            out / groups,
            kernel,
            options));
  return nn::concat(outputs, 3);
}

} // namespace grouped

namespace grouped {

/// Same per-group parameter names and initializer fans, executed by the core's
/// grouped kernel. Avoids separate slice/backward graphs for every group.
inline nn::Tensor FusedGroupedConv2D(
    nn::ParameterStore&         p,
    const std::string&          name,
    const nn::Tensor&           input,
    std::int64_t                out,
    std::int64_t                kernel,
    std::int64_t                groups,
    const nn::ConvLayerOptions& options = {}) {
  const auto in = input.dim() == 4 ? input.size(3) : 0;
  if (groups <= 0 || in <= 0 || out <= 0 || in % groups || out % groups)
    throw std::invalid_argument(
        "grouped convolution channels must be positive multiples of groups");
  if (groups == 1)
    return nn::conv2d(p, name, input, out, kernel, options);
  std::vector<nn::Tensor> weights, biases;
  for (std::int64_t group = 0; group < groups; ++group) {
    const auto prefix = name + "/group" + std::to_string(group);
    weights.push_back(
        p.get(prefix + "/W", {kernel, kernel, in / groups, out / groups}, options.init));
    if (options.bias)
      biases.push_back(p.get(prefix + "/b", {out / groups}, nn::init::zeros()));
  }
  nn::Conv2DOptions op;
  op.groups    = groups;
  op.strides   = {options.stride, options.stride};
  op.dilations = {options.dilation, options.dilation};
  op.padding   = options.padding;
  auto result  = nn::conv2d(input, nn::concat(weights, 3), op);
  if (options.bias)
    result = nn::bias_add(result, nn::concat(biases, 0));
  return nn::activate(result, options.activation, options.alpha);
}

} // namespace grouped
