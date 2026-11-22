/// @file
/// @brief Synchronous population-based training checkpoint selection.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/ops.hpp"
#include "datamunge/optim/population_based_training.hpp"

namespace datamunge::nn {

/// @brief Pair the worst workers with the best validation checkpoints.
///
/// Smaller scores are better. Stable sorting breaks ties by worker index.
/// Workers outside the replacement group retain their own checkpoints.
/// @param losses Finite validation loss for each worker.
/// @param replacements Number of workers to replace, at most half the population.
/// @return Source worker index for each destination, in original worker order.
/// @throws std::invalid_argument Empty or nonfinite losses, or too many replacements.
/// @par Example
/// @code
/// const std::array losses{3.,1.,2.};
/// const auto sources=dm::nn::population_donors(losses,1);
/// std::print("{}\n",sources[0]); // 1
/// @endcode
/// @par References
/// - Jaderberg et al. (2017), Population Based Training of Neural Networks.
///   https://arxiv.org/abs/1711.09846
/// @see inherit_population()
inline std::vector<std::int64_t> population_donors(std::span<const double> losses,
                                                   std::size_t replacements) {
  return optim::detail::population_sources<std::int64_t>(losses,replacements);
}

/// @brief Apply the same synchronous donor mapping to every checkpoint field.
///
/// Each tensor has workers on its first axis. Gathering creates independent
/// outputs and preserves field order, shape and dtype. Use it for
/// model weights, optimizer state and hyperparameters before perturbation.
/// @param fields CPU checkpoint tensors with the same worker count.
/// @param sources Source worker index for every destination.
/// @return Checkpoint fields gathered by worker index, detached from autograd.
/// @throws std::invalid_argument Empty fields or sources, invalid indices or incompatible fields.
/// @par Example
/// @code
/// namespace n=dm::nn;
/// const std::array fields{n::from_vector<double>({2,3}),n::from_vector<double>({5,7})};
/// const std::array<std::int64_t,2> sources{1,1};
/// const auto inherited=n::inherit_population(fields,sources);
/// std::print("{:.0f}\n",inherited[1].to_vector<double>()[0]); // 7
/// @endcode
/// @see population_donors()
inline std::vector<Tensor> inherit_population(std::span<const Tensor> fields,
                                               std::span<const std::int64_t> sources) {
  if (fields.empty() || sources.empty() ||
      !std::all_of(sources.begin(),sources.end(),[&](auto index) {
        return index>=0 && static_cast<std::size_t>(index)<sources.size();
      })) {
    throw std::invalid_argument("inherit_population: invalid fields or indices");
  }
  for (const auto& field:fields) {
    if (!field.defined() || field.dim()==0 || static_cast<std::size_t>(field.size(0))!=sources.size() ||
        field.device().backend!=compute::Backend::CPU) {
      throw std::invalid_argument("inherit_population: incompatible checkpoint field");
    }
  }
  const NoGradGuard no_grad;
  const auto indices=from_vector(std::vector<std::int64_t>(sources.begin(),sources.end()));
  std::vector<Tensor> result;
  result.reserve(fields.size());
  for (const auto& field:fields) result.push_back(field.numel()==0 ? field.detach().clone() : gather(field,indices));
  return result;
}

} // namespace datamunge::nn
