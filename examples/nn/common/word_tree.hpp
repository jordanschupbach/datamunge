#pragma once
#include "datamunge/nn/word_tree_models.hpp"
namespace wordtree {
namespace nn = datamunge::nn;
using nn::models::WordTree;
using nn::models::WordTreeOutputs;
using nn::models::WordTreeProbabilities;
inline nn::Tensor WordTreeLoss(const WordTreeOutputs &output, const std::vector<int> &labels,
                               const WordTree &tree) {
  return nn::models::WordTreeLoss(
      output, nn::from_vector(std::vector<std::int64_t>(labels.begin(), labels.end())), tree);
}
} // namespace wordtree
