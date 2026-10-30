#pragma once
#include "common/checks.hpp"
#include "datamunge/nn/multilabel_data.hpp"
namespace multilabel {
namespace nn = datamunge::nn;
using nn::models::LoadMultilabel;
using nn::models::MultilabelData;
using nn::models::MultilabelPrior;
using nn::models::MultilabelRows;
using nn::models::MultilabelSplits;
using nn::models::MultilabelWeights;
} // namespace multilabel
