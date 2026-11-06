#pragma once
#include "common/checks.hpp"
#include "datamunge/nn/multilabel_data.hpp"
namespace multilabel {
namespace nn = datamunge::nn;
using datamunge::nn::models::LoadMultilabel;
using datamunge::nn::models::MultilabelData;
using datamunge::nn::models::MultilabelPrior;
using datamunge::nn::models::MultilabelRows;
using datamunge::nn::models::MultilabelSplits;
using datamunge::nn::models::MultilabelWeights;
} // namespace multilabel
