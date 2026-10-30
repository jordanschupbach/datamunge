#pragma once
#include <datamunge/nn/perceptual_models.hpp>
#include "common/checks.hpp"
namespace style {namespace nn=datamunge::nn;
using datamunge::nn::models::TensorMap;
using datamunge::nn::models::LoadTensorMap;
using datamunge::nn::models::VggImageNetInput;
using datamunge::nn::models::PerceptualFeatures;
using datamunge::nn::models::FrozenVgg19;
}
