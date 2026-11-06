#pragma once
// Compatibility names for canonical tensor CondenseNet models and conversion.
#include <datamunge/nn/condensenet_models.hpp>
#include "common/grouped.hpp"
#include "common/mobilenet.hpp"
namespace condensenet {
namespace nn=datamunge::nn;
using datamunge::nn::models::MobileLayer;
using datamunge::nn::models::CondenseContext;
using datamunge::nn::models::LearnedGroupConvOutputs;
using datamunge::nn::models::UpdateCondensation;
using datamunge::nn::models::LearnedGroupConv;
using datamunge::nn::models::CondenseNetConfig;
using datamunge::nn::models::CondenseNetOutputs;
using datamunge::nn::models::CondenseNet;
using datamunge::nn::models::CondensedConv;
using datamunge::nn::models::Condense;
using datamunge::nn::models::CondenseNetConvertedCost;
using datamunge::nn::models::ConvertedCondenseNetCost;
/// @copydoc nn::models::CondenseBNReLU
inline nn::Tensor BNReLU(CondenseContext& c,const std::string& name,const nn::Tensor& x){return nn::models::CondenseBNReLU(c,name,x);}
}
