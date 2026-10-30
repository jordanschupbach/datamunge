#pragma once
// Compatibility names for canonical tensor CondenseNet models and conversion.
#include <datamunge/nn/condensenet_models.hpp>
#include "common/grouped.hpp"
#include "common/mobilenet.hpp"
namespace condensenet {
namespace nn=datamunge::nn;
using nn::models::MobileLayer;
using nn::models::CondenseContext;
using nn::models::LearnedGroupConvOutputs;
using nn::models::UpdateCondensation;
using nn::models::LearnedGroupConv;
using nn::models::CondenseNetConfig;
using nn::models::CondenseNetOutputs;
using nn::models::CondenseNet;
using nn::models::CondensedConv;
using nn::models::Condense;
using nn::models::CondenseNetConvertedCost;
using nn::models::ConvertedCondenseNetCost;
/// @copydoc nn::models::CondenseBNReLU
inline nn::Tensor BNReLU(CondenseContext& c,const std::string& name,const nn::Tensor& x){return nn::models::CondenseBNReLU(c,name,x);}
}
