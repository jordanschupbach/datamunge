#pragma once
// Compatibility names for canonical quantized RegNet tensor models.
#include <datamunge/nn/regnet_models.hpp>
#include "common/grouped.hpp"
#include "common/mobilenet.hpp"
namespace regnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::RegNetParams;
using datamunge::nn::models::RegNetModel;
using datamunge::nn::models::RegNetStages;
using datamunge::nn::models::RegNetDesign;
using datamunge::nn::models::RegNetConfig;
using datamunge::nn::models::RegNet;
/// @copydoc nn::models::RegNetConvBN
inline nn::Tensor ConvBN(mobilenet::MobileContext& c,const std::string& name,nn::Tensor x,int out,int kernel,int stride,int groups,bool relu){return nn::models::RegNetConvBN(c,name,x,out,kernel,stride,groups,relu);}
/// @copydoc nn::models::RegNetBlock
inline nn::Tensor Block(mobilenet::MobileContext& c,const std::string& name,const nn::Tensor& x,int out,int stride,int group_width,double se){return nn::models::RegNetBlock(c,name,x,out,stride,group_width,se);}
}
