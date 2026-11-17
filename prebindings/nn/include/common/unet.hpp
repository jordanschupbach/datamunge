#pragma once
#include "datamunge/nn/unet_models.hpp"
namespace unet {
namespace nn = datamunge::nn;
using datamunge::nn::models::Unet;
using datamunge::nn::models::UnetCenterCropTo;
using datamunge::nn::models::UnetConfig;
using datamunge::nn::models::UnetOutputs;
using datamunge::nn::models::UnetOutputSide;
inline nn::Tensor CenterCropTo(const nn::Tensor &source, const nn::Tensor &reference) {
  return UnetCenterCropTo(source, reference);
}
inline nn::Tensor ConvPair(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                           std::int64_t out, nn::Padding padding = nn::Padding::Same) {
  return nn::models::UnetConvPair(p, name, x, out, padding);
}
inline nn::Tensor UpConv(nn::ParameterStore &p, const std::string &name, const nn::Tensor &x,
                         std::int64_t out) {
  return nn::models::UnetUpConv(p, name, x, out);
}
} // namespace unet
