#pragma once
#include "datamunge/nn/unet_models.hpp"
namespace unet {
namespace nn = datamunge::nn;
using nn::models::Unet;
using nn::models::UnetCenterCropTo;
using nn::models::UnetConfig;
using nn::models::UnetOutputs;
using nn::models::UnetOutputSide;
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
