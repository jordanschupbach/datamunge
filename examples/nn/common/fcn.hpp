#pragma once
#include "datamunge/nn/fcn_models.hpp"
namespace fcn {
namespace nn = datamunge::nn;
using nn::models::Fcn;
using nn::models::FcnConfig;
using nn::models::FcnFuseSkip;
using nn::models::FcnOutputs;
inline nn::Tensor SpatialCrop(const nn::Tensor &source, const nn::Tensor &reference,
                              std::int64_t offset = 0) {
  return nn::models::FcnSpatialCrop(source, reference, offset);
}
inline nn::Tensor BilinearKernel(int stride, int channels) {
  return nn::models::FcnBilinearKernel(stride, channels);
}
inline nn::Tensor BilinearDeconvolution(const nn::Tensor &source, int stride, int channels,
                                        bool valid = false) {
  return nn::models::FcnBilinearDeconvolution(source, stride, channels, valid);
}
} // namespace fcn
