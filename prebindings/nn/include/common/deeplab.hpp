#pragma once
#include "common/resnet.hpp"
#include "datamunge/nn/deeplab_models.hpp"
namespace deeplab {
namespace nn = datamunge::nn;
using datamunge::nn::models::AtrousLayer;
using datamunge::nn::models::AtrousScorePyramid;
using datamunge::nn::models::DeepLabOutputs;
using datamunge::nn::models::DeepLabV1;
using datamunge::nn::models::DeepLabV1Config;
using datamunge::nn::models::DeepLabV2;
using datamunge::nn::models::DeepLabV2Config;
using datamunge::nn::models::DenseResNet;
using datamunge::nn::models::DenseResNetBlock;
using datamunge::nn::models::DenseResNetConv;
inline nn::Tensor ResizeLike(const nn::Tensor &x, const nn::Tensor &like) {
  return nn::models::DeepLabResizeLike(x, like);
}
} // namespace deeplab
