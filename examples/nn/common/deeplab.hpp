#pragma once
#include "common/resnet.hpp"
#include "datamunge/nn/deeplab_models.hpp"
namespace deeplab {
namespace nn = datamunge::nn;
using nn::models::AtrousLayer;
using nn::models::AtrousScorePyramid;
using nn::models::DeepLabOutputs;
using nn::models::DeepLabV1;
using nn::models::DeepLabV1Config;
using nn::models::DeepLabV2;
using nn::models::DeepLabV2Config;
using nn::models::DenseResNet;
using nn::models::DenseResNetBlock;
using nn::models::DenseResNetConv;
inline nn::Tensor ResizeLike(const nn::Tensor &x, const nn::Tensor &like) {
  return nn::models::DeepLabResizeLike(x, like);
}
} // namespace deeplab
