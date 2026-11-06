#pragma once
// Compatibility aliases to the canonical tensor implementation.
#include "datamunge/nn/efficientdet_models.hpp"
#include "common/retinanet.hpp"
#include "common/bifpn.hpp"
namespace efficientdet {
namespace nn=datamunge::nn;
using datamunge::nn::models::EfficientDetSpec;
using datamunge::nn::models::EfficientDetScaling;
using datamunge::nn::models::EfficientDetConfig;
using datamunge::nn::models::EfficientDet;
}
