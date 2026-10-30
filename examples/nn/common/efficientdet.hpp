#pragma once
// Compatibility aliases to the canonical tensor implementation.
#include "datamunge/nn/efficientdet_models.hpp"
#include "common/retinanet.hpp"
#include "common/bifpn.hpp"
namespace efficientdet {
namespace nn=datamunge::nn;
using nn::models::EfficientDetSpec;
using nn::models::EfficientDetScaling;
using nn::models::EfficientDetConfig;
using nn::models::EfficientDet;
}
