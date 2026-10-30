#pragma once
#include <datamunge/nn/partial_unet_models.hpp>
#include "common/inpainting.hpp"
namespace inpaint {
using datamunge::nn::models::PartialConfig;
using datamunge::nn::models::PartialUNetResult;
using datamunge::nn::models::PartialResize;
using datamunge::nn::models::PartialAct;
using datamunge::nn::models::PartialUNet;
using datamunge::nn::models::PartialComposite;
}
