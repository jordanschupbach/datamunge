#pragma once
#include <datamunge/nn/partial_inpainting_losses.hpp>
#include "common/johnson.hpp"
#include "common/partial_unet.hpp"
namespace inpaint {
using datamunge::nn::models::PartialFeatures;
using datamunge::nn::models::PartialVgg;
using datamunge::nn::models::PartialGram;
using datamunge::nn::models::PartialL1;
using datamunge::nn::models::PartialFeatureLosses;
using datamunge::nn::models::PartialTV;
using datamunge::nn::models::PartialObjective;
}
