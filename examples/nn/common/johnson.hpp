#pragma once
#include <datamunge/nn/johnson_models.hpp>
#include "common/perceptual.hpp"
namespace style {
using datamunge::nn::models::JohnsonReflect;
using datamunge::nn::models::JohnsonCrop;
using datamunge::nn::models::JohnsonConv;
using datamunge::nn::models::JohnsonDown;
using datamunge::nn::models::JohnsonUp;
using datamunge::nn::models::JohnsonNorm;
using datamunge::nn::models::JohnsonResidual;
using datamunge::nn::models::JohnsonOutputs;
using datamunge::nn::models::JohnsonTransform;
using datamunge::nn::models::JohnsonFeatures;
using datamunge::nn::models::FrozenVgg16;
}
