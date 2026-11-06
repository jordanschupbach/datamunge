#pragma once
// Compatibility names for canonical tensor capsule models.
#include <datamunge/nn/capsule_models.hpp>
namespace capsnet {
namespace nn=datamunge::nn;
using datamunge::nn::models::RoutingOutputs;
using datamunge::nn::models::CapsNetOutputs;
using datamunge::nn::models::Squash;
using datamunge::nn::models::DynamicRouting;
using datamunge::nn::models::CapsuleLengths;
using datamunge::nn::models::MarginLoss;
using datamunge::nn::models::CapsNet;
}
