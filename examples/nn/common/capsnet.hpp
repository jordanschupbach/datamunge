#pragma once
// Compatibility names for canonical tensor capsule models.
#include <datamunge/nn/capsule_models.hpp>
namespace capsnet {
namespace nn=datamunge::nn;
using nn::models::RoutingOutputs;
using nn::models::CapsNetOutputs;
using nn::models::Squash;
using nn::models::DynamicRouting;
using nn::models::CapsuleLengths;
using nn::models::MarginLoss;
using nn::models::CapsNet;
}
