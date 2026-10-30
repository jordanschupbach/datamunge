#pragma once
#include <datamunge/nn/style_losses.hpp>
#include "common/perceptual.hpp"
namespace style {
using datamunge::nn::models::StyleGram;
using datamunge::nn::models::GatysContentLoss;
using datamunge::nn::models::GatysStyleLayer;
using datamunge::nn::models::GatysStyleLoss;
using datamunge::nn::models::LoadStyleImage;
using datamunge::nn::models::WriteStylePng;
}
