#pragma once
// Compatibility names for canonical tensor vision modules.
#include <datamunge/nn/attention_cnn_models.hpp>
#include "common/resnet.hpp"
#include "common/deformable.hpp"
namespace attention_cnn {
namespace nn=datamunge::nn;
using datamunge::nn::models::GateOutputs;
using datamunge::nn::models::CbamOutputs;
using datamunge::nn::models::AttentionResNetConfig;
using datamunge::nn::models::AttentionResNetOutputs;
using datamunge::nn::models::SelectiveKernelOutputs;
using datamunge::nn::models::ChannelAttention;
using datamunge::nn::models::Conv2Hook;
using datamunge::nn::models::AfterBlockHook;
using datamunge::nn::models::SqueezeExcitationModule;
using datamunge::nn::models::Cbam;
using datamunge::nn::models::EcaKernel;
using datamunge::nn::models::Eca;
using datamunge::nn::models::AttentionResNet;
using datamunge::nn::models::SelectiveKernel;
using datamunge::nn::models::SelectiveKernelHook;
}
