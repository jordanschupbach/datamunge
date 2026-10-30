#pragma once
// Compatibility names for canonical tensor vision modules.
#include <datamunge/nn/attention_cnn_models.hpp>
#include "common/resnet.hpp"
#include "common/deformable.hpp"
namespace attention_cnn {
namespace nn=datamunge::nn;
using nn::models::GateOutputs;
using nn::models::CbamOutputs;
using nn::models::AttentionResNetConfig;
using nn::models::AttentionResNetOutputs;
using nn::models::SelectiveKernelOutputs;
using nn::models::ChannelAttention;
using nn::models::Conv2Hook;
using nn::models::AfterBlockHook;
using nn::models::SqueezeExcitationModule;
using nn::models::Cbam;
using nn::models::EcaKernel;
using nn::models::Eca;
using nn::models::AttentionResNet;
using nn::models::SelectiveKernel;
using nn::models::SelectiveKernelHook;
}
