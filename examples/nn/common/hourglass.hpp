#pragma once
#include <datamunge/nn/hourglass_models.hpp>
namespace pose {namespace nn=datamunge::nn;
using datamunge::nn::models::HourglassConfig;
using datamunge::nn::models::HourglassResult;
using datamunge::nn::models::PoseConv;
using datamunge::nn::models::PoseBN;
using datamunge::nn::models::PoseResidual;
using datamunge::nn::models::HourglassModule;
using datamunge::nn::models::StackedHourglass;
using datamunge::nn::models::CompactHourglassConfig;
}
