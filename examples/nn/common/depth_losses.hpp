#pragma once
#include <datamunge/nn/depth_models.hpp>
namespace depth {namespace nn=datamunge::nn;
using datamunge::nn::models::PositiveMetricDepth;
using datamunge::nn::models::MonocularDepthResult;
using datamunge::nn::models::MonocularDepth;
using datamunge::nn::models::DepthError;
using datamunge::nn::models::DepthLogError;
using datamunge::nn::models::DepthNonemptyMean;
using datamunge::nn::models::DepthLogLoss;
using datamunge::nn::models::DepthGradientLoss;
using datamunge::nn::models::DepthObjective;
}
