#pragma once
#include <datamunge/nn/depth_data.hpp>
#include "common/checks.hpp"
namespace depth {namespace nn=datamunge::nn;
using datamunge::nn::models::DepthData;
using datamunge::nn::models::DepthSplits;
using datamunge::nn::models::LoadDepthPairs;
using datamunge::nn::models::DepthBatch;
using datamunge::nn::models::DepthPriors;
using datamunge::nn::models::FitDepthPriors;
using datamunge::nn::models::DepthPriorImages;
using datamunge::nn::models::DepthMetrics;
using datamunge::nn::models::ScoreDepth;
}
