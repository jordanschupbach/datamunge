#pragma once
#include <datamunge/nn/flow_data.hpp>
#include "common/checks.hpp"
namespace flow {namespace nn=datamunge::nn;
using datamunge::nn::models::FlowData;
using datamunge::nn::models::FlowSplits;
using datamunge::nn::models::LoadFlowPairs;
using datamunge::nn::models::FlowBatch;
using datamunge::nn::models::FlowSupervision;
using datamunge::nn::models::PrepareFlowSupervision;
using datamunge::nn::models::FlowPrepared;
using datamunge::nn::models::FlowMetrics;
using datamunge::nn::models::ScoreFlow;
using datamunge::nn::models::FitMeanFlow;
using datamunge::nn::models::ConstantFlow;
}
