#pragma once
#include <datamunge/nn/vqa_data.hpp>
#include "common/checks.hpp"
namespace vqa {namespace nn=datamunge::nn;
using datamunge::nn::models::VqaConfig;
using datamunge::nn::models::VqaData;
using datamunge::nn::models::VqaSplits;
using datamunge::nn::models::VqaRows;
using datamunge::nn::models::LoadVqa;
using datamunge::nn::models::VqaSoftCE;
using datamunge::nn::models::VqaConsensusCount;
using datamunge::nn::models::VqaMetrics;
using datamunge::nn::models::ScoreVqa;
using datamunge::nn::models::VqaPrior;
}
