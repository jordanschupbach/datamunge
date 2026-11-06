#pragma once
// Compatibility names for canonical tensor RepVGG models and inference folding.
#include <datamunge/nn/repvgg_models.hpp>
#include "common/grouped.hpp"
namespace repvgg {
namespace nn=datamunge::nn;
using datamunge::nn::models::RepVGGModel;
using datamunge::nn::models::RepVGGBlockSpec;
using datamunge::nn::models::RepVGGOutputs;
using datamunge::nn::models::RepVGGConfig;
using datamunge::nn::models::FusedBlock;
using datamunge::nn::models::RepVGGLayout;
using datamunge::nn::models::RepVGGNorm;
using datamunge::nn::models::RepVGGBranch;
using datamunge::nn::models::RepVGGBlock;
using datamunge::nn::models::RepVGG;
using datamunge::nn::models::FuseRepVGGBlock;
using datamunge::nn::models::DeployRepVGG;
}
