#pragma once
// Compatibility names for canonical tensor RepVGG models and inference folding.
#include <datamunge/nn/repvgg_models.hpp>
#include "common/grouped.hpp"
namespace repvgg {
namespace nn=datamunge::nn;
using nn::models::RepVGGModel;
using nn::models::RepVGGBlockSpec;
using nn::models::RepVGGOutputs;
using nn::models::RepVGGConfig;
using nn::models::FusedBlock;
using nn::models::RepVGGLayout;
using nn::models::RepVGGNorm;
using nn::models::RepVGGBranch;
using nn::models::RepVGGBlock;
using nn::models::RepVGG;
using nn::models::FuseRepVGGBlock;
using nn::models::DeployRepVGG;
}
