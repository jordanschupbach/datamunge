#pragma once
#include <datamunge/nn/paf_data.hpp>
#include "common/pose_data.hpp"
namespace pose {
using datamunge::nn::models::kPoseEdges;
using datamunge::nn::models::PafData;
using datamunge::nn::models::PafSplits;
using datamunge::nn::models::PafBatch;
using datamunge::nn::models::Coco18;
using datamunge::nn::models::LoadPafImages;
using datamunge::nn::models::PafTargets;
using datamunge::nn::models::PreparePafTargets;
using datamunge::nn::models::PafMaskedMSE;
}
