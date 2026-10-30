#pragma once
#include <datamunge/nn/pose_data.hpp>
#include "common/checks.hpp"
#include "common/hourglass.hpp"
namespace pose {
using datamunge::nn::models::PoseData;
using datamunge::nn::models::PoseSplits;
using datamunge::nn::models::PoseArrays;
using datamunge::nn::models::PoseBatch;
using datamunge::nn::models::LoadPoseCrops;
using datamunge::nn::models::PoseTargets;
using datamunge::nn::models::PoseHeatmaps;
using datamunge::nn::models::PoseHeatmapMSE;
using datamunge::nn::models::DecodePoseHeatmaps;
using datamunge::nn::models::TensorPoseMetrics;
using datamunge::nn::models::PoseMetrics;
using datamunge::nn::models::ScoreTensorPose;
using datamunge::nn::models::ScorePose;
using datamunge::nn::models::FitPoseMean;
using datamunge::nn::models::ConstantPose;
}
