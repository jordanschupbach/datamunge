#pragma once
#include <datamunge/nn/super_resolution_data.hpp>
#include "common/checks.hpp"
#include "common/cifar.hpp"
namespace sr {namespace nn=datamunge::nn;using nn::Tensor;
using datamunge::nn::models::SRCubicTensor;
using datamunge::nn::models::SRCubic;
using datamunge::nn::models::SRSymmetric;
using datamunge::nn::models::SRContributionMatrix;
using datamunge::nn::models::SRWeights;
using datamunge::nn::models::SRContributions;
using datamunge::nn::models::SRBicubic;
using datamunge::nn::models::SRLumaTensor;
using datamunge::nn::models::SRLuma;
using datamunge::nn::models::SRY;
using datamunge::nn::models::SRCrop;
using datamunge::nn::models::SRBatch;
using datamunge::nn::models::SRData;
using datamunge::nn::models::SRPSNRTensor;
using datamunge::nn::models::SRPSNR;
using datamunge::nn::models::SRSSIMTensor;
using datamunge::nn::models::SRSSIM;
using datamunge::nn::models::TensorSRMetrics;
using datamunge::nn::models::SRMetrics;
using datamunge::nn::models::SRScoreImages;
using datamunge::nn::models::ScoreTensorSR;
using datamunge::nn::models::SRScore;
using datamunge::nn::models::SRPrepared;
using datamunge::nn::models::PrepareSR;
using datamunge::nn::models::SRPreviewTiles;
inline SRData SRPairs(const data::CifarDataset& source,const std::vector<std::int64_t>& rows){return nn::models::SRPairs(source.images,rows);}
}
