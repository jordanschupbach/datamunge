#pragma once
#include <datamunge/nn/denoising_data.hpp>
#include "common/checks.hpp"
namespace denoise {namespace nn=datamunge::nn;
using datamunge::nn::models::DnGray;
using datamunge::nn::models::DnImages;
using datamunge::nn::models::DnBatch;
using datamunge::nn::models::DnData;
using datamunge::nn::models::DnCorrupt;
using datamunge::nn::models::DnReflect;
using datamunge::nn::models::DnBoxFilter;
using datamunge::nn::models::DnMetrics;
using datamunge::nn::models::DnPSNR;
using datamunge::nn::models::DnScore;
using datamunge::nn::models::DnPreviewTiles;
}
