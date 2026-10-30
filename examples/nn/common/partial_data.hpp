#pragma once
#include <datamunge/nn/partial_inpainting_data.hpp>
#include "common/inpainting_data.hpp"
namespace inpaint {
using datamunge::nn::models::PartialData;
using datamunge::nn::models::PartialMask;
using datamunge::nn::models::PartialPairs;
using datamunge::nn::models::PartialBatch;
using datamunge::nn::models::PartialMean;
using datamunge::nn::models::ScoreTensorPartial;
using datamunge::nn::models::PartialScore;
using datamunge::nn::models::PartialPreviewTiles;
}
