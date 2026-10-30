#pragma once
#include <datamunge/nn/colorization_data.hpp>
#include "common/checks.hpp"
#include "common/cifar.hpp"
namespace color { namespace nn=datamunge::nn;
using datamunge::nn::models::ColorTriple;
using datamunge::nn::models::ColorLinear;
using datamunge::nn::models::ColorGamma;
using datamunge::nn::models::ColorLabF;
using datamunge::nn::models::ColorLabInverse;
using datamunge::nn::models::ColorRGBToLab;
using datamunge::nn::models::ColorLabToRGB;
using datamunge::nn::models::ColorData;
using datamunge::nn::models::ColorPairs;
using datamunge::nn::models::ColorBatch;
using datamunge::nn::models::ColorMeanChroma;
using datamunge::nn::models::ColorConstantChroma;
using datamunge::nn::models::ColorMetrics;
using datamunge::nn::models::ColorScore;
using datamunge::nn::models::ColorPreviewTiles;
}
