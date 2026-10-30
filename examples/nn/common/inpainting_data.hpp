#pragma once
#include <datamunge/nn/inpainting_data.hpp>
namespace inpaint {namespace nn=datamunge::nn;
using datamunge::nn::models::InpaintData;
using datamunge::nn::models::ContextMask;
using datamunge::nn::models::InpaintBatch;
using datamunge::nn::models::ContextPairs;
using datamunge::nn::models::ContextFillMean;
using datamunge::nn::models::TensorInpaintMetrics;
using datamunge::nn::models::InpaintPSNR;
using datamunge::nn::models::ScoreTensorInpaint;
using datamunge::nn::models::InpaintMetrics;
using datamunge::nn::models::InpaintScore;
using datamunge::nn::models::ContextPreviewTiles;
}
