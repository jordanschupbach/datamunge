#pragma once
#include <datamunge/nn/nn.hpp>
#include <datamunge/nn/ocr_data.hpp>
#include "common/checks.hpp"
namespace ocr {namespace nn=datamunge::nn;using nn::Tensor;
using datamunge::nn::models::kAlphabet;
using datamunge::nn::models::kBlank;
using datamunge::nn::models::kSteps;
using datamunge::nn::models::kClasses;
using datamunge::nn::models::CtcCollapse;
using datamunge::nn::models::OcrDistance;
using datamunge::nn::models::OcrData;
using datamunge::nn::models::OcrRows;
using datamunge::nn::models::OcrText;
using datamunge::nn::models::LoadOcr;
using datamunge::nn::models::OcrDecode;
using datamunge::nn::models::OcrMetrics;
}
