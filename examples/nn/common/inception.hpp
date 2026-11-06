#pragma once
// Compatibility names for canonical tensor components.
#include <datamunge/nn/classic_cnn_components.hpp>
namespace inception {
namespace nn=datamunge::nn;
using datamunge::nn::models::InceptionWidths;
using datamunge::nn::models::InceptionOutputs;
using datamunge::nn::models::InceptionV1;
}
