#pragma once
// Compatibility names for canonical tensor components.
#include <datamunge/nn/classic_cnn_components.hpp>
namespace inception {
namespace nn=datamunge::nn;
using nn::models::InceptionWidths;
using nn::models::InceptionOutputs;
using nn::models::InceptionV1;
}
