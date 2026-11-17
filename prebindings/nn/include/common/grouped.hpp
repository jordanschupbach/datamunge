#pragma once
// Compatibility names for canonical tensor components.
#include <datamunge/nn/classic_cnn_components.hpp>
namespace grouped {
namespace nn=datamunge::nn;
using datamunge::nn::models::GroupedConv2D;
using datamunge::nn::models::FusedGroupedConv2D;
}
