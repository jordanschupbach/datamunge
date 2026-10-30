#pragma once
// Compatibility names for canonical tensor components.
#include <datamunge/nn/classic_cnn_components.hpp>
namespace grouped {
namespace nn=datamunge::nn;
using nn::models::GroupedConv2D;
using nn::models::FusedGroupedConv2D;
}
