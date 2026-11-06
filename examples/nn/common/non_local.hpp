#pragma once
// Compatibility names for canonical tensor non-local blocks.
#include <datamunge/nn/nonlocal_models.hpp>
namespace non_local {
namespace nn=datamunge::nn;
using datamunge::nn::models::Pairwise;
using datamunge::nn::models::NonLocalOutputs;
using datamunge::nn::models::NonLocalBlock;
using datamunge::nn::models::NonLocalResNetHook;
}
