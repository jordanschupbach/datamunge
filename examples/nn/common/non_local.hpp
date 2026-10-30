#pragma once
// Compatibility names for canonical tensor non-local blocks.
#include <datamunge/nn/nonlocal_models.hpp>
namespace non_local {
namespace nn=datamunge::nn;
using nn::models::Pairwise;
using nn::models::NonLocalOutputs;
using nn::models::NonLocalBlock;
using nn::models::NonLocalResNetHook;
}
