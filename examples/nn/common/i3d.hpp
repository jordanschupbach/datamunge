#pragma once
#include <datamunge/nn/i3d_models.hpp>
#include "common/c3d.hpp"
namespace i3d{namespace nn=datamunge::nn;using nn::Tensor;using nn::Padding;
using datamunge::nn::models::InflateI3d;
using datamunge::nn::models::InflatedGlorot;
using datamunge::nn::models::I3dConfig;
using datamunge::nn::models::I3dOutput;
using datamunge::nn::models::I3dUnit;
using datamunge::nn::models::I3d;
// ---- compact real-video model (tflow ml/i3d.h) ---------------------------------------------

/// divisor 8, three KTH classes, 16x32x32 clips; parameters seed 677, dropout seed 673.
struct I3dModel {
  nn::ParameterStore parameters{677}, state{679};
  nn::Generator      dropout{673};
  I3dModel() {
    nn::NoGradGuard guard;  // create every learned and moving tensor
    Logits(nn::zeros({1, 16, 32, 32, 3}), false, false);
  }
  Tensor Logits(const Tensor& rgb, bool training, bool update_stats) {
    return I3d(parameters, state, rgb, {8, 3, 16, 32}, training, update_stats, &dropout).logits;
  }
};

}  // namespace i3d
