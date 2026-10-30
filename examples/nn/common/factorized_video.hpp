#pragma once
#include <datamunge/nn/factorized_video_models.hpp>
#include "common/c3d.hpp"
namespace factorized_video{namespace nn=datamunge::nn;using nn::Tensor;using nn::Padding;
using datamunge::nn::models::FactorMidplanes;
using datamunge::nn::models::FactorConv;
using datamunge::nn::models::P3dKind;
using datamunge::nn::models::P3dMix;
using datamunge::nn::models::FactorVideoConfig;
using datamunge::nn::models::FactorVideoOutput;
using datamunge::nn::models::FactorUnit;
using datamunge::nn::models::FactorVideo;
// ---- compact real-video model (tflow ml/factorized_video.h) ------------------------------

/// base 4, three KTH classes on 16x32x32 clips; parameters seed 727, dropout seed 719.
struct FactorVideoModel {
  bool               p3d;
  nn::ParameterStore parameters{727}, state{729};
  nn::Generator      dropout{719};
  explicit FactorVideoModel(bool p3d_model) : p3d(p3d_model) {
    nn::NoGradGuard guard;  // create every learned and moving tensor
    Logits(nn::zeros({1, 16, 32, 32, 3}), false, false);
  }
  Tensor Logits(const Tensor& rgb, bool training, bool update_stats) {
    return FactorVideo(parameters, state, rgb, {p3d, 4, 3}, training, update_stats, &dropout).logits;
  }
};
struct R2Plus1dModel : FactorVideoModel {
  R2Plus1dModel() : FactorVideoModel(false) {}
};
struct P3dModel : FactorVideoModel {
  P3dModel() : FactorVideoModel(true) {}
};

}  // namespace factorized_video
