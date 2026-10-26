// Port of tflow's common/ml/denoising_autoencoder.h: corrupted-input/clean-target
// training of the dense undercomplete autoencoder with target-free denoising
// inference (the corruption itself is ae::CorruptForDenoising). Training steps
// (forward, clean-target loss, nn::grad / optimizer) are written out in the example.
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

class DenoisingModel {
 public:
  nn::ParameterStore             parameters{1657};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit DenoisingModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      UndercompleteAutoencoder(parameters, nn::zeros({1, 784}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// {code, reconstruction}: inference needs the (corrupted) image ONLY, not a clean target.
  std::vector<nn::Tensor> Predict(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = UndercompleteAutoencoder(parameters, Flatten(images));
    return {n.code, n.reconstruction};
  }
  /// MSE of the reconstruction of `images` against the CLEAN target.
  double Loss(const nn::Tensor& images, const nn::Tensor& clean) {
    nn::NoGradGuard guard;
    return ReconstructionMse(UndercompleteAutoencoder(parameters, Flatten(images)).reconstruction, Flatten(clean))
        .item<float>();
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return nn::dense(parameters, "decoder", codes, 784, nn::Activation::Sigmoid);
  }
};

}  // namespace ae
