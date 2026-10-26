// Port of tflow's common/{nn,ml}/tied_autoencoder.h: a true shared-weight autoencoder whose
// decoder is the live transpose of ONE encoder parameter, and the matched-initial
// independent-decoder control. Training steps (forward, MSE, nn::grad / optimizer)
// are written out in the example.
#pragma once

#include <map>

#include "common/autoencoder.hpp"

namespace ae {

struct TiedAutoencoderOutput {
  nn::Tensor encoder_weight, decoder_weight, code, reconstruction;
};

/// "encoder/W" [features, latent] Glorot uniform, "encoder/b"; code = ReLU(x W + eb);
/// reconstruction = sigmoid(code D + db) with D = transpose(W) when tied, else an
/// independent "decoder/W" [latent, features] of the same Glorot limit.
inline TiedAutoencoderOutput TiedAutoencoder(nn::ParameterStore& p, const nn::Tensor& image, std::int64_t features = 784,
                                             std::int64_t latent = 32, bool tied = true) {
  check::Require(features > 1 && latent > 0 && latent < features, "positive strict undercomplete tied architecture");
  const double          limit = std::sqrt(6.0 / double(features + latent));
  TiedAutoencoderOutput out;
  out.encoder_weight = p.get("encoder/W", {features, latent}, nn::init::uniform(limit));
  auto eb            = p.get("encoder/b", {latent}, nn::init::zeros());
  out.code           = nn::relu(nn::bias_add(nn::matmul(image, out.encoder_weight), eb));
  out.decoder_weight = tied ? nn::transpose(out.encoder_weight, {1, 0})
                            : p.get("decoder/W", {latent, features}, nn::init::uniform(limit));
  auto db            = p.get("decoder/b", {features}, nn::init::zeros());
  out.reconstruction = nn::sigmoid(nn::bias_add(nn::matmul(out.code, out.decoder_weight), db));
  return out;
}

class TiedAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1853};
  std::unique_ptr<nn::Optimizer> optimizer;
  bool                           tied;

  explicit TiedAutoencoderModel(bool tied_ = true, bool training = true) : tied(tied_) {
    {
      nn::NoGradGuard guard;
      TiedAutoencoder(parameters, nn::zeros({1, 784}), 784, 32, tied);
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  TiedAutoencoderOutput Net(const nn::Tensor& images) {
    return TiedAutoencoder(parameters, Flatten(images), 784, 32, tied);
  }
  /// {code, reconstruction, loss}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = Net(images);
    return {n.code, n.reconstruction, ReconstructionMse(n.reconstruction, Flatten(images))};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    auto            d = tied ? nn::transpose(parameters.at("encoder/W"), {1, 0}) : parameters.at("decoder/W");
    return nn::sigmoid(nn::bias_add(nn::matmul(codes, d), parameters.at("decoder/b")));
  }
  /// Loads every parameter by name (all must be supplied with matching shapes).
  void Load(const nn::NamedTensors& arrays) {
    check::Require(arrays.size() == parameters.size(), "all initial model tensors supplied");
    for (const auto& name : parameters.names()) {
      auto value = Find(arrays, name);
      check::Require(value.dtype() == nn::DType::Float32 && value.shape() == parameters.at(name).shape(),
                     "matching initial shape/dtype");
      parameters.set(name, value);
    }
  }
  /// The decoder weight the forward pass actually uses.
  nn::Tensor DecoderWeight() {
    nn::NoGradGuard guard;
    return Net(nn::zeros({1, 784})).decoder_weight.contiguous();
  }
  void CheckTie() {
    check::Require(!parameters.contains("decoder/W") && parameters.size() == 3,
                   "decoder graph directly transposes same encoder parameter");
    auto d = DecoderWeight();
    check::Require(d.shape() == nn::Shape({32, 784}), "actual decoder transpose shape");
    const auto dv = d.to_vector<float>(), w = parameters.at("encoder/W").contiguous().to_vector<float>();
    for (int i = 0; i < 784; ++i)
      for (int j = 0; j < 32; ++j)
        check::Near(dv[j * 784 + i], w[i * 32 + j], 0, "ALL25088current decoder coordinates equal current encoder transpose");
  }
};

}  // namespace ae
