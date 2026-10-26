// Port of tflow's common/ml/recurrent_autoencoder.h + common/nn/recurrent_autoencoder.h:
// a sequence-to-sequence tanh RNN autoencoder over the 28 rows of an MNIST digit
// whose final encoder state is the ONLY bridge to the decoder. The cells are
// nn::rnn_step (tflow's RnnStep: "<name>/Wx", "/Wh", "/b", same initialisers).
#pragma once

#include "common/autoencoder.hpp"

namespace ae {

struct RecurrentAutoencoderOutput {
  nn::Tensor code, reconstruction, encoder_states, decoder_states;  // [B, units], [B, T, F], [B, T, units] x2
};

/// Decoder only: repeat the code, NOT original/previous target rows; all steps share weights.
inline RecurrentAutoencoderOutput RecurrentDecode(nn::ParameterStore& p, const nn::Tensor& code, int steps = 28,
                                                  int features = 28) {
  RecurrentAutoencoderOutput out;
  out.code = code;
  nn::Tensor              h = nn::zeros_like(code);
  std::vector<nn::Tensor> dec, rows;
  for (int t = 0; t < steps; ++t) {
    h = nn::rnn_step(p, "decoder", code, h);
    dec.push_back(h);
    rows.push_back(nn::dense(p, "output", h, features, nn::Activation::Sigmoid));
  }
  out.reconstruction = nn::stack(rows, 1);
  out.decoder_states = nn::stack(dec, 1);
  return out;
}

/// input [B, steps, features].
inline RecurrentAutoencoderOutput RecurrentAutoencoder(nn::ParameterStore& p, const nn::Tensor& input, int steps = 28,
                                                       int features = 28, int units = 32) {
  check::Require(steps > 1 && features > 0 && units > 0 && units < steps * features, "strict sequence bottleneck");
  nn::Tensor              h = nn::zeros({input.size(0), units}, input.dtype());
  std::vector<nn::Tensor> enc;
  for (int t = 0; t < steps; ++t) {
    h = nn::rnn_step(p, "encoder", nn::select(input, 1, t), h);
    enc.push_back(h);
  }
  auto out           = RecurrentDecode(p, h, steps, features);
  out.encoder_states = nn::stack(enc, 1);
  return out;
}

class RecurrentAutoencoderModel {
 public:
  nn::ParameterStore             parameters{1871};
  std::unique_ptr<nn::Optimizer> optimizer;

  explicit RecurrentAutoencoderModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      RecurrentAutoencoder(parameters, nn::zeros({1, 28, 28}));
    }
    if (training) optimizer = MakeAdam(parameters);
  }
  /// [n, 28, 28, 1] viewed as an ordered 28-row sequence, without resize.
  static nn::Tensor Sequence(const nn::Tensor& images) { return nn::reshape(images, {images.size(0), 28, 28}); }
  static nn::Tensor Loss(const nn::Tensor& reconstruction, const nn::Tensor& sequence) {
    return nn::mean(nn::square(reconstruction - sequence));
  }
  /// {code, reconstruction [n, 28, 28], MSE}
  std::vector<nn::Tensor> Evaluate(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            x = Sequence(images);
    auto            n = RecurrentAutoencoder(parameters, x);
    return {n.code, n.reconstruction, Loss(n.reconstruction, x)};
  }
  nn::Tensor Decode(const nn::Tensor& codes) {
    nn::NoGradGuard guard;
    return RecurrentDecode(parameters, codes).reconstruction;
  }
  /// {encoder states, decoder states} [n, 28, 32] each.
  std::vector<nn::Tensor> States(const nn::Tensor& images) {
    nn::NoGradGuard guard;
    auto            n = RecurrentAutoencoder(parameters, Sequence(images));
    return {n.encoder_states, n.decoder_states};
  }
};

}  // namespace ae
