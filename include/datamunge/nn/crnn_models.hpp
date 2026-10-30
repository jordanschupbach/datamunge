#pragma once
/// @file crnn_models.hpp
/// @brief Seven-convolution and two-projected-BLSTM tensor word recognizer.
#include <limits>

#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/model_plot.hpp"
#include "datamunge/nn/ops.hpp"
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure exact convolution-width division and recurrent hidden size.
///
/// Reference divisor1,H256;compact divisor8,H32. Integer architecture metadata only.
///
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible stored
/// parameters.
/// @par Example
/// @code
/// const dm::nn::models::CrnnConfig compact{8, 32};
/// @endcode
/// @par References
/// - Shi,Bai and Yao(2015),An End-to-End Trainable Neural Network for Image-based Sequence
/// Recognition and Its Application to Scene Text Recognition, https://arxiv.org/abs/1507.05717.
/// Repository follows explicit symmetric width padding before asymmetric pooling,zero LSTM forget
/// bias and two projected BLSTMs;compact widths are teaching choices.
/// @see Crnn(), CrnnBiLstm()
struct CrnnConfig {
  int divisor = 1;   ///< Positive exact divisor of64;8 gives compact channels.
  int hidden  = 256; ///< Positive hidden width of each LSTM direction;compact32.
};

/// @brief Own CRNN time-major logits and every convolution stage.
///
/// Tensor handles retain the graph;no CTC labels or optimizer are required by forward.
///
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible stored
/// parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(307), state(311);
/// const auto out = n::models::Crnn(p, state, n::zeros({1, 32, 100, 1}), {8, 32}, false);
/// @endcode
/// @par References
/// - Shi,Bai and Yao(2015),An End-to-End Trainable Neural Network for Image-based Sequence
/// Recognition and Its Application to Scene Text Recognition, https://arxiv.org/abs/1507.05717.
/// Repository follows explicit symmetric width padding before asymmetric pooling,zero LSTM forget
/// bias and two projected BLSTMs;compact widths are teaching choices.
/// @see Crnn(), CrnnBiLstm()
struct CrnnOutput {
  Tensor              logits; ///< Time-major raw CTC logits[26,N,37],blank36.
  std::vector<Tensor> stages; ///< Seven post-ReLU convolution stages,before pooling.
};

/// @brief Run two directional LSTMs and one shared affine projection.
///
/// Architecture tracing groups the complete time-unrolled bidirectional sequence
/// and its projection; no numeric forward values or ordinary gradients change.
///
/// Zero initial states,forget bias0,shared weights across time,Glorot projection.
/// Right states are returned to original time order. Retains input/parameter
/// gradients;host loops enumerate time metadata only. No fitting or BN state.
///
/// @param p Mutable learned store,warm all recurrent/projection leaves before optimizer.
/// @param name Prefix for /forward,/backward and /projection parameters.
/// @param x Floating[N,T,F],all positive,matching p dtype.
/// @param hidden Positive hidden width in each direction.
/// @param output Positive projected width.
/// @return Floating[N,T,output] projection of concatenated forward/backward hidden states.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible stored
/// parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(307), state(311);
/// const auto out = n::models::CrnnBiLstm(p, "b", n::ones({2, 3, 4}), 5, 6);
/// @endcode
/// @par References
/// - Shi,Bai and Yao(2015),An End-to-End Trainable Neural Network for Image-based Sequence
/// Recognition and Its Application to Scene Text Recognition, https://arxiv.org/abs/1507.05717.
/// Repository follows explicit symmetric width padding before asymmetric pooling,zero LSTM forget
/// bias and two projected BLSTMs;compact widths are teaching choices.
/// @see Crnn(), CrnnBiLstm()
inline Tensor CrnnBiLstm(
    nn::ParameterStore& p,
    const std::string&  name,
    const Tensor&       x,
    std::int64_t        hidden,
    std::int64_t        output) {
  if (x.dim() != 3 || x.size(0) < 1 || x.size(1) < 1 || x.size(2) < 1 || hidden < 1 || output < 1 ||
      x.dtype() != p.dtype())
    throw std::invalid_argument("positive floating BTF projected BLSTM input required");
  const auto        xs = nn::time_steps(x);
  const std::size_t T  = xs.size();
  const auto        N  = x.shape()[0];
  nn::LstmState     fw = nn::lstm_zero_state(N, hidden, x.dtype()),
                bw     = nn::lstm_zero_state(N, hidden, x.dtype());
  std::vector<Tensor> left(T), right(T), both(T);
  for (std::size_t t = 0; t < T; ++t) {
    fw                  = nn::lstm_step(p, name + "/forward", xs[t], fw, 0.0);
    left[t]             = fw.h;
    const std::size_t r = T - 1 - t;
    bw                  = nn::lstm_step(p, name + "/backward", xs[r], bw, 0.0);
    right[r]            = bw.h;
  }
  for (std::size_t t = 0; t < T; ++t)
    both[t] = nn::concat({left[t], right[t]}, 1);
  nn::DenseOptions projection;
  projection.init = nn::init::glorot_uniform();
  const auto projected =
      nn::dense(p, name + "/projection", nn::stack_time(both), output, projection);
  return nn::model_plot_detail::recording()
             ? nn::model_stage("Bidirectional LSTM / " + name, x, projected)
             : projected;
}

/// @brief Recognize word images as26 unaligned CTC time steps.
///
/// Input scales to(255*pixels-128)/128. Seven convolutions,three BN/ReLU,
/// two ordinary and two asymmetric max pools,then2x2 VALID convolution yield
/// height1,width26. Symmetric -infinity padding adds a column on BOTH sides
/// before each asymmetric pool. Two projected BLSTMs produce37 logits,blank36.
/// No softmax,targets or hidden optimizer. Save p and state together for inference.
///
/// @param p Mutable learned convolution,BN-affine,LSTM and projection store.
/// @param state Mutable separate BN moving statistics.
/// @param pixels Floating[N,32,100,1] grayscale[0,1],positive N,matching p dtype.
/// @param config Validated divisor/hidden metadata,default reference widths.
/// @param training Use batch BN moments when true,default true.
/// @param update_stats Refresh BN only during training and true,default true.
/// @return Raw[26,N,37] logits and seven convolution stage activations.
/// @throws std::invalid_argument For invalid geometry,configuration,dtype or incompatible stored
/// parameters.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ParameterStore p(307), state(311);
/// const auto out = n::models::Crnn(p, state, n::zeros({2, 32, 100, 1}), {8, 32}, true, false);
/// @endcode
/// @par References
/// - Shi,Bai and Yao(2015),An End-to-End Trainable Neural Network for Image-based Sequence
/// Recognition and Its Application to Scene Text Recognition, https://arxiv.org/abs/1507.05717.
/// Repository follows explicit symmetric width padding before asymmetric pooling,zero LSTM forget
/// bias and two projected BLSTMs;compact widths are teaching choices.
/// @see Crnn(), CrnnBiLstm()
inline CrnnOutput Crnn(
    nn::ParameterStore& p,
    nn::ParameterStore& state,
    const Tensor&       pixels,
    const CrnnConfig&   config       = {},
    bool                training     = true,
    bool                update_stats = true) {
  if (config.divisor < 1 || 64 % config.divisor || config.hidden < 1 || pixels.dim() != 4 ||
      pixels.size(0) < 1 || pixels.shape() != nn::Shape{pixels.size(0), 32, 100, 1} ||
      pixels.dtype() != p.dtype())
    throw std::invalid_argument("CRNN width configuration");
  CrnnOutput result;
  Tensor     x = nn::div(nn::sub(nn::mul(pixels, 255.0), 128.0), 128.0);
  /// @brief Reference convolution widths before exact metadata division.
  const int widths[] = {64, 128, 256, 256, 512, 512, 512};
  for (int layer = 0; layer < 7; ++layer) {
    const int            out = widths[layer] / config.divisor;
    nn::ConvLayerOptions opt;
    opt.padding = layer == 6 ? nn::Padding::Valid : nn::Padding::Same;
    x = nn::conv2d(p, "conv" + std::to_string(layer + 1), x, out, layer == 6 ? 2 : 3, opt);
    if (layer == 2 || layer == 4 || layer == 6) {
      nn::BatchNormOptions bn;
      bn.momentum     = .9;
      bn.eps          = 1e-5;
      bn.update_stats = update_stats;
      x               = nn::batch_norm(p, state, "bn" + std::to_string(layer + 1), x, training, bn);
    }
    x = nn::relu(x);
    result.stages.push_back(x);
    if (layer == 0 || layer == 1)
      x = nn::max_pool2d(x, 2, 2);
    if (layer == 3 || layer == 5) {
      // Torch padding is one column on BOTH sides; TensorFlow SAME differs.
      x = nn::pad(
          x,
          {{0, 0}, {0, 0}, {1, 1}, {0, 0}},
          nn::PadMode::Constant,
          -std::numeric_limits<double>::infinity());
      x = nn::max_pool2d(x, {2, 2}, {2, 1});
    }
  }
  const Tensor sequence = nn::squeeze(x, {1}); // [N, 26, C]
  const Tensor first    = CrnnBiLstm(p, "blstm1", sequence, config.hidden, config.hidden);
  const Tensor second   = CrnnBiLstm(p, "blstm2", first, config.hidden, 37);
  result.logits         = nn::transpose(second, {1, 0, 2});
  return result;
}

} // namespace datamunge::nn::models
