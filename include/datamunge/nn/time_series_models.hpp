#pragma once
/// @file time_series_models.hpp
/// @brief Full and compact temporal FCN with explicit normalization/BN state policy.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/ops.hpp"
#include <cmath>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Configure temporal FCN width and saved fitting-only scalar normalization.
///
/// Metadata only. Existing checkpoint normalization overrides creation defaults.
/// No statistic is estimated from an inference signal.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// m::SeriesConfig c{16, .2f, 1.3f};
/// @endcode
/// @par References
/// - Wang,Yan and Oates(2017),Time Series Classification from Scratch with Deep Neural Networks:A Strong Baseline, https://arxiv.org/abs/1611.06455. Source FCN uses8/5/3 temporal kernels,BN/ReLU and global average;the teaching fit additionally uses fitting-only global normalization and sequential fitting-only population BN calibration.
/// @see SeriesCnn()
struct SeriesConfig {
  int base = 128; ///< Positive first/last temporal width,middle block doubles it.
  float mean = 0; ///< Finite fitting-only input population mean;default0.
  float standard_deviation =
      1; ///< Positive fitting-only input population standard deviation;default1.
};
/// @brief Own raw temporal classifier logits and retained normalized/intermediate features.
///
/// Handles retain input/learned differentiation graphs;normalization/running
/// statistics are detached. Copying this result shares tensor storage.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(419), state(421);
/// const auto out = m::SeriesCnn(p, state, n::ones({2, 31, 1}), m::SeriesConfig{16, 0, 1});
/// @endcode
/// @par References
/// - Wang,Yan and Oates(2017),Time Series Classification from Scratch with Deep Neural Networks:A Strong Baseline, https://arxiv.org/abs/1611.06455. Source FCN uses8/5/3 temporal kernels,BN/ReLU and global average;the teaching fit additionally uses fitting-only global normalization and sequential fitting-only population BN calibration.
/// @see SeriesCnn()
struct SeriesOutput {
  Tensor logits;     ///< Floating[N,2] raw two-class head scores.
  Tensor normalized; ///< Raw signal centered/scaled by detached checkpoint normalization.
  Tensor features;   ///< Final[N,T,base] BN/ReLU temporal feature sequence.
  std::vector<Tensor> stages; ///< Three SAME-length temporal BN/ReLU activations.
};

/// @brief Select frozen,batch-updating,batch-readonly or fitting-calibration BN policy.
///
/// Calibration is a deliberate fitting-data action and must never run on
/// validation/test/inference observations. Normal inference leaves state fixed.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto mode = m::BnMode::Infer;
/// @endcode
/// @par References
/// - Wang,Yan and Oates(2017),Time Series Classification from Scratch with Deep Neural Networks:A Strong Baseline, https://arxiv.org/abs/1611.06455. Source FCN uses8/5/3 temporal kernels,BN/ReLU and global average;the teaching fit additionally uses fitting-only global normalization and sequential fitting-only population BN calibration.
/// @see SeriesCnn()
enum class BnMode {
  Infer,         ///< Frozen checkpoint population mean/variance;no state update.
  Train,         ///< Batch statistics and momentum.99 moving-state update.
  TrainNoUpdate, ///< Batch statistics without moving-state mutation.
  Calibrate ///< Sequentially set each layer to fitting population statistics before frozen BN.
};

/// @brief Classify raw univariate signals with three SAME temporal convolutions,BN/ReLU and GAP.
///
/// Center/scale by detached saved scalars. Conv widths base/2base/base use
/// kernels8/5/3,stride1 and TensorFlow SAME,including asymmetric even8 padding.
/// Default learned initialization matches conv1d/dense layers. BN eps.001,
/// momentum.99;no dropout or hidden optimizer. Global time mean feeds2logits.
/// Calibrate sets each layer moving moments from ALL fitting inputs before
/// evaluating the next layer frozen;it uses no labels. Caller runs it under
/// NoGradGuard after updates,never on held-out observations. Infer may create
/// missing initial leaves,but repeated warmed inference does not mutate values.
///
/// @param p Mutable learned store;warm all14 leaves before optimizer construction.
/// @param state Separate same-dtype store for2scalar normalization and6BN-state leaves.
/// @param signal Floating[N,T,1],N/T>0,matching both stores dtype.
/// @param config Positive width and finite mean/positive std creation defaults.
/// @param mode Frozen BN by default;explicit Train,TrainNoUpdate or fitting-only Calibrate.
/// @return Raw binary logits and all three feature maps,with temporal length preserved.
/// @throws std::invalid_argument For invalid width/statistics,input/dtype/mode or incompatible stored leaves.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// n::ParameterStore p(419), state(421);
/// const auto out = m::SeriesCnn(p, state, n::ones({2, 31, 1}), m::SeriesConfig{16, 0, 1});
/// @endcode
/// @par References
/// - Wang,Yan and Oates(2017),Time Series Classification from Scratch with Deep Neural Networks:A Strong Baseline, https://arxiv.org/abs/1611.06455. Source FCN uses8/5/3 temporal kernels,BN/ReLU and global average;the teaching fit additionally uses fitting-only global normalization and sequential fitting-only population BN calibration.
/// @see SeriesOutput, BnMode
inline SeriesOutput SeriesCnn(nn::ParameterStore &p, nn::ParameterStore &state,
                              const Tensor &signal, const SeriesConfig &config = {},
                              BnMode mode = BnMode::Infer) {
  if (config.base < 1 || !std::isfinite(config.mean) ||
      !std::isfinite(config.standard_deviation) || config.standard_deviation <= 0)
    throw std::invalid_argument("time-series configuration");
  if (signal.dim() != 3 || signal.size(0) < 1 || signal.size(1) < 1 || signal.size(2) != 1 ||
      signal.dtype() != p.dtype() || state.dtype() != p.dtype())
    throw std::invalid_argument("aligned floating univariate temporal input/stores required");
  if (mode != BnMode::Infer && mode != BnMode::Train && mode != BnMode::TrainNoUpdate &&
      mode != BnMode::Calibrate)
    throw std::invalid_argument("known temporal BN policy required");
  SeriesOutput out;
  const Tensor mean = state.get("normalization/mean", {}, nn::init::constant(config.mean));
  const Tensor stddev =
      state.get("normalization/std", {}, nn::init::constant(config.standard_deviation));
  out.normalized = nn::div(nn::sub(signal, mean.detach()), stddev.detach());
  Tensor x = out.normalized;
  const int kernels[] = {8, 5, 3};
  for (int layer = 0; layer < 3; ++layer) {
    const int width = config.base * (layer == 1 ? 2 : 1);
    const std::string bn = "bn" + std::to_string(layer + 1);
    x = nn::conv1d(p, "conv" + std::to_string(layer + 1), x, width, kernels[layer]);
    nn::BatchNormOptions options;
    options.momentum = .99;
    options.eps = 1e-3;
    options.update_stats = mode == BnMode::Train;
    if (mode == BnMode::Calibrate) {
      nn::batch_norm(p, state, bn, x, false, options); // creates the tensors on first use
      nn::NoGradGuard guard;
      const Tensor population_mean = nn::mean(x, {0, 1});
      const Tensor population_variance =
          nn::mean(nn::square(nn::sub(x, population_mean)), {0, 1});
      state.set(bn + "/moving_mean", population_mean.detach());
      state.set(bn + "/moving_var", population_variance.detach());
    }
    x = nn::relu(nn::batch_norm(
        p, state, bn, x, mode == BnMode::Train || mode == BnMode::TrainNoUpdate, options));
    out.stages.push_back(x);
  }
  out.features = x;
  out.logits = nn::dense(p, "head", nn::mean(x, {1}), 2);
  return out;
}

} // namespace datamunge::nn::models
