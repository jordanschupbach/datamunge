#pragma once
/// @file wavenet_stack_data.hpp
/// @brief Pinned genuine audio ingress and tensor sufficient statistics/teacher-forced metrics.
#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/training.hpp"
#include "datamunge/nn/wavenet_stack_models.hpp"
#include <filesystem>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
using nn::Tensor;
/// @brief Internal pinned-audio validation and archive lookup helpers.
namespace wave_detail {
/// @brief Throw for a violated pinned audio metadata contract.
/// @param ok Evaluated condition.
/// @param message Failure description.
inline void WaveRequire(bool ok, const std::string &message) {
  if (!ok)
    throw std::invalid_argument(message);
}
/// @brief Find an owned array by exact archive key.
/// @param arrays Loaded named tensor handles.
/// @param key Required key.
/// @return Existing owning tensor handle.
inline Tensor WaveArray(const nn::NamedTensors &arrays, const std::string &key) {
  for (const auto &[name, value] : arrays)
    if (name == key)
      return value;
  throw std::invalid_argument("missing audio array " + key);
}
} // namespace wave_detail
/// @brief Own original audio-window codes and aligned source metadata.
///
/// Tensor handles own storage. Metadata indexes original clips/speakers;
/// no numeric preprocessing or fitted reference is hidden here.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// const auto d = data.pools[0];
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
struct WaveData {
  Tensor input;   ///< Int32[N,256] observed mu-law codes.
  Tensor target;  ///< Int32[N,256] next-sample codes,strict shift.
  Tensor source;  ///< Int32[N] original clip identity0..26.
  Tensor speaker; ///< Int32[N] original speaker identity0..5.
  Tensor start;   ///< Int32[N] original start0/256/512/768.

  /// @brief Return original window count.
  /// @return Input first-axis length.
  [[nodiscard]] std::int64_t size() const {
    return input.size(0);
  }
};
/// @brief Own all original speaker-disjoint pools and held-out prompt codes.
///
/// Pinned custom27clip FSDD subset;not the official take-based test protocol.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
struct WaveSplits {
  std::vector<WaveData> pools; ///< Fit/validation/test speaker-disjoint windows.
  Tensor prompts;              ///< Int32[3,384] real held-out128seed+256reference.
  Tensor prompt_source_ids;    ///< Int32[3] original held-out clip IDs.
};

/// @brief Select aligned original audio windows by metadata row identities.
///
/// Uses tensor take_rows for every numerical/metadata array;no fitting.
///
/// @param d Audio-window container.
/// @param rows Nonempty bounded host row indices.
/// @return New owning/gathered tensor handles for all five fields.
/// @throws std::invalid_argument For invalid row indices.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// const auto batch = m::WaveRows(data.pools[0], {0, 1});
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline WaveData WaveRows(const WaveData &d, const std::vector<std::int64_t> &rows) {
  using nn::take_rows;
  return {take_rows(d.input, rows), take_rows(d.target, rows), take_rows(d.source, rows),
          take_rows(d.speaker, rows), take_rows(d.start, rows)};
}

/// @brief Load the pinned real FSDD arrays with speaker/source/shift validation.
///
/// Int32code bounds/strict sample shifts are tensor checks. Source/speaker
/// sets and unique window identities are topology metadata. No download,
/// audio synthesis,normalization or parameter estimation occurs here.
///
/// @param root Directory containing wavenet_stack.npz and original recordings.
/// @return 72fit/12validation/24test256-code windows and three genuine prompts.
/// @throws std::invalid_argument For missing/invalid arrays,shapes,dtypes,source leakage,shift or prompt codes.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline WaveSplits LoadWaveStack(const std::filesystem::path &root) {
  nn::NoGradGuard guard;
  const auto arrays = nn::load_npz((root / "wavenet_stack.npz").string());
  WaveSplits result;
  std::set<int> previous_sources, previous_speakers;
  std::set<std::pair<int, int>> windows;
  for (const std::string split : {"fit", "validation", "test"}) {
    auto Npz = [&](const nn::NamedTensors &a, const std::string &key) {
      for (const auto &[name, value] : a)
        if (name == key)
          return value;
      throw std::invalid_argument("missing audio array " + key);
    };
    WaveData d{Npz(arrays, split + "_input"), Npz(arrays, split + "_target"),
               Npz(arrays, split + "_source_id"), Npz(arrays, split + "_speaker_id"),
               Npz(arrays, split + "_start")};
    const std::int64_t n = d.input.size(0);
    wave_detail::WaveRequire(
        d.input.dtype() == nn::DType::Int32 && d.target.dtype() == nn::DType::Int32 &&
            d.input.shape() == nn::Shape{n, 256} && d.target.shape() == d.input.shape(),
        "audio input/next-code target matrices");
    for (const auto *t : {&d.source, &d.speaker, &d.start})
      wave_detail::WaveRequire(t->dtype() == nn::DType::Int32 && t->shape() == nn::Shape{n},
                               "audio source metadata alignment");
    wave_detail::WaveRequire(
        nn::all(nn::logical_and(nn::greater_equal(d.input, 0.), nn::less(d.input, 256.)))
                .item<bool>() &&
            nn::all(nn::logical_and(nn::greater_equal(d.target, 0.), nn::less(d.target, 256.)))
                .item<bool>(),
        "bounded mu-law codes");
    wave_detail::WaveRequire(nn::all(nn::equal(nn::slice_axis(d.input, 1, 1, 256),
                                               nn::slice_axis(d.target, 1, 0, 255)))
                                 .item<bool>(),
                             "strict next-sample shift");
    const auto source = d.source.to_vector<int>(), speaker = d.speaker.to_vector<int>(),
               start = d.start.to_vector<int>();
    std::set<int> sources, speakers;
    for (std::int64_t i = 0; i < n; ++i) {
      wave_detail::WaveRequire(source[i] >= 0 && source[i] < 27 && speaker[i] >= 0 &&
                                   speaker[i] < 6 && start[i] >= 0 && start[i] <= 768 &&
                                   start[i] % 256 == 0 && !previous_sources.count(source[i]) &&
                                   !previous_speakers.count(speaker[i]) &&
                                   windows.insert({source[i], start[i]}).second,
                               "audio source/speaker disjointness and unique fixed windows");
      sources.insert(source[i]);
      speakers.insert(speaker[i]);
    }
    previous_sources.insert(sources.begin(), sources.end());
    previous_speakers.insert(speakers.begin(), speakers.end());

    result.pools.push_back(d);
  }
  wave_detail::WaveRequire(result.pools[0].size() == 72 && result.pools[1].size() == 12 &&
                               result.pools[2].size() == 24,
                           "pinned tiny audio pool counts");
  result.prompts = wave_detail::WaveArray(arrays, "prompts");
  result.prompt_source_ids = wave_detail::WaveArray(arrays, "prompt_source_ids");
  wave_detail::WaveRequire(result.prompts.dtype() == nn::DType::Int32 &&
                               result.prompts.shape() == nn::Shape{3, 384} &&
                               result.prompt_source_ids.dtype() == nn::DType::Int32 &&
                               result.prompt_source_ids.shape() == nn::Shape{3},
                           "three held-out 128-seed/256-reference prompts");
  wave_detail::WaveRequire(nn::all(nn::logical_and(nn::greater_equal(result.prompts, 0.),
                                                   nn::less(result.prompts, 256.)))
                               .item<bool>(),
                           "prompt code range");
  for (int id : result.prompt_source_ids.to_vector<int>())
    wave_detail::WaveRequire(previous_sources.count(id) && id >= 21,
                             "held-out prompt original source required");
  return result;
}

/// @brief Count scored fitting audio unigram and transition targets using tensors.
///
/// Parameter-free sufficient statistics;reports fit matching count leaves with
/// explicit SGD before normalization/scoring. Not a hidden fitting optimizer.
///
/// @param fit Original fitting windows.
/// @param start First scored time0..255,default127.
/// @return Double add-one unigram[256] and transition[256,256] COUNT tensors.
/// @throws std::invalid_argument For invalid codes,shape or scored start.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// const auto targets = m::WaveCountTargets(data.pools[0]);
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline std::pair<Tensor, Tensor> WaveCountTargets(const WaveData &fit, int start = 127) {
  (void)WaveNll(nn::zeros({fit.size(), 256, 256}), fit.target, 256, start);
  wave_detail::WaveRequire(
      fit.input.shape() == fit.target.shape() &&
          nn::all(nn::logical_and(nn::greater_equal(fit.input, 0.), nn::less(fit.input, 256.)))
              .item<bool>(),
      "valid observed audio codes");
  const auto y = nn::slice_axis(fit.target, 1, start, 256).reshape({-1}).to(nn::DType::Int64),
             x = nn::slice_axis(fit.input, 1, start, 256).reshape({-1}).to(nn::DType::Int64);
  const auto weights = nn::ones(y.shape(), nn::DType::Float64);
  return {nn::unsorted_segment_sum(weights, y, 256) + 1.,
          (nn::unsorted_segment_sum(weights, x * 256 + y, 65536) + 1.).reshape({256, 256})};
}
/// @brief Broadcast unigram or gather transition scores for every observed audio code.
///
/// No fitting:caller supplies unchanged fitted log counts. Softmax normalizes
/// when evaluating NLL or expected amplitudes.
///
/// @param d Original audio windows.
/// @param prior Floating log-count vector[256] or matrix[256,256].
/// @return Floating[N,256,256] scores preserving prior dtype.
/// @throws std::invalid_argument For invalid prior/input shape or code bounds.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// const auto z = m::WavePriorLogits(data.pools[2], n::zeros({256}));
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline Tensor WavePriorLogits(const WaveData &d, const Tensor &prior) {
  wave_detail::WaveRequire(
      (prior.shape() == nn::Shape{256} || prior.shape() == nn::Shape{256, 256}) &&
          d.input.dim() == 2 && d.input.size(1) == 256 &&
          nn::all(nn::logical_and(nn::greater_equal(d.input, 0.), nn::less(d.input, 256.)))
              .item<bool>(),
      "valid prior and audio input required");
  if (prior.dim() == 1)
    return nn::broadcast_to(prior.reshape({1, 1, 256}), {d.size(), 256, 256});
  return nn::gather(prior, d.input, 0);
}
/// @brief Evaluate teacher-forced audio likelihood,code accuracy and expected-amplitude error.
///
/// All reductions/softmax/argmax/codec arithmetic use tensors. Expected decoded
/// amplitude is sum P(code)*decode(code),not decode(mean code). No rollout
/// or perceptual-quality claim;each position uses true previous input codes.
///
/// @param logits Finite floating[N,256,256] next-code scores.
/// @param d Matching bounded original target codes.
/// @param start First scored sample0..255,default127.
/// @return Double[3]:NLL,mean squared decoded expected amplitude error,exact-code fraction.
/// @throws std::invalid_argument For invalid logits,shape,target codes or warmup.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// namespace m = n::models;
/// const auto data = m::LoadWaveStack("build/cnn-datasets/wavenet");
/// const auto scores = m::ScoreWave(n::zeros({24, 256, 256}), data.pools[2]);
/// @endcode
/// @par References
/// - van den Oord et al.(2016),WaveNet:A Generative Model for Raw Audio, https://arxiv.org/abs/1609.03499. Teaching unconditional stack uses source two-tap causal ingress,12gated layers/dilations1..32 twice,omits the final unused residual projection,and a mu-law256 categorical head;not a paper-quality speech model.
/// @see WaveStack()
inline Tensor ScoreWave(const Tensor &logits, const WaveData &d, int start = 127) {
  wave_detail::WaveRequire(logits.shape() == nn::Shape{d.size(), 256, 256} &&
                               nn::all(nn::is_finite(logits)).item<bool>(),
                           "finite matching all-time audio scores");
  const auto loss = WaveNll(logits, d.target, 256, start).to(nn::DType::Float64);
  const auto z = nn::slice_axis(logits, 1, start, 256).to(nn::DType::Float64),
             y = nn::slice_axis(d.target, 1, start, 256);
  const auto decoded =
      nn::mu_law_decode(nn::arange(0, 256).to(nn::DType::Int32), 256, nn::DType::Float64);
  const auto expected = nn::sum(nn::softmax(z) * decoded, {2});
  return nn::stack(
      {loss, nn::mean(nn::square(expected - nn::mu_law_decode(y, 256, nn::DType::Float64))),
       nn::mean(nn::equal(nn::argmax(z, 2), y).to(nn::DType::Float64))},
      0);
}
} // namespace datamunge::nn::models
