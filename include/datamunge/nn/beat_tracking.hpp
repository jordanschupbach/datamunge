#pragma once

/// @file beat_tracking.hpp
/// @brief Spectral-flux onset detection and constant-tempo beat tracking.

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Timing estimates expressed in STFT hop units.
/// A zero period denotes an unresolved pulse.
struct BeatTrack {
  std::vector<double> flux; ///< Positive log-magnitude differences, indexed by the earlier STFT frame.
  std::vector<std::int64_t> onsets; ///< Peak times corrected to the later window center.
  std::vector<std::int64_t> beats; ///< Estimated beat grid around the detected onsets.
  std::int64_t period = 0; ///< Frame lag maximizing the weighted onset autocorrelation.
  std::int64_t phase = 0; ///< Beat-grid residue modulo the period.
};

/// @brief Detect onsets and estimate a constant-tempo beat grid from a mono waveform.
///
/// Uses a periodic Hann STFT, positive log-magnitude differences, and
/// peak picking with a relative threshold and minimum separation.
/// Autocorrelates the weighted peaks over the candidate lags and folds
/// them modulo the strongest lag to estimate phase. Times are in hop units.
/// @param wave Finite rank-one float32 or float64 waveform.
/// @param window STFT window length in samples.
/// @param hop STFT frame stride in samples.
/// @param min_period Smallest candidate beat spacing in hops.
/// @param max_period Largest candidate beat spacing in hops.
/// @param threshold Minimum onset height relative to maximum flux, default 0.14.
/// @param separation Minimum separation between onsets in hops, default 5.
/// @return Flux, picked onsets and estimated beat grid. The grid is empty when no positive autocorrelation is found.
/// @throws std::invalid_argument For invalid shape, dtype or options.
///
/// @par Example
/// @code
/// auto timing = dm::nn::track_beats(dm::nn::zeros({1024}), 256, 80, 3, 6);
/// std::print("{}\n", timing.beats.size());  // 0
/// @endcode
/// @par References
/// - J. P. Bello et al. (2005), A tutorial on onset detection in music signals. doi:10.1109/TSA.2005.851998
/// - D. P. W. Ellis (2007), Beat tracking by dynamic programming. doi:10.1080/09298210701653344
/// @see stft(), magnitude_spectrogram()
inline BeatTrack track_beats(const Tensor& wave, std::int64_t window, std::int64_t hop,
                             std::int64_t min_period, std::int64_t max_period,
                             double threshold = 0.14, std::int64_t separation = 5) {
  if (!wave.defined() || wave.dim() != 1 ||
      (wave.dtype() != DType::Float32 && wave.dtype() != DType::Float64) ||
      window < 1 || hop < 1 || min_period < 1 || max_period < min_period ||
      !(threshold >= 0 && threshold <= 1) || separation < 1) {
    throw std::invalid_argument("Invalid beat tracking waveform or options");
  }
  BeatTrack result;
  const auto samples = wave.size(0);
  if (samples < window || samples - window < hop) return result;
  NoGradGuard guard;
  const auto spectrum = log1p(magnitude_spectrogram(stft(wave, window, hop, window)));
  const auto frames = spectrum.size(0);
  result.flux = sum(relu(slice_axis(spectrum, 0, 1, frames) -
                        slice_axis(spectrum, 0, 0, frames - 1)), {1}).to_vector<double>();
  const auto cutoff = threshold * *std::max_element(result.flux.begin(), result.flux.end());
  std::vector<double> envelope(frames, 0);
  for (std::int64_t i = 1; i + 2 < frames; ++i) {
    const auto strength = result.flux[i];
    if (strength < cutoff || strength <= result.flux[i - 1] || strength < result.flux[i + 1]) continue;
    const auto peak = i + 1 + window / 2 / hop;
    if (peak >= frames || (!result.onsets.empty() && peak - result.onsets.back() < separation)) continue;
    result.onsets.push_back(peak);
    envelope[peak] = strength;
  }
  double best = 0;
  for (auto lag = min_period; lag <= std::min(max_period, frames - 1); ++lag) {
    double score = 0;
    for (auto i = lag; i < frames; ++i) score += envelope[i] * envelope[i - lag];
    if (score > best) {
      best = score;
      result.period = lag;
    }
  }
  if (result.period == 0) return result;
  std::vector<double> folded(result.period, 0);
  for (auto peak : result.onsets) {
    for (int shift : {-1, 0, 1}) {
      folded[(peak + shift) % result.period] += envelope[peak] * (shift == 0 ? 1 : 0.5);
    }
  }
  result.phase = std::max_element(folded.begin(), folded.end()) - folded.begin();
  for (auto i = result.phase; i < frames; i += result.period) {
    if (i >= result.onsets.front() - 3 && i <= result.onsets.back() + 3) result.beats.push_back(i);
  }
  return result;
}

}  // namespace datamunge::nn
