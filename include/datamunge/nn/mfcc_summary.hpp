#pragma once

/// @file mfcc_summary.hpp
/// @brief Fixed-length cepstral summaries of batched log-mel spectrograms.

#include <cstdint>
#include <stdexcept>

#include "datamunge/nn/audio.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

/// @brief Summarize orthonormal MFCCs by their temporal means and standard deviations.
/// Applies an orthonormal DCT-II along the mel axis, retaining coefficients
/// starting at zero. Standard deviations use the population denominator.
/// @param log_mel Floating [batch, frames, bands] log-mel features with nonempty axes.
/// @param coefficients Number of retained coefficients, from one to bands, default 13.
/// @return Feature matrix [batch, 2*coefficients], means followed by standard deviations.
/// @throws std::invalid_argument For invalid dimensions or dtype.
/// @par Example
/// @code
/// auto features = dm::nn::mfcc_summary(dm::nn::zeros({2,49,32}));
/// @endcode
/// @see dct(), log_mel_spectrogram()
inline Tensor mfcc_summary(const Tensor& log_mel, std::int64_t coefficients = 13) {
  if (!log_mel.defined() || log_mel.dim()!=3 || log_mel.size(0)<1 || log_mel.size(1)<1 ||
      coefficients<1 || coefficients>log_mel.size(2) ||
      (log_mel.dtype()!=DType::Float32 && log_mel.dtype()!=DType::Float64)) {
    throw std::invalid_argument("Invalid log-mel tensor or MFCC coefficient count");
  }
  const auto cepstra=slice_axis(dct(log_mel,2,true),2,0,coefficients);
  const auto center=mean(cepstra,{1},true);
  const auto deviation=sqrt(mean(square(cepstra-center),{1}));
  return concat({squeeze(center,{1}),deviation},1);
}

}  // namespace datamunge::nn
