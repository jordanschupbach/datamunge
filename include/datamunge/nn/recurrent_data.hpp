#pragma once
/// @file recurrent_data.hpp
/// @brief Original marked-token and moving-dot fixtures and exact source heatmap rasterization.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/recurrent_components.hpp>
#include <limits>
#include <random>
namespace datamunge::nn::models {
/// @brief Own the detached original marked-token memory fixture.
///
/// All fields own independent CPU storage beyond generator/record collection. No model, optimizer,
/// sampling cache or external files. Source random draws are recorded explicitly.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;m::MarkedSequenceData data;
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Repository teaching-fixture protocol and full editable example:
/// docs/mepml/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.mepml . These
/// marked-token and moving-dot datasets are local controlled fixtures.
struct MarkedSequenceData {
  Tensor inputs;     ///< Float32 CPU[N,12,2]: signed token and marker channels.
  Tensor lengths;    ///< Int32 CPU[N] valid sequence lengths.
  Tensor labels;     ///< Int32 CPU[N] alternating class IDs0/1.
  Tensor source_ids; ///< Int32 CPU[N] caller-offset provenance IDs.
  Tensor uniforms;   ///< Double CPU[N,14] exact source midpoint-MT draw bank.
};
/// @brief Own the detached original moving-dot next-frame fixture.
///
/// All fields own independent CPU storage beyond generator/record collection. No model, optimizer,
/// sampling cache or external files. Source random draws are recorded explicitly.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;m::MovingDotData data;
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Repository teaching-fixture protocol and full editable example:
/// docs/mepml/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.mepml . These
/// marked-token and moving-dot datasets are local controlled fixtures.
struct MovingDotData {
  Tensor inputs;     ///< Float32 CPU[N,4,4,4,1]: time/y/x/channel moving-dot inputs.
  Tensor lengths;    ///< Int32 CPU[N] valid sequence lengths.
  Tensor labels;     ///< Float32 CPU[N,4,4,1] binary next-frame targets.
  Tensor source_ids; ///< Int32 CPU[N] caller-offset provenance IDs.
  Tensor uniforms;   ///< Double CPU[N,4] exact source midpoint-MT draw bank.
};
/// @brief Generate the original single-marker memory sequences.
///
/// Consume exactly14 Double midpoint uniforms per row; lengths4..12, one cue in0..length-2,
/// alternating target class, padded inputs zero. Local RNG only; global/store RNG untouched. No
/// fitting or external data. Work/storage O(n*24).
///
/// @param n Row count in[1,1000000].
/// @param seed Local UInt32 MT19937 seed, default26001.
/// @param offset Int32 source-ID offset, default0; offset+n-1 must fit Int32.
/// @return Owning detached source observations, lengths, labels, IDs and all draw words.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto data=m::MakeMarkedSequences(3);
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Repository teaching-fixture protocol and full editable example:
/// docs/mepml/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.mepml . These
/// marked-token and moving-dot datasets are local controlled fixtures.
inline MarkedSequenceData
MakeMarkedSequences(std::int64_t n, std::uint32_t seed = 26001, std::int32_t offset = 0) {
  if (n < 1 || n > 1000000 ||
      std::int64_t(offset) + n - 1 > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("recurrent fixture count or IDs");
  std::vector<float>        inputs(static_cast<std::size_t>(n) * 24, 0.f);
  std::vector<std::int32_t> lengths(n), labels(n), ids(n);
  std::vector<double>       uniforms(static_cast<std::size_t>(n) * 14);
  std::mt19937              rng(static_cast<std::mt19937::result_type>(seed));
  for (int b = 0; b < n; ++b) {
    double* u = uniforms.data() + b * 14;
    for (int j = 0; j < 14; ++j)
      u[j] = (double(rng()) + .5) / 4294967296.;
    const int length = 4 + int(9 * u[0]), cue = int((length - 1) * u[1]), label = b % 2;
    lengths[b] = length, labels[b] = label, ids[b] = offset + b;
    float* x = inputs.data() + b * 24;
    for (int t = 0; t < length; ++t)
      x[t * 2] = u[t + 2] < .5 ? -1.f : 1.f;
    x[cue * 2]     = label ? 1.f : -1.f;
    x[cue * 2 + 1] = 1.f;
  }
  return {
      from_vector(inputs, {n, 12, 2}),
      from_vector(lengths),
      from_vector(labels),
      from_vector(ids),
      from_vector(uniforms, {n, 14})};
}
/// @brief Generate the original variable-length moving-dot sequences.
///
/// Consume exactly4 Double midpoint uniforms per row; lengths2..4, cardinal unit motion on a4x4
/// torus, target at the next step and padded inputs zero. Local RNG only; global/store RNG
/// untouched. No fitting or external data. Work/storage O(n*64).
///
/// @param n Row count in[1,1000000].
/// @param seed Local UInt32 MT19937 seed, default37001.
/// @param offset Int32 source-ID offset, default0; offset+n-1 must fit Int32.
/// @return Owning detached source observations, lengths, labels, IDs and all draw words.
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto data=m::MakeMovingDotSequences(3);
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Repository teaching-fixture protocol and full editable example:
/// docs/mepml/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.mepml . These
/// marked-token and moving-dot datasets are local controlled fixtures.
inline MovingDotData
MakeMovingDotSequences(std::int64_t n, std::uint32_t seed = 37001, std::int32_t offset = 0) {
  if (n < 1 || n > 1000000 ||
      std::int64_t(offset) + n - 1 > std::numeric_limits<std::int32_t>::max())
    throw std::invalid_argument("recurrent fixture count or IDs");
  std::vector<float> inputs(static_cast<std::size_t>(n) * 64, 0.f),
      labels(static_cast<std::size_t>(n) * 16, 0.f);
  std::vector<std::int32_t> lengths(n), ids(n);
  std::vector<double>       uniforms(static_cast<std::size_t>(n) * 4);
  std::mt19937              rng(static_cast<std::mt19937::result_type>(seed));
  const int                 dy[4] = {0, 0, 1, -1}, dx[4] = {1, -1, 0, 0};
  for (int b = 0; b < n; ++b) {
    double* u = uniforms.data() + b * 4;
    for (int j = 0; j < 4; ++j)
      u[j] = (double(rng()) + .5) / 4294967296.;
    const int length = 2 + int(3 * u[0]), y = int(4 * u[1]), x = int(4 * u[2]),
              direction = int(4 * u[3]);
    lengths[b] = length, ids[b] = offset + b;
    for (int t = 0; t <= length; ++t) {
      const int yy = (y + dy[direction] * t + 16) % 4, xx = (x + dx[direction] * t + 16) % 4;
      if (t < length)
        inputs[static_cast<std::size_t>(b) * 64 + t * 16 + yy * 4 + xx] = 1;
      else
        labels[static_cast<std::size_t>(b) * 16 + yy * 4 + xx] = 1;
    }
  }
  return {
      from_vector(inputs, {n, 4, 4, 4, 1}),
      from_vector(lengths),
      from_vector(labels, {n, 4, 4, 1}),
      from_vector(ids),
      from_vector(uniforms, {n, 4})};
}
/// @brief Rasterize the source marked-token or input-gradient preview without writing files.
///
/// Display exactly the first min(N,24) rows. Source29x15 colored cells have one-pixel gray220 gaps;
/// data cues have a three-pixel black top strip. Gradient normalization floors the displayed peak
/// at1e-12 and omits cue strips. Float64 input is source-rounded to Float32 for display; nonfinite
/// conversion rejects. Input observations/graphs remain unchanged. No model, RNG or fitting.
/// O(N*24) validation and O(min(N,24)*16*360) output work/storage.
///
/// @param x Finite floating CPU[N,12,2], N>0. Data mode channel0 must be in[-1,1]; positive
/// channel1 marks cues.
/// @param gradients False(default): source data colors/cue strip. True: channel0 normalized by its
/// maximum absolute value over the displayed rows.
/// @return Owning detached Float32 RGB byte-scale CPU[min(N,24)*16,360,3].
/// @throws std::invalid_argument Invalid finite CPU geometry, dtype, domain or parameter shape.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto
/// image=m::MarkedSequenceHeatmap(m::MakeMarkedSequences(24).inputs);
/// @endcode
/// @see MaskedGatedSequence(), RecurrentGradientControl()
/// @par References
/// - Repository teaching-fixture protocol and full editable example:
/// docs/mepml/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.mepml . These
/// marked-token and moving-dot datasets are local controlled fixtures.
inline Tensor MarkedSequenceHeatmap(const Tensor& x, bool gradients = false) {
  recurrent_component_detail::Floating(x);
  if (x.dim() != 3 || x.size(1) != 12 || x.size(2) != 2)
    throw std::invalid_argument("marked heatmap geometry");
  NoGradGuard guard;
  const int   n = static_cast<int>(std::min<std::int64_t>(24, x.shape()[0]));
  const auto  v = x.to_vector<float>();
  for (std::size_t k = 0; k < v.size(); ++k)
    if (!std::isfinite(v[k]) || (!gradients && k % 2 == 0 && (v[k] < -1 || v[k] > 1)))
      throw std::invalid_argument("marked heatmap finite range");
  /// Read one source-rounded displayed channel.
  auto at = [&](int b, int t, int c) { return v[(static_cast<std::size_t>(b) * 12 + t) * 2 + c]; };
  std::vector<float> rgb(static_cast<std::size_t>(n) * 16 * 360 * 3, 220.f);
  /// Set one in-bounds RGB byte-scale pixel in the owning raster buffer.
  auto put = [&](int y, int col, int r, int g, int bl) {
    float* p = rgb.data() + (static_cast<std::size_t>(y) * 360 + col) * 3;
    p[0] = float(r), p[1] = float(g), p[2] = float(bl);
  };
  double peak = 1;
  if (gradients) {
    peak = 0;
    for (int b = 0; b < n; ++b)
      for (int t = 0; t < 12; ++t)
        peak = std::max(peak, std::abs(double(at(b, t, 0))));
    peak = std::max(peak, 1e-12);
  }
  for (int b = 0; b < n; ++b)
    for (int t = 0; t < 12; ++t) {
      const double value = at(b, t, 0) / peak;
      const int    r     = int(std::floor(255 * (1 - std::max(0., -value)) + .5)),
                g        = int(std::floor(255 * (1 - std::abs(value)) + .5)),
                blue     = int(std::floor(255 * (1 - std::max(0., value)) + .5));
      for (int y = 0; y < 15; ++y)
        for (int col = 0; col < 29; ++col)
          put(b * 16 + y, t * 30 + col, r, g, blue);
      if (!gradients && at(b, t, 1) > 0)
        for (int y = 0; y < 3; ++y)
          for (int col = 0; col < 29; ++col)
            put(b * 16 + y, t * 30 + col, 0, 0, 0);
    }
  return from_vector(rgb, {n * 16, 360, 3});
}

/// @brief Rasterize the original moving-dot input/target/prediction panels without file I/O.
/// Display the first min(N,16) rows: four input frames, target frame, predicted
/// probability. Each4x4 frame uses5x5 source-rounded grayscale cells at stride6;
/// six25-pixel panels share gray180 gaps. Display clipping to[0,1] changes no
/// observations. Float64 values source-round to Float32 before rendering;
/// nonfinite conversion rejects. O(input elements) validation and O(min(N,16))
/// fixed-size raster work; output is detached with no model/RNG/optimizer.
/// @param inputs Finite Float32/64 CPU[N,4,4,4,1], positive N.
/// @param labels Finite floating CPU[N,4,4,1] presentation-only target frames.
/// @param probability Finite floating CPU[N,4,4,1] presentation-only predictions.
/// @return Owning detached Float32 RGB byte-scale CPU[min(N,16)*25,150,3].
/// @throws std::invalid_argument Invalid geometry, dtype, device or finite display conversion.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace m=n::models;auto data=m::MakeMovingDotSequences(2);auto
/// image=m::MovingDotPanels(data.inputs,data.labels,n::zeros({2,4,4,1}));
/// @endcode
/// @see MakeMovingDotSequences(), MaskedConvLstmSequence()
/// @par References
/// - Repository teaching-fixture protocol and full editable example:
/// docs/mepml/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.mepml . These
/// marked-token and moving-dot datasets are local controlled fixtures.
inline Tensor
MovingDotPanels(const Tensor& inputs, const Tensor& labels, const Tensor& probability) {
  recurrent_component_detail::Floating(inputs);
  recurrent_component_detail::Floating(labels);
  recurrent_component_detail::Floating(probability);
  if (inputs.dim() != 5 || inputs.size(1) != 4 || inputs.size(2) != 4 || inputs.size(3) != 4 ||
      inputs.size(4) != 1 || labels.shape() != Shape{inputs.size(0), 4, 4, 1} ||
      probability.shape() != labels.shape())
    throw std::invalid_argument("moving-dot panel geometry");
  NoGradGuard guard;
  const int   n  = int(std::min<std::int64_t>(16, inputs.size(0)));
  const auto  in = inputs.to_vector<float>(), lab = labels.to_vector<float>(),
             p = probability.to_vector<float>();
  for (const auto& values : {in, lab, p})
    for (float value : values)
      if (!std::isfinite(value))
        throw std::invalid_argument("moving-dot source-rounded display values");
  std::vector<float> rgb(static_cast<std::size_t>(n) * 25 * 150 * 3, 180.f);
  for (int b = 0; b < n; ++b)
    for (int panel = 0; panel < 6; ++panel)
      for (int y = 0; y < 4; ++y)
        for (int col = 0; col < 4; ++col) {
          const double v     = panel < 4    ? in[b * 64 + panel * 16 + y * 4 + col]
                               : panel == 4 ? lab[b * 16 + y * 4 + col]
                                            : p[b * 16 + y * 4 + col];
          const int    shade = int(std::floor(255 * std::clamp(v, 0., 1.) + .5));
          for (int yy = 0; yy < 5; ++yy)
            for (int xx = 0; xx < 5; ++xx)
              for (int ch = 0; ch < 3; ++ch)
                rgb[(static_cast<std::size_t>(b * 25 + y * 6 + yy) * 150 + panel * 25 + col * 6 +
                     xx) *
                        3 +
                    ch] = float(shade);
        }
  return from_vector(rgb, {n * 25, 150, 3});
}
} // namespace datamunge::nn::models
