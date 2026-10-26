// Synthetic variable-length single-marker memory task; padding is never a
// model input through the mask. Port of tflow's common/ml/marked_sequence_data.h
// plus the per-example helpers (SaveData, Copy, Accuracy, Heatmap) that the
// tflow 4.1 recurrent-cell examples repeat in every .cc file.
//
// The generator is std::mt19937 exactly as in tflow, so every split is
// bit-identical to tflow's data.npz.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <numeric>
#include <random>
#include <vector>

#include "common/recurrent_model.hpp"

namespace rnn_cells {

/// inputs float [n, 12, 2] (value, marker), lengths/labels/source_ids int32 [n], uniforms float64 [n, 14].
struct MarkedSequences {
  Tensor inputs, lengths, labels, source_ids, uniforms;
  [[nodiscard]] Feed feed() const { return {inputs, lengths, labels}; }
  [[nodiscard]] std::int64_t size() const { return inputs.shape()[0]; }
};

inline MarkedSequences MakeMarkedSequences(int n, int seed, int offset) {
  std::vector<float>        inputs(static_cast<std::size_t>(n) * 24, 0.f);
  std::vector<std::int32_t> lengths(n), labels(n), ids(n);
  std::vector<double>       uniforms(static_cast<std::size_t>(n) * 14);
  std::mt19937              rng(static_cast<std::mt19937::result_type>(seed));
  for (int b = 0; b < n; ++b) {
    double* u = uniforms.data() + b * 14;
    for (int j = 0; j < 14; ++j) u[j] = (double(rng()) + .5) / 4294967296.;
    const int length = 4 + int(9 * u[0]), cue = int((length - 1) * u[1]), label = b % 2;
    lengths[b] = length, labels[b] = label, ids[b] = offset + b;
    float* x = inputs.data() + b * 24;
    for (int t = 0; t < length; ++t) x[t * 2] = u[t + 2] < .5 ? -1.f : 1.f;
    x[cue * 2]     = label ? 1.f : -1.f;
    x[cue * 2 + 1] = 1.f;
  }
  return {nn::from_vector(inputs, {n, 12, 2}), nn::from_vector(lengths), nn::from_vector(labels), nn::from_vector(ids),
          nn::from_vector(uniforms, {n, 14})};
}

inline MarkedSequences MarkedRows(const MarkedSequences& x, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(x.inputs, rows), nn::take_rows(x.lengths, rows), nn::take_rows(x.labels, rows),
          nn::take_rows(x.source_ids, rows), nn::take_rows(x.uniforms, rows)};
}

inline MarkedSequences Copy(const MarkedSequences& x) {
  std::vector<std::int64_t> rows(static_cast<std::size_t>(x.size()));
  std::iota(rows.begin(), rows.end(), 0);
  return MarkedRows(x, rows);
}

inline void SaveData(const std::filesystem::path& file, const MarkedSequences& x) {
  nn::save_npz(file.string(), {{"inputs", x.inputs},
                               {"lengths", x.lengths},
                               {"labels", x.labels},
                               {"source_ids", x.source_ids},
                               {"uniforms", x.uniforms}});
}

inline MarkedSequences LoadData(const std::filesystem::path& file) {
  const auto a = nn::load_npz(file.string());
  return {Find(a, "inputs"), Find(a, "lengths"), Find(a, "labels"), Find(a, "source_ids"), Find(a, "uniforms")};
}

/// The three splits (512 fit / 128 validation / 128 test) and their data.npz arrays.
inline std::vector<MarkedSequences> MarkedPools(nn::NamedTensors* arrays = nullptr) {
  std::vector<MarkedSequences> pools;
  for (int k = 0; k < 3; ++k) {
    const std::string split = k == 0 ? "fit" : k == 1 ? "validation" : "test";
    pools.push_back(MakeMarkedSequences(k ? 128 : 512, 26001 + 2 * k, 10000 * k));
    const auto& x = pools.back();
    if (arrays) {
      arrays->emplace_back(split + "_inputs", x.inputs);
      arrays->emplace_back(split + "_lengths", x.lengths);
      arrays->emplace_back(split + "_labels", x.labels);
      arrays->emplace_back(split + "_source_ids", x.source_ids);
      arrays->emplace_back(split + "_uniforms", x.uniforms);
    }
  }
  return pools;
}

/// Fraction of rows whose logit 1 exceeds logit 0 exactly when the label is 1.
inline double Accuracy(const Tensor& logits, const Tensor& labels) {
  const auto l = logits.to_vector<float>();
  const auto y = labels.to_vector<std::int32_t>();
  int        correct = 0;
  for (std::size_t b = 0; b < y.size(); ++b) correct += (l[b * 2 + 1] > l[b * 2]) == (y[b] != 0);
  return double(correct) / static_cast<double>(y.size());
}

/// tflow's Heatmap: first 24 sequences, 12 columns of 29x15 cells; blue = -1,
/// red = +1, white = 0, black strip = marker. With `gradients`, values are
/// scaled by the global peak |value| of channel 0.
inline void Heatmap(const std::filesystem::path& file, const Tensor& x, bool gradients = false) {
  const int  n = static_cast<int>(std::min<std::int64_t>(24, x.shape()[0]));
  const auto v = x.to_vector<float>();
  auto       at = [&](int b, int t, int c) { return v[(static_cast<std::size_t>(b) * 12 + t) * 2 + c]; };
  std::vector<float> rgb(static_cast<std::size_t>(n) * 16 * 360 * 3, 220.f);
  auto               put = [&](int y, int col, int r, int g, int bl) {
    float* p = rgb.data() + (static_cast<std::size_t>(y) * 360 + col) * 3;
    p[0] = float(r), p[1] = float(g), p[2] = float(bl);
  };
  double peak = 1;
  if (gradients) {
    peak = 0;
    for (int b = 0; b < n; ++b)
      for (int t = 0; t < 12; ++t) peak = std::max(peak, std::abs(double(at(b, t, 0))));
    peak = std::max(peak, 1e-12);
  }
  for (int b = 0; b < n; ++b)
    for (int t = 0; t < 12; ++t) {
      const double value = at(b, t, 0) / peak;
      const int    r = int(std::floor(255 * (1 - std::max(0., -value)) + .5)), g = int(std::floor(255 * (1 - std::abs(value)) + .5)),
                   blue = int(std::floor(255 * (1 - std::max(0., value)) + .5));
      for (int y = 0; y < 15; ++y)
        for (int col = 0; col < 29; ++col) put(b * 16 + y, t * 30 + col, r, g, blue);
      if (!gradients && at(b, t, 1) > 0)
        for (int y = 0; y < 3; ++y)
          for (int col = 0; col < 29; ++col) put(b * 16 + y, t * 30 + col, 0, 0, 0);
    }
  nn::save_png(nn::from_vector(rgb, {n * 16, 360, 3}), file.string(), 0, 255);
}

}  // namespace rnn_cells
