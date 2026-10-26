// Tiny variable-length moving-dot sequences on a 4x4 torus; the next-frame
// target never enters inference. Port of tflow's common/ml/moving_dot_data.h
// (std::mt19937 recipe, so every split is bit-identical to tflow's data.npz).
#pragma once

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <random>
#include <vector>

#include "common/recurrent_model.hpp"

namespace rnn_cells {

/// inputs float [n, 4, 4, 4, 1] (time, y, x, 1), lengths int32 [n] in 2..4,
/// labels float [n, 4, 4, 1] (the next frame), source_ids int32 [n], uniforms float64 [n, 4].
struct MovingDots {
  Tensor inputs, lengths, labels, source_ids, uniforms;
  [[nodiscard]] Feed feed() const { return {inputs, lengths, labels}; }
  [[nodiscard]] std::int64_t size() const { return inputs.shape()[0]; }
};

inline MovingDots MakeMovingDots(int n, int seed, int offset) {
  std::vector<float>        inputs(static_cast<std::size_t>(n) * 64, 0.f), labels(static_cast<std::size_t>(n) * 16, 0.f);
  std::vector<std::int32_t> lengths(n), ids(n);
  std::vector<double>       uniforms(static_cast<std::size_t>(n) * 4);
  std::mt19937              rng(static_cast<std::mt19937::result_type>(seed));
  const int                 dy[4] = {0, 0, 1, -1}, dx[4] = {1, -1, 0, 0};
  for (int b = 0; b < n; ++b) {
    double* u = uniforms.data() + b * 4;
    for (int j = 0; j < 4; ++j) u[j] = (double(rng()) + .5) / 4294967296.;
    const int length = 2 + int(3 * u[0]), y = int(4 * u[1]), x = int(4 * u[2]), direction = int(4 * u[3]);
    lengths[b] = length, ids[b] = offset + b;
    for (int t = 0; t <= length; ++t) {
      const int yy = (y + dy[direction] * t + 16) % 4, xx = (x + dx[direction] * t + 16) % 4;
      if (t < length) inputs[static_cast<std::size_t>(b) * 64 + t * 16 + yy * 4 + xx] = 1;
      else labels[static_cast<std::size_t>(b) * 16 + yy * 4 + xx] = 1;
    }
  }
  return {nn::from_vector(inputs, {n, 4, 4, 4, 1}), nn::from_vector(lengths), nn::from_vector(labels, {n, 4, 4, 1}),
          nn::from_vector(ids), nn::from_vector(uniforms, {n, 4})};
}

inline MovingDots MovingDotRows(const MovingDots& x, const std::vector<std::int64_t>& rows) {
  return {nn::take_rows(x.inputs, rows), nn::take_rows(x.lengths, rows), nn::take_rows(x.labels, rows),
          nn::take_rows(x.source_ids, rows), nn::take_rows(x.uniforms, rows)};
}

inline void SaveMovingDots(const std::filesystem::path& file, const MovingDots& x) {
  nn::save_npz(file.string(), {{"inputs", x.inputs},
                               {"lengths", x.lengths},
                               {"labels", x.labels},
                               {"source_ids", x.source_ids},
                               {"uniforms", x.uniforms}});
}

inline MovingDots LoadMovingDots(const std::filesystem::path& file) {
  const auto a = nn::load_npz(file.string());
  return {Find(a, "inputs"), Find(a, "lengths"), Find(a, "labels"), Find(a, "source_ids"), Find(a, "uniforms")};
}

}  // namespace rnn_cells
