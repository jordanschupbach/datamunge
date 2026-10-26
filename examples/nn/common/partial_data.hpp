// Fixed, seed-separated irregular masks on original natural RGB images: a
// port of tflow's common/ml/partial_data.h. Masks use tflow's std::mt19937
// (seed + 65537 * row) and std::uniform_int_distribution draws, so they are
// bit-identical to TF's.
#pragma once

#include <algorithm>
#include <random>

#include "common/inpainting_data.hpp"

namespace inpaint {

struct PartialData {
  nn::Tensor truth, input, mask;  // [n,32,32,3] each: RGB [0,1], observed RGB (0 in holes), 3-channel validity
};

inline PartialData PartialPairs(const nn::Tensor& images, const std::vector<std::size_t>& rows, std::uint32_t seed) {
  const nn::Tensor   truth = ContextPairs(images, rows).truth;
  const auto         tv    = truth.to_vector<float>();
  std::vector<float> input(tv.size()), mask(tv.size());
  for (std::size_t n = 0; n < rows.size(); ++n) {
    std::mt19937 rng(std::uint32_t(seed + 65537 * rows[n]));
    auto         integer = [&](int low, int high) { return std::uniform_int_distribution<int>(low, high)(rng); };
    std::vector<int> valid(1024);
    bool             accepted = false;
    for (int attempt = 0; attempt < 100 && !accepted; ++attempt) {
      std::fill(valid.begin(), valid.end(), 1);
      const int y0 = integer(0, 25), x0 = integer(0, 25), rh = integer(5, 12), rw = integer(5, 12);
      for (int y = y0; y < std::min(y0 + rh, 32); ++y)
        for (int x = x0; x < std::min(x0 + rw, 32); ++x) valid[std::size_t(y * 32 + x)] = 0;
      for (int stroke = 0; stroke < 2; ++stroke) {
        const int ay = integer(0, 31), ax = integer(0, 31), by = integer(0, 31), bx = integer(0, 31), radius = integer(1, 3);
        for (int step = 0; step <= 64; ++step) {
          const double cy = ay + (by - ay) * step / 64., cx = ax + (bx - ax) * step / 64.;
          for (int y = std::max(0, int(cy) - radius); y <= std::min(31, int(cy) + radius + 1); ++y)
            for (int x = std::max(0, int(cx) - radius); x <= std::min(31, int(cx) + radius + 1); ++x)
              if ((y - cy) * (y - cy) + (x - cx) * (x - cx) <= radius * radius) valid[std::size_t(y * 32 + x)] = 0;
        }
      }
      const auto holes = std::count(valid.begin(), valid.end(), 0);
      accepted         = holes >= 123 && holes <= 563;
    }
    check::Require(accepted, "irregular mask hole fraction12..55% with visible context");
    for (std::size_t i = 0; i < 1024; ++i)
      for (std::size_t c = 0; c < 3; ++c) {
        const std::size_t j = (n * 1024 + i) * 3 + c;
        mask[j]             = float(valid[i]);
        input[j]            = valid[i] ? tv[j] : 0.f;
      }
  }
  return {truth, nn::from_vector(input, truth.shape()), nn::from_vector(mask, truth.shape())};
}

inline PartialData PartialBatch(const PartialData& data, const std::vector<std::size_t>& rows) {
  return {InpaintBatch(data.truth, rows), InpaintBatch(data.input, rows), InpaintBatch(data.mask, rows)};
}

/// Visible pixels kept, holes filled with the per-image visible mean colour.
inline nn::Tensor PartialMean(const PartialData& data) {
  const auto         in = data.input.to_vector<float>(), m = data.mask.to_vector<float>();
  std::vector<float> out(in.size());
  for (std::int64_t n = 0; n < data.truth.size(0); ++n) {
    double sum[3] = {};
    int    count  = 0;
    for (std::int64_t i = 0; i < 1024; ++i)
      if (m[std::size_t((n * 1024 + i) * 3)] > 0) {
        ++count;
        for (int c = 0; c < 3; ++c) sum[c] += in[std::size_t((n * 1024 + i) * 3 + c)];
      }
    check::Require(count > 0, "visible-only partial mean baseline");
    for (std::int64_t i = 0; i < 1024; ++i)
      for (int c = 0; c < 3; ++c) {
        const std::size_t j = std::size_t((n * 1024 + i) * 3 + c);
        out[j]              = m[j] != 0 ? in[j] : float(sum[c] / count);
      }
  }
  return nn::from_vector(out, data.truth.shape());
}

inline InpaintMetrics PartialScore(const PartialData& data, const nn::Tensor& output, bool clip = false) {
  InpaintData adapter;
  adapter.truth = data.truth;
  adapter.valid = nn::slice(data.mask, {0, 0, 0, 0}, {data.mask.size(0), 32, 32, 1}).contiguous();
  return InpaintScore(adapter, clip ? nn::clip(output, 0.0, 1.0) : output);
}

}  // namespace inpaint
