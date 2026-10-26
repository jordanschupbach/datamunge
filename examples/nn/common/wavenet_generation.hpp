// Per-layer circular-cache autoregressive WaveNet synthesis from trained stack
// weights: port of tflow's common/ml/wavenet_generation.h (the stack itself is
// wavenet_stack.hpp, 1.9).
//
// WaveCache keeps, for each causal block, a ring of its OWN past inputs at that
// block's dilation, so one new code costs one host-side step (float32) instead
// of a full forward pass; the width-2 input convolution remembers the previous
// code. CachedWaveDraw feeds SAMPLED codes back and checks the cache against the
// full native forward (RF-cropped context) at the first 65 steps, every 128th
// step and the last one.
#pragma once

#include <algorithm>
#include <cmath>
#include <random>

#include "common/wavenet_stack.hpp"

namespace wavestack {

inline replay::Map WaveNamedArrays(const WaveStackModel& model) { return replay::Arrays(model.parameters); }
inline replay::Map WaveNamedGradient(WaveStackModel& model, const WaveData& d) {
  return replay::Named(model.parameters, model.Gradients(d));
}

class WaveCache {
 public:
  WaveCache(const replay::Map& weights, WaveStackConfig c) : config(c) {
    for (const auto& [k, v] : weights) arrays[k] = v.contiguous().to_vector<float>();
    for (int i = 0; i < c.levels * c.cycles; ++i) ring.emplace_back(static_cast<std::size_t>((1 << (i % c.levels)) * c.residual), 0.f);
  }
  std::vector<float> Point(const std::string& name, const std::vector<float>& x, int out) const {
    std::vector<float> result(out);
    const auto&        w = arrays.at(name + "/W");
    const auto&        b = arrays.at(name + "/b");
    for (int j = 0; j < out; ++j) {
      float sum = 0;
      for (std::size_t i = 0; i < x.size(); ++i) sum += x[i] * w[i * out + j];
      result[j] = sum + b[j];
    }
    return result;
  }
  /// Feeds one code; returns the logits for the NEXT code.
  std::vector<float> Next(int code) {
    check::Require(code >= 0 && code < config.classes, "valid mu-law code for cached synthesis");
    const int          r = config.residual, sk = config.skip;
    std::vector<float> x(r), skip(sk, 0.f);
    const auto&        iw = arrays.at("input/W");
    const auto&        ib = arrays.at("input/b");
    for (int j = 0; j < r; ++j) x[j] = (previous < 0 ? 0 : iw[previous * r + j]) + iw[(config.classes + code) * r + j] + ib[j];
    previous = code;
    for (int block = 0; block < config.levels * config.cycles; ++block) {
      const int          dilation = 1 << (block % config.levels), slot = time % dilation;
      const std::string  name = "block" + std::to_string(block + 1);
      const auto&        w    = arrays.at(name + "/gate/W");
      const auto&        b    = arrays.at(name + "/gate/b");
      std::vector<float> joined(2 * r), gate(r);
      for (int j = 0; j < 2 * r; ++j) {
        float sum = 0;
        for (int i = 0; i < r; ++i) sum += ring[block][slot * r + i] * w[i * (2 * r) + j];
        for (int i = 0; i < r; ++i) sum += x[i] * w[(r + i) * (2 * r) + j];
        joined[j] = sum + b[j];
      }
      for (int i = 0; i < r; ++i) ring[block][slot * r + i] = x[i];
      for (int i = 0; i < r; ++i) gate[i] = std::tanh(joined[i]) / (1.f + std::exp(-joined[r + i]));
      const auto contribution = Point(name + "/skip", gate, sk);
      for (int j = 0; j < sk; ++j) skip[j] += contribution[j];
      if (block + 1 < config.levels * config.cycles) {
        const auto residual = Point(name + "/residual", gate, r);
        for (int j = 0; j < r; ++j) x[j] += residual[j];
      }
    }
    for (auto& v : skip) v = std::max(0.f, v);
    auto h = Point("post", skip, sk);
    for (auto& v : h) v = std::max(0.f, v);
    ++time;
    return Point("output", h, config.classes);
  }

 private:
  std::map<std::string, std::vector<float>> arrays;
  WaveStackConfig                           config;
  int                                       time = 0, previous = -1;
  std::vector<std::vector<float>>           ring;
};

struct WaveDraw {
  Tensor codes, logits, uniform;
  double parity = 0;
  int    probes = 0;
};

/// tflow's CachedWaveDraw: every row warms the cache on its observed prefix,
/// then draws `count` codes by inverse CDF (u bank [n, count] from mt19937(seed)).
inline WaveDraw CachedWaveDraw(WaveStackModel& model, const replay::Map& arrays, WaveStackConfig config, const Tensor& prefix, int count,
                               std::uint32_t seed) {
  check::Require(prefix.dtype() == nn::DType::Int32 && prefix.dim() == 2 && prefix.size(1) > 0 && count > 0,
                 "nonempty observed prefix and positive generation");
  const int                 n = int(prefix.size(0)), warm = int(prefix.size(1)), field = WaveStackField(config);
  const auto                pv = prefix.to_vector<int>();
  std::vector<std::int32_t> codes(static_cast<std::size_t>(n * (warm + count)));
  std::vector<float>        logits(static_cast<std::size_t>(n * count * 256));
  std::vector<double>       uniform(static_cast<std::size_t>(n * count));
  WaveDraw                  result;
  std::mt19937              rng(seed);
  for (auto& u : uniform) u = (double(rng()) + .5) / 4294967296.;
  const int width = warm + count;
  for (int b = 0; b < n; ++b) {
    WaveCache          cache(arrays, config);
    std::vector<float> scores;
    for (int t = 0; t < warm; ++t) {
      codes[b * width + t] = pv[b * warm + t];
      scores               = cache.Next(pv[b * warm + t]);
    }
    for (int t = 0; t < count; ++t) {
      std::copy(scores.begin(), scores.end(), logits.begin() + (b * count + t) * 256);
      if (t < 65 || t % 128 == 0 || t + 1 == count) {
        const int end = warm + t, start = std::max(0, end - field);
        std::vector<std::int32_t> context(codes.begin() + b * width + start, codes.begin() + b * width + end);
        const auto native = model.Predict(nn::from_vector(context, {1, end - start}), true).to_vector<float>();
        for (int c = 0; c < 256; ++c) result.parity = std::max(result.parity, std::abs(double(scores[c]) - native[c]));
        ++result.probes;
      }
      const double maximum = *std::max_element(scores.begin(), scores.end());
      double       mass = 0, cumulative = 0;
      for (float x : scores) mass += std::exp(double(x) - maximum);
      int code = 255;
      for (int c = 0; c < 256; ++c) {
        cumulative += std::exp(double(scores[c]) - maximum) / mass;
        if (uniform[b * count + t] < cumulative) {
          code = c;
          break;
        }
      }
      codes[b * width + warm + t] = code;
      scores                      = cache.Next(code);
    }
  }
  check::Require(result.parity < 2e-5, "cached synthesis agrees with native RF forward probes");
  result.codes   = nn::from_vector(codes, {n, width});
  result.logits  = nn::from_vector(logits, {n, count, 256});
  result.uniform = nn::from_vector(uniform, {n, count});
  return result;
}

/// tflow's WaveCausalFixture: with the trained weights, d logits[0, 48, 0] /
/// d one-hot input [1, 64, 256] must be exactly zero outside the receptive field
/// (t < 49 - RF) and for t > 48, and active inside. Saves encoded_input/logits/gradient.
inline replay::Map WaveCausalFixture(const replay::Map& arrays, WaveStackConfig config, const Tensor& codes) {
  nn::ParameterStore p(8707);
  for (const auto& [name, value] : arrays) p.set(name, value);
  const auto         cv = codes.to_vector<int>();
  std::vector<float> x(64 * 256, 0.f);
  for (int t = 0; t < 64; ++t) x[t * 256 + cv[t]] = 1;  // row 0 of [N, 256] codes
  Tensor input = nn::from_vector(x, {1, 64, 256});
  input.requires_grad_(true);
  const Tensor logits    = WaveStack(p, input, config).logits;
  const Tensor objective = nn::sum(nn::slice_axis(nn::slice_axis(logits, 1, 48, 49), 2, 0, 1));
  const Tensor gradient  = nn::grad(objective, {input})[0];
  const auto   g         = gradient.to_vector<float>();
  double       reachable = 0;
  for (int t = 0; t < 64; ++t)
    for (int c = 0; c < 256; ++c) {
      const float v = g[t * 256 + c];
      if (t < 49 - WaveStackField(config) || t > 48) check::Require(v == 0, "ALL outside-RF and future encoded input partials EXACT ZERO");
      else reachable += std::abs(v);
    }
  check::Require(reachable > 1e-6, "causal WaveNet input partials active");
  return {{"encoded_input", input.detach()}, {"logits", logits.detach()}, {"gradient", gradient.detach()}};
}

}  // namespace wavestack
