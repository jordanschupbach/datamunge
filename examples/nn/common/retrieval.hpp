// CNN descriptors with learned generalized-mean (GeM) pooling, a short
// batch-hard triplet training, exact gallery search and restored retrieval:
// a port of tflow's common/nn/retrieval.h (encoder) and common/ml/retrieval.h
// (RetrievalModel, TrainRetrieval).
// TF -> datamunge: the checkpoint is best.npz (all 28 parameter tensors); the
// "deployment graph inspection" becomes a structural check on a second model
// built without an optimizer (TF inspects GraphDef placeholders). PNGs are
// written with nn::save_png.
#pragma once

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>

#include "common/retrieval_index.hpp"

namespace metric {

/// (mean_{H,W} max(x, eps)^p)^(1/p) for x [N, H, W, C] and a scalar exponent p.
inline nn::Tensor GeneralizedMean(const nn::Tensor& input, const nn::Tensor& exponent, float epsilon = 1e-6f) {
  const nn::Tensor clamped = nn::maximum(input, nn::scalar_like(double(epsilon), input));
  return nn::pow(nn::mean(nn::pow(clamped, exponent), {1, 2}), nn::div(1.0, exponent));
}

/// Rows divided by sqrt(max(|row|^2, 1e-12)).
inline nn::Tensor DescriptorUnit(const nn::Tensor& input) {
  return input / nn::sqrt(nn::maximum(nn::sum(nn::square(input), {1}, true), nn::scalar_like(double(1e-12f), input)));
}

struct RetrievalResult {
  nn::Tensor              descriptor, pooled, raw, exponent;
  std::vector<nn::Tensor> stages;
};

/// VGG-16-style 13 convolutions (widths / divisor, He normal, ReLU) on rgb - .5,
/// 4 max pools, GeM with shared learnable p = 1 + softplus(rho) (initially 3),
/// a bias-free linear projection and unit descriptors.
inline RetrievalResult RetrievalEncoder(nn::ParameterStore& p, const nn::Tensor& rgb, int divisor = 1, int dimensions = 128) {
  if (divisor < 1 || 64 % divisor || dimensions < 1) throw std::invalid_argument("retrieval divisor must divide64;positive descriptor dimensions");
  RetrievalResult      result;
  nn::Tensor           x = rgb - .5;
  nn::ConvLayerOptions config;
  config.init       = nn::init::he_normal();
  config.activation = nn::Activation::Relu;
  const int counts[] = {2, 2, 3, 3, 3}, widths[] = {64, 128, 256, 512, 512};
  for (int block = 0; block < 5; ++block) {
    for (int layer = 0; layer < counts[block]; ++layer) {
      x = nn::conv2d(p, "block" + std::to_string(block) + "/conv" + std::to_string(layer), x, widths[block] / divisor, 3, config);
      result.stages.push_back(x);
    }
    if (block < 4) {
      x = nn::max_pool2d(x, 2, 2);
      result.stages.push_back(x);
    }
  }
  const std::int64_t in  = x.size(3);
  const nn::Tensor   rho = p.get("gem/rho", {}, nn::init::constant(double(std::log(std::expm1(2.f)))));
  result.exponent        = 1.0 + nn::softplus(rho);
  result.pooled          = GeneralizedMean(x, result.exponent);
  result.stages.push_back(result.pooled);
  const nn::Tensor weights = p.get("projection/W", {in, dimensions}, nn::init::normal(std::sqrt(1.f / float(in))));
  result.raw               = nn::matmul(result.pooled, weights);
  result.descriptor        = DescriptorUnit(result.raw);
  result.stages.push_back(result.descriptor);
  return result;
}

using RetrievalSnapshot = std::vector<std::vector<float>>;

/// Compact GeM retrieval model (divisor 8, 32-D descriptors, Adam .0005).
class RetrievalModel {
 public:
  nn::ParameterStore             parameters{227};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit RetrievalModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      RetrievalEncoder(parameters, nn::zeros({1, 32, 32, 3}), 8, 32);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer   = std::make_unique<nn::Optimizer>(parameters, .0005, config);
    }
  }
  /// RGB-only unit descriptors [N, 32] (checked unit norm).
  HostEmbeddings Encode(const nn::Tensor& rgb) {
    nn::NoGradGuard guard;
    const HostEmbeddings out(RetrievalEncoder(parameters, rgb, 8, 32).descriptor);
    for (std::int64_t n = 0; n < out.rows; ++n) {
      double norm = 0;
      for (std::int64_t c = 0; c < 32; ++c) {
        const double value = out.at(n, c);
        check::Require(std::isfinite(value), "finite descriptor");
        norm += value * value;
      }
      check::Near(norm, 1, 2e-6, "unit retrieval descriptor");
    }
    return out;
  }
  double Loss(const nn::Tensor& rgb, const std::vector<TripletIndex>& triples, bool update = false) {
    check::Require(bool(optimizer), "triplet training graph required");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update) guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss  = MeanTriplet(RetrievalEncoder(parameters, rgb, 8, 32).descriptor, triples, .2f);
    const double     value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite triplet hinge");
    if (update) optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  double Exponent() {
    nn::NoGradGuard guard;
    return (1.0 + nn::softplus(parameters.get("gem/rho", {}, nn::init::zeros()))).item();
  }
  RetrievalSnapshot Snapshot() const {
    RetrievalSnapshot out;
    for (const auto& v : parameters.parameters()) out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string& path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i) arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string& path) {
    nn::NoGradGuard guard;
    const auto      arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete retrieval checkpoint");
    for (const auto& [name, value] : arrays) parameters.set(name, value);
  }
};

inline HostEmbeddings EncodeRetrieval(RetrievalModel& model, const RetrievalPool& data) {
  HostEmbeddings output;
  output.rows = std::int64_t(data.size()), output.dims = 32;
  for (std::size_t start = 0; start < data.size(); start += 20) {
    std::vector<std::size_t> rows;
    for (std::size_t i = start; i < std::min(start + 20, data.size()); ++i) rows.push_back(i);
    const auto encoded = model.Encode(RetrievalRows(data, rows).images);
    output.values.insert(output.values.end(), encoded.values.begin(), encoded.values.end());
  }
  return output;
}

inline RetrievalMetrics RetrievalScore(const HostEmbeddings& queries, const RetrievalPool& query, const HostEmbeddings& gallery,
                                       const RetrievalPool& pool) {
  return ScoreRetrieval(queries, query, DescriptorIndex(gallery, pool.ids), pool.labels);
}

inline void PrintRetrieval(const std::string& name, const RetrievalMetrics& score) {
  std::cout << " " << name << ":mAP=" << 100 * score.map << "%,hit@1/5=" << 100 * score.hit1 << "/" << 100 * score.hit5
            << "%,queries=" << score.eligible << ",without relevant=" << score.without_positive << "\n";
}

/// 8 query rows x (query + top-5 neighbours), 2x upscaled; green/red borders mark relevant/irrelevant, blue the query.
inline void RetrievalPreview(const RetrievalPool& query, const RetrievalPool& gallery, const HostEmbeddings& descriptors,
                             const DescriptorIndex& index, const std::filesystem::path& file) {
  const std::vector<int> rows = {0, 16, 32, 1, 17, 33, 2, 18};
  const int              W    = 6 * 64;
  std::vector<float>     pixels(std::size_t(8 * 64) * W * 3, 0.f);
  for (int n = 0; n < 8; ++n) {
    const int  row     = rows[std::size_t(n)];
    const auto ranking = index.Rank(descriptors, row, query.ids[std::size_t(row)]);
    for (int col = 0; col < 6; ++col) {
      const auto& image    = col ? gallery.pixels : query.pixels;
      const int   source   = col ? int(ranking[std::size_t(col - 1)].row) : row;
      const bool  relevant = col && gallery.labels[std::size_t(source)] == query.labels[std::size_t(row)];
      for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
          for (int c = 0; c < 3; ++c) {
            const bool border = y < 2 || x < 2 || y >= 62 || x >= 62;
            const int  color  = col ? (relevant ? 1 : 0) : 2;
            pixels[(std::size_t(n * 64 + y) * W + std::size_t(col * 64 + x)) * 3 + std::size_t(c)] =
                border ? (c == color ? 255.f : 0.f)
                       : float(std::clamp(int(std::lround(255 * image[std::size_t(source) * 3072 + std::size_t(((y / 2) * 32 + x / 2) * 3 + c)])), 0, 255));
          }
    }
  }
  nn::save_png(nn::from_vector(pixels, {8 * 64, W, 3}), file.string(), 0, 255);
}

inline nn::Tensor RetrievalIds(const std::vector<std::int64_t>& input) { return nn::from_vector(input, {std::int64_t(input.size())}); }
inline nn::Tensor RetrievalLabels(const std::vector<int>& input) {
  return nn::from_vector(std::vector<std::int32_t>(input.begin(), input.end()), {std::int64_t(input.size())});
}
inline nn::Tensor DescriptorTensor(const HostEmbeddings& e) { return nn::from_vector(e.values, {e.rows, e.dims}); }

}  // namespace metric
