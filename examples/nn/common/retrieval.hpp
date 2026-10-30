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
#include "datamunge/nn/retrieval_models.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/checkpoint.hpp"
#include "datamunge/nn/image_io.hpp"

namespace metric {

using nn::models::DescriptorUnit;
using nn::models::GeneralizedMean;
using nn::models::RetrievalEncoder;
using nn::models::RetrievalResult;
using RetrievalSnapshot = std::vector<std::vector<float>>;

/// Compact GeM retrieval model (divisor 8, 32-D descriptors, Adam .0005).
class RetrievalModel {
public:
  nn::ParameterStore parameters{227};
  std::unique_ptr<nn::Optimizer> optimizer;
  explicit RetrievalModel(bool training = true) {
    {
      nn::NoGradGuard guard;
      RetrievalEncoder(parameters, nn::zeros({1, 32, 32, 3}), 8, 32);
    }
    if (training) {
      nn::OptimizerConfig config;
      config.kind = nn::OptimizerKind::Adam;
      optimizer = std::make_unique<nn::Optimizer>(parameters, .0005, config);
    }
  }
  /// RGB-only unit descriptors [N, 32] (checked unit norm).
  HostEmbeddings Encode(const nn::Tensor &rgb) {
    nn::NoGradGuard guard;
    const HostEmbeddings out(RetrievalEncoder(parameters, rgb, 8, 32).descriptor);
    const auto values =
        nn::from_vector(out.values, {out.rows, out.dims}).to(nn::DType::Float64);
    check::Require(
        nn::all(nn::less_equal(nn::abs(nn::sum(nn::square(values), {1}) - 1.), 2e-6))
            .item<bool>(),
        "finite unit retrieval descriptors");
    return out;
  }
  double Loss(const nn::Tensor &rgb, const std::vector<TripletIndex> &triples,
              bool update = false) {
    check::Require(bool(optimizer), "triplet training graph required");
    std::unique_ptr<nn::NoGradGuard> guard;
    if (!update)
      guard = std::make_unique<nn::NoGradGuard>();
    const nn::Tensor loss =
        MeanTriplet(RetrievalEncoder(parameters, rgb, 8, 32).descriptor, triples, .2f);
    const double value = loss.item();
    check::Require(std::isfinite(value) && value >= 0, "finite triplet hinge");
    if (update)
      optimizer->step(nn::grad(loss, parameters.parameters()));
    return value;
  }
  double Exponent() {
    nn::NoGradGuard guard;
    return (1.0 + nn::softplus(parameters.get("gem/rho", {}, nn::init::zeros()))).item();
  }
  RetrievalSnapshot Snapshot() const {
    RetrievalSnapshot out;
    for (const auto &v : parameters.parameters())
      out.push_back(v.to_vector<float>());
    return out;
  }
  void Save(const std::string &path) const {
    nn::NamedTensors arrays;
    for (std::size_t i = 0; i < parameters.size(); ++i)
      arrays.push_back({parameters.names()[i], parameters.parameters()[i]});
    nn::save_npz(path, arrays);
  }
  void Restore(const std::string &path) {
    nn::NoGradGuard guard;
    const auto arrays = nn::load_npz(path);
    check::Require(arrays.size() == parameters.size(), "complete retrieval checkpoint");
    for (const auto &[name, value] : arrays)
      parameters.set(name, value);
  }
};

inline HostEmbeddings EncodeRetrieval(RetrievalModel &model, const RetrievalPool &data) {
  HostEmbeddings output;
  output.rows = std::int64_t(data.size()), output.dims = 32;
  for (std::size_t start = 0; start < data.size(); start += 20) {
    std::vector<std::size_t> rows;
    for (std::size_t i = start; i < std::min(start + 20, data.size()); ++i)
      rows.push_back(i);
    const auto encoded = model.Encode(RetrievalRows(data, rows).images);
    output.values.insert(output.values.end(), encoded.values.begin(), encoded.values.end());
  }
  return output;
}

inline RetrievalMetrics RetrievalScore(const HostEmbeddings &queries,
                                       const RetrievalPool &query,
                                       const HostEmbeddings &gallery,
                                       const RetrievalPool &pool) {
  return ScoreRetrieval(queries, query, DescriptorIndex(gallery, pool.ids), pool.labels);
}

inline void PrintRetrieval(const std::string &name, const RetrievalMetrics &score) {
  std::cout << " " << name << ":mAP=" << 100 * score.map << "%,hit@1/5=" << 100 * score.hit1
            << "/" << 100 * score.hit5 << "%,queries=" << score.eligible
            << ",without relevant=" << score.without_positive << "\n";
}

/// 8 query rows x (query + top-5 neighbours), 2x upscaled; green/red borders mark relevant/irrelevant, blue the query.
inline void RetrievalPreview(const RetrievalPool &query, const RetrievalPool &gallery,
                             const HostEmbeddings &descriptors, const DescriptorIndex &index,
                             const std::filesystem::path &file) {
  const auto features =
      nn::from_vector(descriptors.values, {descriptors.rows, descriptors.dims});
  nn::save_image_grid(nn::models::RetrievalPreviewTiles(CanonicalRetrievalPool(query),
                                                        CanonicalRetrievalPool(gallery),
                                                        features, index.Canonical()),
                      file.string(), 1, 0, 1, 0);
}

inline nn::Tensor RetrievalIds(const std::vector<std::int64_t> &input) {
  return nn::from_vector(input, {std::int64_t(input.size())});
}
inline nn::Tensor RetrievalLabels(const std::vector<int> &input) {
  return nn::from_vector(std::vector<std::int32_t>(input.begin(), input.end()),
                         {std::int64_t(input.size())});
}
inline nn::Tensor DescriptorTensor(const HostEmbeddings &e) {
  return nn::from_vector(e.values, {e.rows, e.dims});
}

} // namespace metric
