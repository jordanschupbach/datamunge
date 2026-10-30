// FaceNet NN1-style encoder (eleven ReLU convolutions, two LRN stages, four
// pools, two p=2 maxout dense layers, linear 128 and unit-L2 output), compact
// NN1 triplet training and class-disjoint digit verification/retrieval: a
// port of tflow's common/nn/facenet.h and common/ml/facenet.h.
//
// TF -> datamunge: the model is eager (nn::ParameterStore "facenet", seed 17,
// datamunge's generator, so trained numbers match TF only statistically);
// checkpoints are .npz parameter stores instead of TF SaveV2 bundles; the
// retrieval preview PNG is written with nn::save_png.
#pragma once

#include "common/dataset_assets.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/lrn.hpp"
#include "common/paths.hpp"
#include "common/siamese_data.hpp"
#include "common/triplet.hpp"

namespace metric {

namespace nn = datamunge::nn;

// ---- nn/facenet.h -------------------------------------------------------------------

using nn::models::FaceNetCensus;
using nn::models::FaceNetConfig;
using nn::models::FaceNetKernels;
using nn::models::FaceNetNN1;
using nn::models::FaceNetOutputs;
using nn::models::FaceNetWidths;
using nn::models::MaxoutPairs;
using nn::models::UnitEmbedding;
using nn::models::WeightSquares;
// ---- ml/facenet.h -------------------------------------------------------------------

/// Compact NN1 on MNIST, mean semi-hard triplet + .00005 sum W^2, Adam .001.
class FaceNetModel {
public:
  nn::ParameterStore parameters{17};
  std::unique_ptr<nn::Optimizer> optimizer;
  FaceNetModel() {
    nn::NoGradGuard guard;
    FaceNetNN1(parameters, nn::zeros({1, 28, 28, 1}), FaceNetConfig::Compact());
    nn::OptimizerConfig oc;
    oc.kind = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, oc);
  }
  FaceNetOutputs Forward(const nn::Tensor &x) {
    return FaceNetNN1(parameters, x, FaceNetConfig::Compact());
  }
  nn::Tensor Encode(const nn::Tensor &x) {
    nn::NoGradGuard guard;
    const auto net = Forward(x);
    const nn::Tensor e = net.embedding;
    check::Require(e.dim() == 2 && e.size(0) == x.size(0) && e.size(1) == 128,
                   "embedding shape");
    const auto norm = nn::sum(nn::square(e.to(nn::DType::Float64)), {1});
    check::Require(nn::all(nn::less_equal(nn::abs(norm - 1.), 2e-6)).item<bool>(),
                   "finite unit embedding squared norms");
    return e;
  }
  double Train(const nn::Tensor &x, const std::vector<TripletIndex> &triplets) {
    const nn::Tensor loss = MeanTriplet(Forward(x).embedding, triplets);
    const nn::Tensor objective = loss + .00005 * WeightSquares(parameters);
    const auto grads = nn::grad(objective, parameters.parameters());
    const double value = loss.item();
    check::Require(std::isfinite(value), "finite mined triplet objective");
    optimizer->step(grads);
    return value;
  }
  void Save(const std::string &path) const {
    parameters.save(path);
  }
  void Restore(const std::string &path) {
    parameters.load(path);
  }
};

struct DigitEmbeddingPool {
  nn::Tensor values;
  HostEmbeddings host;
  std::vector<std::size_t> rows;
  std::vector<int> labels;
  std::map<std::size_t, int> index;
};

template <class Model>
inline DigitEmbeddingPool EmbedDigitPool(Model &model, const nn::ImageDataset &d,
                                         const DigitBuckets &buckets) {
  DigitEmbeddingPool pool;
  const auto labels = Labels(d);
  for (const auto &bucket : buckets)
    for (std::size_t row : bucket) {
      pool.index[row] = int(pool.rows.size());
      pool.rows.push_back(row);
      pool.labels.push_back(labels[row]);
    }
  std::vector<nn::Tensor> parts;
  for (std::size_t start = 0; start < pool.rows.size(); start += 64) {
    const auto end = std::min(pool.rows.size(), start + 64);
    parts.push_back(model.Encode(SiameseImages(d, {pool.rows.begin() + std::ptrdiff_t(start),
                                                   pool.rows.begin() + std::ptrdiff_t(end)})));
  }
  pool.values = nn::concat(parts, 0);
  pool.host = HostEmbeddings(pool.values);
  return pool;
}

/// P classes x K = `each` distinct rows, shuffled.
using nn::models::FixedMetricTriplets;
using nn::models::MetricBatchRows;
struct FixedTripletScore {
  double loss = 0, margin_fraction = 0;
};
inline FixedTripletScore ScoreFixedTriplets(const HostEmbeddings &e,
                                            const std::vector<TripletIndex> &indices) {
  if (indices.empty())
    throw std::invalid_argument("nonempty fixed triplets required");
  const auto features = nn::from_vector(e.values, {e.rows, e.dims});
  std::vector<std::int64_t> a, p, n;
  for (const auto &t : indices) {
    a.push_back(t[0]);
    p.push_back(t[1]);
    n.push_back(t[2]);
  }
  const auto delta = nn::models::MetricPairDistance(features, a, p) -
                     nn::models::MetricPairDistance(features, a, n) + .2;
  return {nn::mean(nn::relu(delta)).item(),
          nn::mean(nn::less_equal(delta, 0.).to(nn::DType::Float64)).item()};
}

inline std::vector<double> MetricPairDistances(const DigitEmbeddingPool &pool,
                                               const std::vector<VerificationPair> &pairs) {
  std::vector<std::int64_t> left, right;
  for (const auto &pair : pairs) {
    left.push_back(pool.index.at(pair.left));
    right.push_back(pool.index.at(pair.right));
  }
  return nn::models::MetricPairDistance(pool.values, left, right).to_vector<double>();
}

/// Accuracy-maximising threshold over midpoints of the sorted distances.
inline double CalibrateMetricThreshold(const std::vector<double> &distances,
                                       const std::vector<VerificationPair> &pairs) {
  std::vector<double> labels;
  for (const auto &pair : pairs)
    labels.push_back(pair.same);
  return nn::models::CalibrateMetricDistanceThreshold(nn::from_vector(distances),
                                                      nn::from_vector(labels))
      .item();
}

struct MetricPairScore {
  double accuracy = 0, same = 0, different = 0, auc = 0;
  int tp = 0, fp = 0, fn = 0, tn = 0;
};
inline MetricPairScore ScoreMetricPairs(const std::vector<double> &distances,
                                        const std::vector<VerificationPair> &pairs,
                                        double threshold) {
  std::vector<double> labels;
  for (const auto &pair : pairs)
    labels.push_back(pair.same);
  const auto values = nn::models::MetricDistanceMetrics(nn::from_vector(distances),
                                                        nn::from_vector(labels), threshold)
                          .to_vector<double>();
  return {values[0],      values[1],      values[2],      values[3],
          int(values[4]), int(values[5]), int(values[6]), int(values[7])};
}

struct MetricTasksScore {
  double accuracy = 0;
  std::vector<int> predicted;
  std::vector<std::vector<double>> scores;
};
inline MetricTasksScore ScoreMetricTasks(const DigitEmbeddingPool &pool,
                                         const std::vector<OneShotTask> &tasks) {
  if (tasks.empty())
    throw std::invalid_argument("nonempty metric episodes required");
  std::vector<std::int64_t> left, right, correct;
  const auto ways = tasks.front().support.size();
  for (const auto &task : tasks) {
    if (task.support.size() != ways)
      throw std::invalid_argument("consistent metric ways required");
    correct.push_back(task.correct);
    for (const auto row : task.support) {
      left.push_back(pool.index.at(task.query));
      right.push_back(pool.index.at(row));
    }
  }
  const auto scores = (-nn::models::MetricPairDistance(pool.values, left, right))
                          .reshape({std::int64_t(tasks.size()), std::int64_t(ways)});
  MetricTasksScore result;
  result.accuracy =
      nn::models::SiameseOneShotAccuracy(scores, nn::from_vector(correct)).item();
  result.predicted = nn::argmax(scores, 1).to_vector<int>();
  for (std::size_t i = 0; i < tasks.size(); ++i)
    result.scores.push_back(nn::slice_axis(scores, 0, i, i + 1).to_vector<double>());
  return result;
}

/// retrieval.tsv, embeddings.tsv and an 8-task retrieval.png (query + 3 supports,
/// the predicted support framed green when correct, red otherwise).
inline void ExportMetricTasks(const nn::ImageDataset &d, const std::vector<OneShotTask> &tasks,
                              const MetricTasksScore &result, const DigitEmbeddingPool &pool,
                              const std::filesystem::path &dir) {
  const auto labels = Labels(d);
  std::ofstream scores(dir / "retrieval.tsv"), embeddings(dir / "embeddings.tsv");
  scores << "task\tquery_digit\tsupport_digits\tpredicted_digit\tnegative_squared_distances\n";
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    const auto &t = tasks[i];
    scores << i << '\t' << labels[t.query] << '\t';
    for (int c : t.classes)
      scores << c << ',';
    scores << '\t' << t.classes[result.predicted[i]] << '\t';
    for (double v : result.scores[i])
      scores << v << ',';
    scores << '\n';
  }
  embeddings << "row\tdigit\t128_embedding_coordinates\n";
  for (std::size_t i = 0; i < pool.rows.size(); ++i) {
    embeddings << pool.rows[i] << '\t' << pool.labels[i] << '\t';
    for (int j = 0; j < 128; ++j)
      embeddings << pool.host.at(std::int64_t(i), j) << ',';
    embeddings << '\n';
  }
  std::vector<double> values;
  for (const auto &row : result.scores)
    values.insert(values.end(), row.begin(), row.end());
  const auto tensor = nn::from_vector(
      values, {std::int64_t(tasks.size()), std::int64_t(tasks.front().support.size())});
  nn::save_image_grid(nn::models::SiameseEpisodeGallery(d, tasks, tensor),
                      (dir / "retrieval.png").string(), 1, 0, 1, 0);
  scores.close();
  embeddings.close();
  check::Require(bool(scores) && bool(embeddings), "write metric artifacts");
}

/// MNIST from `data_dir` (nn::load_mnist).
inline std::pair<nn::ImageDataset, nn::ImageDataset>
LoadDigits(const std::filesystem::path &data_dir) {
  auto train = dataset_assets::LoadMnist(data_dir.string(), "train"),
       test = dataset_assets::LoadMnist(data_dir.string(), "test");
  check::Require(train.images.size(0) == 60000 && test.images.size(0) == 10000,
                 "official MNIST sizes");
  return {std::move(train), std::move(test)};
}

} // namespace metric
