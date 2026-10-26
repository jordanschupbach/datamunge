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

/// x / sqrt(max(sum(x^2, axis 1), epsilon)) per row.
inline nn::Tensor UnitEmbedding(const nn::Tensor& x, float epsilon = 1e-12f) {
  return x * nn::rsqrt(nn::maximum(nn::sum(nn::square(x), {1}, true), nn::scalar_like(double(epsilon), x)));
}

/// Max over adjacent pairs: [B, 2 outputs] -> [B, outputs].
inline nn::Tensor MaxoutPairs(const nn::Tensor& x, std::int64_t outputs) {
  return nn::max(nn::reshape(x, {-1, outputs, 2}), {2});
}

struct FaceNetConfig {
  int side = 220, input = 3, divisor = 1, hidden = 4096;
  static FaceNetConfig Compact() { return {28, 1, 8, 128}; }
};

struct FaceNetOutputs {
  nn::Tensor              raw, embedding;
  std::vector<nn::Tensor> stages;
};

inline const std::array<int, 11>& FaceNetWidths() {
  static const std::array<int, 11> w = {64, 64, 192, 192, 384, 384, 256, 256, 256, 256, 256};
  return w;
}
inline const std::array<int, 11>& FaceNetKernels() {
  static const std::array<int, 11> k = {7, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3};
  return k;
}

inline FaceNetOutputs FaceNetNN1(nn::ParameterStore& p, const nn::Tensor& input, const FaceNetConfig& cfg) {
  static const std::array<const char*, 11> names = {"conv1", "conv2a", "conv2", "conv3a", "conv3", "conv4a",
                                                    "conv4", "conv5a", "conv5", "conv6a", "conv6"};
  if (cfg.divisor < 1 || cfg.side < 1 || cfg.input < 1 || cfg.hidden < 1) throw std::invalid_argument("FaceNet dimensions");
  FaceNetOutputs out;
  nn::Tensor     x = input;
  for (int i = 0; i < 11; ++i) {
    if (FaceNetWidths()[i] % cfg.divisor) throw std::invalid_argument("FaceNet width divisor");
    nn::ConvLayerOptions c;
    c.activation = nn::Activation::Relu;
    c.init       = nn::init::he_normal();
    c.stride     = i == 0 ? 2 : 1;
    x = nn::conv2d(p, names[i], x, FaceNetWidths()[i] / cfg.divisor, FaceNetKernels()[i], c);
    out.stages.push_back(x);
    if (i == 2) x = lrn::LocalResponseNorm(x, 2, 1.0, 1e-4, .75);
    if (i == 0 || i == 2 || i == 4 || i == 10) x = nn::max_pool2d(x, 3, 2, nn::Padding::Same);
    if (i == 0) x = lrn::LocalResponseNorm(x, 2, 1.0, 1e-4, .75);
  }
  nn::DenseOptions dense;
  dense.init = nn::init::he_normal();
  x = MaxoutPairs(nn::dense(p, "fc1", nn::reshape(x, {x.size(0), -1}), cfg.hidden * 2, dense), cfg.hidden);
  out.stages.push_back(x);
  x = MaxoutPairs(nn::dense(p, "fc2", x, cfg.hidden * 2, dense), cfg.hidden);
  out.stages.push_back(x);
  dense.init    = nn::init::glorot_uniform();
  out.raw       = nn::dense(p, "embedding", x, 128, dense);
  out.embedding = UnitEmbedding(out.raw);
  return out;
}

/// Parameter count of FaceNetNN1 from the layer shapes alone (no allocation;
/// tflow builds the full 220 px model as a graph only).
inline std::int64_t FaceNetCensus(const FaceNetConfig& cfg, int* tensors = nullptr) {
  std::int64_t total = 0, channels = cfg.input, side = cfg.side;
  int          count = 0;
  for (int i = 0; i < 11; ++i) {
    const std::int64_t next = FaceNetWidths()[i] / cfg.divisor, k = FaceNetKernels()[i];
    total += k * k * channels * next + next;
    count += 2;
    channels = next;
    if (i == 0) side = nn::conv_output_size(side, k, 2, 1, nn::Padding::Same);
    if (i == 0 || i == 2 || i == 4 || i == 10) side = nn::conv_output_size(side, 3, 2, 1, nn::Padding::Same);
  }
  const std::int64_t flat = side * side * channels, h = cfg.hidden;
  total += flat * 2 * h + 2 * h + h * 2 * h + 2 * h + h * 128 + 128;
  count += 6;
  if (tensors) *tensors = count;
  return total;
}

/// sum of squares of every parameter of rank > 1 (tflow's L2 term before the .5 l2 factor).
inline nn::Tensor WeightSquares(const nn::ParameterStore& p) {
  nn::Tensor total;
  for (const auto& v : p.parameters())
    if (v.dim() > 1) total = total.defined() ? total + nn::sum(nn::square(v)) : nn::sum(nn::square(v));
  return total;
}

// ---- ml/facenet.h -------------------------------------------------------------------

/// Compact NN1 on MNIST, mean semi-hard triplet + .00005 sum W^2, Adam .001.
class FaceNetModel {
 public:
  nn::ParameterStore             parameters{17};
  std::unique_ptr<nn::Optimizer> optimizer;
  FaceNetModel() {
    nn::NoGradGuard guard;
    FaceNetNN1(parameters, nn::zeros({1, 28, 28, 1}), FaceNetConfig::Compact());
    nn::OptimizerConfig oc;
    oc.kind   = nn::OptimizerKind::Adam;
    optimizer = std::make_unique<nn::Optimizer>(parameters, .001, oc);
  }
  FaceNetOutputs Forward(const nn::Tensor& x) { return FaceNetNN1(parameters, x, FaceNetConfig::Compact()); }
  nn::Tensor Encode(const nn::Tensor& x) {
    nn::NoGradGuard guard;
    const auto      net = Forward(x);
    const nn::Tensor e  = net.embedding;
    check::Require(e.dim() == 2 && e.size(0) == x.size(0) && e.size(1) == 128, "embedding shape");
    const auto v = e.contiguous().to_vector<float>();
    for (std::int64_t i = 0; i < e.size(0); ++i) {
      double norm = 0;
      for (int j = 0; j < 128; ++j) {
        const double value = v[std::size_t(i * 128 + j)];
        check::Require(std::isfinite(value), "finite embedding");
        norm += value * value;
      }
      if (std::abs(norm - 1) > 2e-6) {
        std::cerr << "invalid embedding batch row " << i << "; stage maximum magnitudes:";
        auto stages = net.stages;
        stages.push_back(net.raw);
        for (const auto& s : stages) {
          const auto row = nn::select(s, 0, i);
          std::cerr << ' ' << nn::max(nn::abs(row)).item();
        }
        std::cerr << std::endl;
      }
      check::Near(norm, 1, 2e-6, "unit embedding squared norm");
    }
    return e;
  }
  double Train(const nn::Tensor& x, const std::vector<TripletIndex>& triplets) {
    const nn::Tensor loss      = MeanTriplet(Forward(x).embedding, triplets);
    const nn::Tensor objective = loss + .00005 * WeightSquares(parameters);
    const auto       grads     = nn::grad(objective, parameters.parameters());
    const double     value     = loss.item();
    check::Require(std::isfinite(value), "finite mined triplet objective");
    optimizer->step(grads);
    return value;
  }
  void Save(const std::string& path) const { parameters.save(path); }
  void Restore(const std::string& path) { parameters.load(path); }
};

struct DigitEmbeddingPool {
  nn::Tensor                       values;
  HostEmbeddings                   host;
  std::vector<std::size_t>         rows;
  std::vector<int>                 labels;
  std::map<std::size_t, int>       index;
};

template <class Model>
inline DigitEmbeddingPool EmbedDigitPool(Model& model, const nn::ImageDataset& d, const DigitBuckets& buckets) {
  DigitEmbeddingPool pool;
  const auto         labels = Labels(d);
  for (const auto& bucket : buckets)
    for (std::size_t row : bucket) {
      pool.index[row] = int(pool.rows.size());
      pool.rows.push_back(row);
      pool.labels.push_back(labels[row]);
    }
  std::vector<nn::Tensor> parts;
  for (std::size_t start = 0; start < pool.rows.size(); start += 64) {
    const auto end = std::min(pool.rows.size(), start + 64);
    parts.push_back(model.Encode(SiameseImages(d, {pool.rows.begin() + std::ptrdiff_t(start), pool.rows.begin() + std::ptrdiff_t(end)})));
  }
  pool.values = nn::concat(parts, 0);
  pool.host   = HostEmbeddings(pool.values);
  return pool;
}

/// P classes x K = `each` distinct rows, shuffled.
inline std::vector<std::size_t> MetricBatchRows(const DigitBuckets& buckets, int each, int seed) {
  std::mt19937             rng(seed);
  std::vector<std::size_t> rows;
  for (int c : SiameseClasses(buckets)) {
    auto candidates = buckets[c];
    check::Require(each >= 2 && std::size_t(each) <= candidates.size(), "unique P-by-K batch size");
    std::shuffle(candidates.begin(), candidates.end(), rng);
    rows.insert(rows.end(), candidates.begin(), candidates.begin() + each);
  }
  std::shuffle(rows.begin(), rows.end(), rng);
  return rows;
}

inline std::vector<TripletIndex> FixedMetricTriplets(const std::vector<int>& labels, int count, int seed) {
  std::mt19937              rng(seed);
  std::vector<TripletIndex> out;
  for (int i = 0; i < count; ++i) {
    const int        a = std::uniform_int_distribution<int>(0, int(labels.size()) - 1)(rng);
    std::vector<int> positives, negatives;
    for (std::size_t j = 0; j < labels.size(); ++j)
      if (int(j) != a) (labels[j] == labels[a] ? positives : negatives).push_back(int(j));
    check::Require(!positives.empty() && !negatives.empty(), "valid fixed triplet classes");
    const int p = positives[std::uniform_int_distribution<std::size_t>(0, positives.size() - 1)(rng)];
    const int n = negatives[std::uniform_int_distribution<std::size_t>(0, negatives.size() - 1)(rng)];
    out.push_back({a, p, n});
  }
  return out;
}

struct FixedTripletScore {
  double loss = 0, margin_fraction = 0;
};
inline FixedTripletScore ScoreFixedTriplets(const HostEmbeddings& e, const std::vector<TripletIndex>& indices) {
  FixedTripletScore score;
  for (const auto& t : indices) {
    const double delta = EmbeddingDistance(e, t[0], t[1]) - EmbeddingDistance(e, t[0], t[2]) + .2;
    score.loss += std::max(0., delta);
    score.margin_fraction += delta <= 0;
  }
  score.loss /= double(indices.size());
  score.margin_fraction /= double(indices.size());
  return score;
}

inline std::vector<double> MetricPairDistances(const DigitEmbeddingPool& pool, const std::vector<VerificationPair>& pairs) {
  std::vector<double> distances;
  for (const auto& p : pairs) {
    const double distance = EmbeddingDistance(pool.host, pool.index.at(p.left), pool.index.at(p.right));
    check::Require(distance >= 0 && distance <= 4.00001, "unit sphere squared distance range");
    distances.push_back(distance);
  }
  return distances;
}

/// Accuracy-maximising threshold over midpoints of the sorted distances.
inline double CalibrateMetricThreshold(const std::vector<double>& distances, const std::vector<VerificationPair>& pairs) {
  check::Require(!distances.empty() && distances.size() == pairs.size(), "verification calibration inputs");
  auto sorted = distances;
  std::sort(sorted.begin(), sorted.end());
  std::vector<double> candidates = {-1};
  for (std::size_t i = 1; i < sorted.size(); ++i)
    if (sorted[i] != sorted[i - 1]) candidates.push_back((sorted[i] + sorted[i - 1]) / 2);
  candidates.push_back(4.00001);
  int    best     = -1;
  double selected = -1;
  for (double threshold : candidates) {
    int correct = 0;
    for (std::size_t i = 0; i < distances.size(); ++i) correct += (distances[i] <= threshold) == bool(pairs[i].same);
    if (correct > best) best = correct, selected = threshold;
  }
  return selected;
}

struct MetricPairScore {
  double accuracy = 0, same = 0, different = 0, auc = 0;
  int    tp = 0, fp = 0, fn = 0, tn = 0;
};
inline MetricPairScore ScoreMetricPairs(const std::vector<double>& distances, const std::vector<VerificationPair>& pairs,
                                        double threshold) {
  MetricPairScore     score;
  std::vector<double> positive, negative;
  for (std::size_t i = 0; i < pairs.size(); ++i) {
    const bool predicted = distances[i] <= threshold;
    if (pairs[i].same) {
      positive.push_back(distances[i]);
      score.same += distances[i];
      if (predicted) ++score.tp; else ++score.fn;
    } else {
      negative.push_back(distances[i]);
      score.different += distances[i];
      if (predicted) ++score.fp; else ++score.tn;
    }
  }
  check::Require(!positive.empty() && !negative.empty(), "balanced verification metric classes");
  for (double p : positive)
    for (double n : negative) score.auc += p < n ? 1 : p == n ? .5 : 0;
  score.auc /= double(positive.size()) * double(negative.size());
  score.same /= double(positive.size());
  score.different /= double(negative.size());
  score.accuracy = double(score.tp + score.tn) / double(pairs.size());
  return score;
}

struct MetricTasksScore {
  double                           accuracy = 0;
  std::vector<int>                 predicted;
  std::vector<std::vector<double>> scores;
};
inline MetricTasksScore ScoreMetricTasks(const DigitEmbeddingPool& pool, const std::vector<OneShotTask>& tasks) {
  MetricTasksScore result;
  int              correct = 0;
  for (const auto& task : tasks) {
    std::vector<double> scores;
    for (std::size_t row : task.support)
      scores.push_back(-EmbeddingDistance(pool.host, pool.index.at(task.query), pool.index.at(row)));
    const int prediction = BestOneShot(scores);
    correct += prediction == task.correct;
    result.predicted.push_back(prediction);
    result.scores.push_back(scores);
  }
  result.accuracy = double(correct) / double(tasks.size());
  return result;
}

/// retrieval.tsv, embeddings.tsv and an 8-task retrieval.png (query + 3 supports,
/// the predicted support framed green when correct, red otherwise).
inline void ExportMetricTasks(const nn::ImageDataset& d, const std::vector<OneShotTask>& tasks,
                              const MetricTasksScore& result, const DigitEmbeddingPool& pool,
                              const std::filesystem::path& dir) {
  const int          count = 8, ways = 3, cell = 60, width = (ways + 1) * cell;
  std::vector<float> rgb(std::size_t(count * cell) * width * 3, 0.f);
  const auto         labels = Labels(d);
  const auto         pixels = d.images.contiguous().to_vector<float>();
  std::ofstream      scores(dir / "retrieval.tsv"), embeddings(dir / "embeddings.tsv");
  scores << "task\tquery_digit\tsupport_digits\tpredicted_digit\tnegative_squared_distances\n";
  for (std::size_t i = 0; i < tasks.size(); ++i) {
    const auto& t = tasks[i];
    scores << i << '\t' << labels[t.query] << '\t';
    for (int c : t.classes) scores << c << ',';
    scores << '\t' << t.classes[result.predicted[i]] << '\t';
    for (double v : result.scores[i]) scores << v << ',';
    scores << '\n';
    if (int(i) >= count) continue;
    for (int col = 0; col <= ways; ++col) {
      const std::size_t row = col ? t.support[col - 1] : t.query;
      for (int y = 0; y < cell; ++y)
        for (int x = 0; x < cell; ++x)
          for (int c = 0; c < 3; ++c) {
            float value = 25;
            if (y >= 2 && y < 58 && x >= 2 && x < 58)
              value = float(std::lround(pixels[row * 784 + std::size_t(((y - 2) / 2) * 28 + (x - 2) / 2)] * 255.f));
            else if (col && col - 1 == result.predicted[i])
              value = c == (result.predicted[i] == t.correct ? 1 : 0) ? 255 : 30;
            rgb[(std::size_t(i * cell + y) * width + std::size_t(col * cell + x)) * 3 + std::size_t(c)] = value;
          }
    }
  }
  embeddings << "row\tdigit\t128_embedding_coordinates\n";
  for (std::size_t i = 0; i < pool.rows.size(); ++i) {
    embeddings << pool.rows[i] << '\t' << pool.labels[i] << '\t';
    for (int j = 0; j < 128; ++j) embeddings << pool.host.at(std::int64_t(i), j) << ',';
    embeddings << '\n';
  }
  nn::save_png(nn::from_vector(rgb, {count * cell, width, 3}), (dir / "retrieval.png").string(), 0, 255);
  scores.close();
  embeddings.close();
  check::Require(bool(scores) && bool(embeddings), "write metric artifacts");
}

/// MNIST from `data_dir` (nn::load_mnist).
inline std::pair<nn::ImageDataset, nn::ImageDataset> LoadDigits(const std::filesystem::path& data_dir) {
  auto train = nn::load_mnist(data_dir.string(), "train"), test = nn::load_mnist(data_dir.string(), "test");
  check::Require(train.images.size(0) == 60000 && test.images.size(0) == 10000, "official MNIST sizes");
  return {std::move(train), std::move(test)};
}

}  // namespace metric
