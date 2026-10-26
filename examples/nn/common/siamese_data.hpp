// Class-disjoint verification pairs and genuine one-support-per-class tasks:
// a port of tflow's common/ml/siamese_data.h (used by facenet_triplet,
// angular_margin and siamese).
//
// Datasets are nn::ImageDataset (MNIST from nn::load_mnist: float32 images
// [N, 28, 28, 1] = uint8 / 255, int64 labels). Buckets, pairs and tasks are
// drawn with std::mt19937 + std::uniform_int_distribution / std::shuffle
// exactly as tflow does, so with the same standard library (libc++ on macOS)
// the rows are identical to TF's.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "common/checks.hpp"

namespace metric {

namespace nn = datamunge::nn;

/// Integer labels of a dataset (int64 tensor -> host ints).
inline std::vector<int> Labels(const nn::ImageDataset& d) { return d.labels.to_vector<int>(); }

using DigitBuckets = std::array<std::vector<std::size_t>, 10>;

/// The first `limit` rows of each class in `classes` (scan order).
inline DigitBuckets SiameseBuckets(const nn::ImageDataset& d, const std::vector<int>& classes, std::size_t limit = 512) {
  const auto   labels = Labels(d);
  DigitBuckets b;
  for (int c : classes) {
    if (c < 0 || c >= 10) throw std::invalid_argument("digit class");
    for (std::size_t i = 0; i < labels.size() && b[c].size() < limit; ++i)
      if (labels[i] == c) b[c].push_back(i);
    check::Require(b[c].size() >= 2, "two distinct images per class required");
  }
  return b;
}

inline std::vector<int> SiameseClasses(const DigitBuckets& b) {
  std::vector<int> c;
  for (int i = 0; i < 10; ++i)
    if (!b[i].empty()) c.push_back(i);
  return c;
}

struct VerificationPair {
  std::size_t left = 0, right = 0;
  float       same = 0;
  bool operator==(const VerificationPair& p) const { return left == p.left && right == p.right && same == p.same; }
};

/// `count` balanced (alternating different/same) pairs, shuffled.
inline std::vector<VerificationPair> SiamesePairs(const DigitBuckets& b, int count, int seed) {
  const auto classes = SiameseClasses(b);
  if (classes.size() < 2 || count < 2 || count % 2)
    throw std::invalid_argument("balanced pairs need two classes and positive even count");
  std::mt19937 rng(seed);
  auto pick = [&](std::size_t n) { return std::uniform_int_distribution<std::size_t>(0, n - 1)(rng); };
  std::vector<VerificationPair> pairs;
  for (int i = 0; i < count; ++i) {
    const std::size_t ci = pick(classes.size()), cj = i % 2 ? ci : (ci + 1 + pick(classes.size() - 1)) % classes.size();
    const auto&       a = b[classes[ci]];
    const auto&       other = b[classes[cj]];
    const std::size_t ia    = pick(a.size());
    std::size_t       ib    = pick(other.size());
    if (ci == cj) ib = (ia + 1 + pick(a.size() - 1)) % a.size();
    pairs.push_back({a[ia], other[ib], float(ci == cj)});
  }
  std::shuffle(pairs.begin(), pairs.end(), rng);
  return pairs;
}

struct OneShotTask {
  std::size_t              query = 0;
  std::vector<std::size_t> support;
  std::vector<int>         classes;
  int                      correct = -1;
  bool operator==(const OneShotTask& t) const {
    return query == t.query && support == t.support && classes == t.classes && correct == t.correct;
  }
};

/// One support image per class; the query is another image of the correct class.
inline std::vector<OneShotTask> SiameseTasks(const DigitBuckets& b, int count, int seed) {
  const auto classes = SiameseClasses(b);
  if (classes.size() < 2 || count < 1) throw std::invalid_argument("one-shot task classes/count");
  std::mt19937 rng(seed);
  auto pick = [&](std::size_t n) { return std::uniform_int_distribution<std::size_t>(0, n - 1)(rng); };
  std::vector<OneShotTask> out;
  for (int i = 0; i < count; ++i) {
    OneShotTask task;
    task.classes = classes;
    std::shuffle(task.classes.begin(), task.classes.end(), rng);
    task.correct = int(pick(classes.size()));
    for (int c : task.classes) task.support.push_back(b[c][pick(b[c].size())]);
    const int   c       = task.classes[task.correct];
    const auto& rows    = b[c];
    const auto  support = std::find(rows.begin(), rows.end(), task.support[task.correct]) - rows.begin();
    task.query          = rows[(support + 1 + pick(rows.size() - 1)) % rows.size()];
    out.push_back(task);
  }
  return out;
}

/// Images of `rows` as float32 [rows, 28, 28, 1] in [0, 1].
inline nn::Tensor SiameseImages(const nn::ImageDataset& d, const std::vector<std::size_t>& rows) {
  check::Require(d.images.dim() == 4 && d.images.size(1) == 28 && d.images.size(2) == 28, "MNIST image dimensions");
  std::vector<std::int64_t> ids;
  for (auto r : rows) {
    check::Require(std::int64_t(r) < d.images.size(0), "pair image row bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::take_rows(d.images, ids);
}

struct SiameseBatch {
  nn::Tensor left, right, labels;
};
inline SiameseBatch MakeSiameseBatch(const nn::ImageDataset& d, const std::vector<VerificationPair>& pairs) {
  std::vector<std::size_t> left, right;
  std::vector<float>       y;
  for (const auto& p : pairs) {
    left.push_back(p.left);
    right.push_back(p.right);
    y.push_back(p.same);
  }
  return {SiameseImages(d, left), SiameseImages(d, right), nn::from_vector(y)};
}

inline int BestOneShot(const std::vector<double>& scores) {
  check::Require(!scores.empty(), "nonempty one-shot candidates");
  for (double s : scores) check::Require(std::isfinite(s), "finite one-shot similarity");
  return int(std::max_element(scores.begin(), scores.end()) - scores.begin());
}

}  // namespace metric
