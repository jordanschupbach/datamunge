#pragma once
/// @file metric_learning_data.hpp
/// @brief Class-disjoint pair/task metadata,tensor verification metrics,pixel baselines and one-shot galleries.
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/datasets.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/siamese_models.hpp"
#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <set>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Read integer dataset class IDs as index-sampling metadata.
///
/// No pixel arithmetic or fitting. This host conversion serves only discrete
/// row/class sampling metadata.
///
/// @param data Image dataset with one Int64 label per image.
/// @return Owning vector<int> class IDs,each representable as signed int.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto labels = n::models::MetricDatasetLabels(data);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiameseBuckets()
inline std::vector<int> MetricDatasetLabels(const nn::ImageDataset &data) {
  if (data.images.dim() != 4 || data.labels.shape() != nn::Shape{data.images.shape()[0]} ||
      data.labels.dtype() != nn::DType::Int64 ||
      !nn::all(nn::logical_and(
                   nn::greater_equal(data.labels, 0.),
                   nn::less_equal(data.labels, double(std::numeric_limits<int>::max()))))
           .item<bool>())
    throw std::invalid_argument("invalid metric dataset labels");
  return data.labels.to_vector<int>();
}
/// @brief Group source-row indices for each of ten digit classes.
///
/// Index metadata only,not host model/data numerics. Empty buckets mark
/// excluded classes;nonempty sampled buckets require distinct image rows.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto buckets = n::models::SiameseBuckets(data, {0, 1});
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiameseBuckets()
using DigitBuckets = std::array<std::vector<std::size_t>, 10>;
/// @brief Collect the first requested number of source rows per selected digit class.
///
/// Host scans select only row/class metadata. Pixel values are neither read
/// nor changed;no fitting or RNG.
///
/// @param d MNIST-like dataset with aligned Int64 class labels.
/// @param classes Unique selected digit classes0..9.
/// @param limit Maximum rows per class,default512;at least two must be available.
/// @return Owning buckets retaining original scan order.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto buckets = n::models::SiameseBuckets(data, {0, 1}, 3);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiamesePairs(), SiameseTasks()
inline DigitBuckets SiameseBuckets(const nn::ImageDataset &d, const std::vector<int> &classes,
                                   std::size_t limit = 512) {
  const auto labels = MetricDatasetLabels(d);
  DigitBuckets b;
  std::set<int> seen;
  for (int c : classes) {
    if (!seen.insert(c).second)
      throw std::invalid_argument("duplicate metric class");
    if (c < 0 || c >= 10)
      throw std::invalid_argument("digit class");
    for (std::size_t i = 0; i < labels.size() && b[c].size() < limit; ++i)
      if (labels[i] == c)
        b[c].push_back(i);
    if (b[c].size() < 2)
      throw std::invalid_argument("two distinct images per class required");
  }
  return b;
}

/// @brief List included digit classes in ascending category order.
///
/// Pure index metadata;no fitting,mutation or RNG.
///
/// @param b Ten source-row buckets.
/// @return Class IDs whose buckets are nonempty.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto classes = n::models::SiameseClasses(n::models::SiameseBuckets(data, {0, 1}));
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiameseBuckets()
inline std::vector<int> SiameseClasses(const DigitBuckets &b) {
  std::vector<int> c;
  for (int i = 0; i < 10; ++i)
    if (!b[i].empty())
      c.push_back(i);
  return c;
}

/// @brief Describe two dataset rows and their soft same-class training target.
///
/// Row IDs are metadata;the balanced sampler emits hard0/1 targets.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto pair = n::models::SiamesePairs(n::models::SiameseBuckets(data, {0, 1}), 4, 11)[0];
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see MakeSiameseBatch()
struct VerificationPair {
  std::size_t left = 0;  ///< Left image row ID.
  std::size_t right = 0; ///< Right image row ID,distinct for sampled pairs.
  float same = 0;        ///< Same-class target in[0,1],hard binary for generated pairs.

  /// @brief Compare pair row IDs and target exactly.
  ///
  /// No numeric inference or mutation.
  ///
  /// @param p Other pair metadata.
  /// @return True when both rows and target match.
  /// @par Example
  /// @code
  /// const dm::nn::models::VerificationPair pair{0, 1, 0};
  /// const bool same = pair == pair;
  /// @endcode
  /// @par References
  /// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
  /// @see VerificationPair
  bool operator==(const VerificationPair &p) const {
    return left == p.left && right == p.right && same == p.same;
  }
};
/// @brief Internal validation of discrete per-class row sampling metadata.
namespace metric_data_detail {
/// @brief Require every included class to have at least two distinct source rows.
/// @param buckets Ten source-row index lists;empty lists are excluded.
/// @throws std::invalid_argument For singleton or duplicate row lists.
inline void Validate(const DigitBuckets &buckets) {
  for (const auto &rows : buckets) {
    if (rows.empty())
      continue;
    std::set<std::size_t> unique(rows.begin(), rows.end());
    if (rows.size() < 2 || unique.size() != rows.size())
      throw std::invalid_argument("metric buckets require distinct source rows");
  }
}
} // namespace metric_data_detail
/// @brief Sample exactly balanced nonidentity same/different pairs and shuffle their order.
///
/// Retains source discrete index draw order under the same C++ standard
/// library. Sampling only chooses metadata;images and model numerics remain
/// tensors. Same-class rows always differ. No model fitting or source mutation.
///
/// @param b Valid buckets with at least two included digit classes.
/// @param count Positive even pair count,at least two.
/// @param seed Seed for std::mt19937 row-index draws and shuffle.
/// @return Owning shuffled pairs,half same-class and half different-class.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto pairs = n::models::SiamesePairs(n::models::SiameseBuckets(data, {0, 1}), 8, 11);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiamesePairs(), SiameseTasks()
inline std::vector<VerificationPair> SiamesePairs(const DigitBuckets &b, int count, int seed) {
  metric_data_detail::Validate(b);
  const auto classes = SiameseClasses(b);
  if (classes.size() < 2 || count < 2 || count % 2)
    throw std::invalid_argument("balanced pairs need two classes and positive even count");
  std::mt19937 rng(seed);
  /// @brief Draw one valid metadata index from the seeded source stream.
  /// @param n Positive candidate count.
  /// @return Uniform index in[0,n).
  auto pick = [&](std::size_t n) {
    return std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
  };
  std::vector<VerificationPair> pairs;
  for (int i = 0; i < count; ++i) {
    const std::size_t ci = pick(classes.size()),
                      cj = i % 2 ? ci : (ci + 1 + pick(classes.size() - 1)) % classes.size();
    const auto &a = b[classes[ci]];
    const auto &other = b[classes[cj]];
    const std::size_t ia = pick(a.size());
    std::size_t ib = pick(other.size());
    if (ci == cj)
      ib = (ia + 1 + pick(a.size() - 1)) % a.size();
    pairs.push_back({a[ia], other[ib], float(ci == cj)});
  }
  std::shuffle(pairs.begin(), pairs.end(), rng);
  return pairs;
}

/// @brief Describe a query and one support image per included class.
///
/// The query is a different image of the class at correct. Supports and class
/// IDs are aligned in shuffled order;no fine-tuning is implied.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto task = n::models::SiameseTasks(n::models::SiameseBuckets(data, {0, 1}), 1, 17)[0];
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiamesePairs(), SiameseTasks()
struct OneShotTask {
  std::size_t query = 0;            ///< Query image row,distinct from every support image.
  std::vector<std::size_t> support; ///< One support row per class,in episode order.
  std::vector<int> classes;         ///< Corresponding unique digit category IDs.
  int correct = -1;                 ///< Correct support-column index.

  /// @brief Compare all query/support/class/correct metadata exactly.
  ///
  /// No fitting or mutation.
  ///
  /// @param t Other one-shot task.
  /// @return True if every episode field matches.
  /// @par Example
  /// @code
  /// const dm::nn::models::OneShotTask task;
  /// const bool same = task == task;
  /// @endcode
  /// @par References
  /// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
  /// @see OneShotTask
  bool operator==(const OneShotTask &t) const {
    return query == t.query && support == t.support && classes == t.classes &&
           correct == t.correct;
  }
};
/// @brief Sample genuine one-support-per-class episodes with distinct same-class queries.
///
/// Metadata sampling preserves source draw order within the same standard
/// library. No pixel/model arithmetic,fitting or source mutation.
///
/// @param b Valid buckets with at least two included classes.
/// @param count Positive episode count.
/// @param seed Seed for source std::mt19937 index draws/class shuffle.
/// @return Owning episodes with aligned shuffled classes/supports and known correct column.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto tasks = n::models::SiameseTasks(n::models::SiameseBuckets(data, {0, 1}), 3, 17);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiamesePairs(), SiameseTasks()
inline std::vector<OneShotTask> SiameseTasks(const DigitBuckets &b, int count, int seed) {
  metric_data_detail::Validate(b);
  const auto classes = SiameseClasses(b);
  if (classes.size() < 2 || count < 1)
    throw std::invalid_argument("one-shot task classes/count");
  std::mt19937 rng(seed);
  /// @brief Draw one valid metadata index from the seeded source stream.
  /// @param n Positive candidate count.
  /// @return Uniform index in[0,n).
  auto pick = [&](std::size_t n) {
    return std::uniform_int_distribution<std::size_t>(0, n - 1)(rng);
  };
  std::vector<OneShotTask> out;
  for (int i = 0; i < count; ++i) {
    OneShotTask task;
    task.classes = classes;
    std::shuffle(task.classes.begin(), task.classes.end(), rng);
    task.correct = int(pick(classes.size()));
    for (int c : task.classes)
      task.support.push_back(b[c][pick(b[c].size())]);
    const int c = task.classes[task.correct];
    const auto &rows = b[c];
    const auto support =
        std::find(rows.begin(), rows.end(), task.support[task.correct]) - rows.begin();
    task.query = rows[(support + 1 + pick(rows.size() - 1)) % rows.size()];
    out.push_back(task);
  }
  return out;
}

/// @brief Gather requested MNIST image rows without host pixel copies or resampling.
///
/// Only row IDs become a tensor. Source pixels are not changed;no fitting.
///
/// @param d Floating[N,28,28,1] image dataset.
/// @param rows Valid source row IDs,duplicates/empty allowed.
/// @return Floating[rows.size(),28,28,1] gathered images with source gradients if requested.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto images = n::models::SiameseImages(data, {2, 0});
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see MakeSiameseBatch()
inline nn::Tensor SiameseImages(const nn::ImageDataset &d,
                                const std::vector<std::size_t> &rows) {
  if (d.images.dim() != 4 || d.images.shape()[1] != 28 || d.images.shape()[2] != 28 ||
      d.images.shape()[3] != 1 ||
      (d.images.dtype() != nn::DType::Float32 && d.images.dtype() != nn::DType::Float64))
    throw std::invalid_argument("MNIST-like images required");
  std::vector<std::int64_t> ids;
  for (const auto r : rows) {
    if (r >= std::size_t(d.images.shape()[0]))
      throw std::invalid_argument("metric source row out of bounds");
    ids.push_back(std::int64_t(r));
  }
  return nn::gather(d.images, nn::from_vector(ids));
}
/// @brief Own aligned left/right image tensors and same-class training targets.
///
/// Gathered image tensors retain input graphs;labels are detached ingress
/// constants. No learned state or optimizer.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto batch = n::models::MakeSiameseBatch(data, {{0, 1, 1}, {0, 3, 0}});
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see MakeSiameseBatch()
struct SiameseBatch {
  nn::Tensor left;   ///< Floating[B,28,28,1] left images.
  nn::Tensor right;  ///< Same-shaped right images.
  nn::Tensor labels; ///< Same-dtype[B] soft same-class targets.
};
/// @brief Gather both image rows and labels for a requested pair order.
///
/// No fitting or source mutation. Host containers carry only row/label ingress
/// metadata;pixel gathering uses tensors.
///
/// @param d Valid floating MNIST-like image dataset.
/// @param pairs Valid row metadata and finite targets in[0,1],empty allowed.
/// @return Aligned image batches and same-image-dtype soft labels.
/// @throws std::invalid_argument For invalid dataset,geometry,dtypes,rows,classes,labels,scores or sampling counts.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto batch = n::models::MakeSiameseBatch(data, {{0, 1, 1}, {0, 3, 0}});
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. Class-disjoint MNIST pair/task seeds and byte-pixel baseline follow the source teaching adaptation.
/// @see SiameseImages(), SiameseBinaryCrossEntropy()
inline SiameseBatch MakeSiameseBatch(const nn::ImageDataset &d,
                                     const std::vector<VerificationPair> &pairs) {
  std::vector<std::size_t> left, right;
  std::vector<float> y;
  for (const auto &p : pairs) {
    if (!std::isfinite(p.same) || p.same < 0 || p.same > 1)
      throw std::invalid_argument("invalid same-class target");
    left.push_back(p.left);
    right.push_back(p.right);
    y.push_back(p.same);
  }
  return {SiameseImages(d, left), SiameseImages(d, right),
          nn::from_vector(y).to(d.images.dtype())};
}

/// @brief Summarize binary verification logits using detached tensor reductions.
///
/// Predictions use logit>=0. Missing-class probability means are zero. Counts
/// are exact for ordinary batch sizes. No fitting,mutation or gradient graph.
///
/// @param logits Finite floating[B] logits,B positive.
/// @param labels Same-shaped,same-dtype hard0/1 targets.
/// @return Float64[8]:mean BCE,accuracy,positive probability,negative probability,TP,FP,FN,TN.
/// @throws std::invalid_argument For incompatible shapes,dtypes,nonfinite scores,invalid targets or episode metadata.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto metrics = n::models::SiameseVerificationMetrics(
///     n::from_vector(std::vector<float>{2, -2}), n::from_vector(std::vector<float>{1, 0}));
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. MNIST adaptation;not the paper Omniglot benchmark.
/// @see SiameseTasks(), SiameseBinaryCrossEntropy()
inline nn::Tensor SiameseVerificationMetrics(const nn::Tensor &logits,
                                             const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  if (logits.dim() != 1 || logits.numel() < 1 || labels.shape() != logits.shape() ||
      labels.dtype() != logits.dtype() ||
      (logits.dtype() != nn::DType::Float32 && logits.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(logits), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::logical_or(nn::equal(labels, 0.), nn::equal(labels, 1.))).item<bool>())
    throw std::invalid_argument("invalid verification logits or binary labels");
  const auto y = labels.to(nn::DType::Float64);
  const auto z = logits.to(nn::DType::Float64);
  const auto prediction = nn::greater_equal(z, 0.).to(nn::DType::Float64);
  const auto probability = nn::sigmoid(z);
  const auto positive = nn::sum(y), negative = nn::sum(1 - y);
  return nn::stack(
      {nn::mean(SiameseBinaryCrossEntropy(z, y)),
       nn::mean(nn::equal(prediction, y).to(nn::DType::Float64)),
       nn::sum(y * probability) / nn::maximum(positive, nn::ones_like(positive)),
       nn::sum((1 - y) * probability) / nn::maximum(negative, nn::ones_like(negative)),
       nn::sum(prediction * y), nn::sum(prediction * (1 - y)), nn::sum((1 - prediction) * y),
       nn::sum((1 - prediction) * (1 - y))});
}
/// @brief Measure first-maximum one-shot classification accuracy.
///
/// Ties select the first support column. No fitting or query fine-tuning.
///
/// @param scores Finite floating[episodes,ways],episodes positive,ways at least two.
/// @param correct Int64[episodes] valid correct support columns.
/// @return Detached Float64 scalar fraction of correct episodes.
/// @throws std::invalid_argument For incompatible shapes,dtypes,nonfinite scores,invalid targets or episode metadata.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto accuracy =
///     n::models::SiameseOneShotAccuracy(n::from_vector(std::vector<float>{1, 0, 0, 1}, {2, 2}),
///                                       n::from_vector(std::vector<std::int64_t>{0, 1}));
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. MNIST adaptation;not the paper Omniglot benchmark.
/// @see SiameseTasks(), SiameseBinaryCrossEntropy()
inline nn::Tensor SiameseOneShotAccuracy(const nn::Tensor &scores, const nn::Tensor &correct) {
  nn::NoGradGuard guard;
  if (scores.dim() != 2 || scores.shape()[0] < 1 || scores.shape()[1] < 2 ||
      (scores.dtype() != nn::DType::Float32 && scores.dtype() != nn::DType::Float64) ||
      correct.dtype() != nn::DType::Int64 || correct.shape() != nn::Shape{scores.shape()[0]} ||
      !nn::all(nn::less(nn::abs(scores), std::numeric_limits<double>::infinity()))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(correct, 0.),
                               nn::less(correct, double(scores.shape()[1]))))
           .item<bool>())
    throw std::invalid_argument("invalid one-shot scores or correct columns");
  return nn::mean(nn::equal(nn::argmax(scores, 1), correct).to(nn::DType::Float64));
}
/// @brief Score episode supports by negative squared byte-pixel distance.
///
/// Images are rounded to bytes as floor(255*x+.5),matching positive lround
/// from the source baseline. Pixel arithmetic,gather and reductions use
/// tensors;host loops carry episode row metadata only. No learned fitting.
///
/// @param data Floating[N,28,28,1] finite image values in[0,1].
/// @param tasks Nonempty episodes of a common ways>=2,valid query/support rows.
/// @return Detached Float64[episodes,ways] negative SSD scores;larger is closer.
/// @throws std::invalid_argument For incompatible shapes,dtypes,nonfinite scores,invalid targets or episode metadata.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto tasks = n::models::SiameseTasks(n::models::SiameseBuckets(data, {0, 1}), 2, 17);
/// const auto scores = n::models::SiamesePixelScores(data, tasks);
/// @endcode
/// @par References
/// - Koch,Zemel and Salakhutdinov(2015),Siamese Neural Networks for One-shot Image Recognition, https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. MNIST adaptation;not the paper Omniglot benchmark.
/// @see SiameseTasks(), SiameseBinaryCrossEntropy()
inline nn::Tensor SiamesePixelScores(const nn::ImageDataset &data,
                                     const std::vector<OneShotTask> &tasks) {
  nn::NoGradGuard guard;
  if (tasks.empty() || tasks.front().support.size() < 2)
    throw std::invalid_argument("nonempty multiway episodes required");
  const auto ways = tasks.front().support.size();
  std::vector<std::size_t> query, support;
  for (const auto &task : tasks) {
    if (task.support.size() != ways)
      throw std::invalid_argument("episode ways mismatch");
    for (const auto row : task.support) {
      query.push_back(task.query);
      support.push_back(row);
    }
  }
  const auto a = SiameseImages(data, query).to(nn::DType::Float64),
             b = SiameseImages(data, support).to(nn::DType::Float64);
  if (!nn::all(nn::logical_and(nn::greater_equal(a, 0.), nn::less_equal(a, 1.)))
           .item<bool>() ||
      !nn::all(nn::logical_and(nn::greater_equal(b, 0.), nn::less_equal(b, 1.))).item<bool>())
    throw std::invalid_argument("finite normalized pixels required");
  return (-nn::sum(nn::square(nn::floor(a * 255. + .5) - nn::floor(b * 255. + .5)), {1, 2, 3}))
      .reshape({std::int64_t(tasks.size()), std::int64_t(ways)});
}

/// @brief Render query/support episodes with tensor image tiles and selected-column borders.
///
/// Each row is query then supports in episode order. A two-pixel border is
/// gray except the selected support:green if correct,red otherwise. Pixels are
/// nearest-upsampled2x. No host pixel arithmetic,fitting or gradient graph.
///
/// @param data Floating[N,28,28,1] normalized image dataset.
/// @param tasks Nonempty common-way episodes with valid correct indices.
/// @param scores Finite floating[episodes,ways] model scores.
/// @param count Positive maximum number of displayed episodes,default eight.
/// @return Detached Float32[min(count,episodes),60,60*(ways+1),3] RGB[0,1].
/// @throws std::invalid_argument For invalid scores,rows,episode metadata,count or image geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::ImageDataset data{n::zeros({6, 28, 28, 1}),
///                      n::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
/// const auto tasks = n::models::SiameseTasks(n::models::SiameseBuckets(data, {0, 1}), 2, 17);
/// const auto gallery = n::models::SiameseEpisodeGallery(data, tasks, n::zeros({2, 2}));
/// @endcode
/// @par References
/// - Koch et al.(2015), https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf. MNIST teaching gallery convention.
/// @see SiameseOneShotAccuracy(), SiameseImages()
inline nn::Tensor SiameseEpisodeGallery(const nn::ImageDataset &data,
                                        const std::vector<OneShotTask> &tasks,
                                        const nn::Tensor &scores, int count = 8) {
  nn::NoGradGuard guard;
  if (count < 1 || tasks.empty())
    throw std::invalid_argument("positive episode gallery count required");
  std::vector<std::int64_t> correct;
  for (const auto &task : tasks)
    correct.push_back(task.correct);
  SiameseOneShotAccuracy(scores, nn::from_vector(correct));
  if (scores.shape()[0] != std::int64_t(tasks.size()))
    throw std::invalid_argument("episode score count mismatch");
  const auto predicted = nn::argmax(scores, 1).to_vector<std::int64_t>();
  const auto ways = scores.shape()[1];
  std::vector<nn::Tensor> rows;
  for (std::size_t i = 0; i < std::min(tasks.size(), std::size_t(count)); ++i) {
    if (tasks[i].support.size() != std::size_t(ways))
      throw std::invalid_argument("episode ways mismatch");
    std::vector<std::size_t> ids{tasks[i].query};
    ids.insert(ids.end(), tasks[i].support.begin(), tasks[i].support.end());
    const auto images =
        nn::resize_nearest(SiameseImages(data, ids), {56, 56}).to(nn::DType::Float32);
    std::vector<nn::Tensor> tiles;
    for (std::int64_t col = 0; col <= ways; ++col) {
      const auto image = nn::slice_axis(images, 0, col, col + 1);
      const auto padded = nn::pad(image, {{0, 0}, {2, 2}, {2, 2}, {0, 0}});
      const auto mask =
          nn::greater(nn::pad(nn::ones_like(image), {{0, 0}, {2, 2}, {2, 2}, {0, 0}}), 0.);
      auto colour = nn::full({1, 1, 1, 3}, 25. / 255.);
      if (col > 0 && col - 1 == predicted[i])
        colour = nn::from_vector(tasks[i].correct == predicted[i]
                                     ? std::vector<float>{30.f / 255, 1, 30.f / 255}
                                     : std::vector<float>{1, 30.f / 255, 30.f / 255},
                                 {1, 1, 1, 3});
      tiles.push_back(nn::where(mask, padded * nn::ones({1, 1, 1, 3}), colour));
    }
    rows.push_back(nn::concat(tiles, 2));
  }
  return nn::concat(rows, 0);
}

} // namespace datamunge::nn::models
