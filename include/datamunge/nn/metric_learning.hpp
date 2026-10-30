#pragma once
/// @file metric_learning.hpp
/// @brief Tensor triplet objectives,strict semi-hard mining and validation-only distance evaluation.
#include "datamunge/nn/metric_learning_data.hpp"
#include <array>
#include <cmath>
namespace datamunge::nn::models {
namespace nn = datamunge::nn;
/// @brief Name anchor,positive and negative batch-row indices.
///
/// Discrete index metadata;no arithmetic,fitting or mutation.
///
/// @par Example
/// @code
/// const dm::nn::models::TripletIndex indices{0, 1, 2};
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
using TripletIndex = std::array<int, 3>;
/// @brief Compute squared-L2 hinge values for requested batch triplets.
///
/// No mining or fitting. ReLU has zero derivative at an exact margin
/// boundary. Empty indices retain a zero-sized input graph;indices need not
/// have distinct roles or actual class labels,which are caller metadata.
///
/// @param embeddings Float32/64[B,D],D positive.
/// @param indices Valid anchor/positive/negative batch rows,empty allowed.
/// @param margin Finite positive squared-distance margin,default.2.
/// @return Same-dtype[K] max(0,dAP-dAN+margin),retaining repeated-row gradients.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto x = n::tensor({0., 1., 2.}, {3, 1}).requires_grad_();
/// const auto loss = n::models::TripletHinge(x, {{0, 1, 2}});
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
inline nn::Tensor TripletHinge(const nn::Tensor &embeddings,
                               const std::vector<TripletIndex> &indices, float margin = .2f) {
  if (embeddings.dim() != 2 || embeddings.shape()[1] < 1 ||
      (embeddings.dtype() != nn::DType::Float32 && embeddings.dtype() != nn::DType::Float64) ||
      !std::isfinite(margin) || margin <= 0)
    throw std::invalid_argument("invalid triplet matrix or margin");
  for (const auto &t : indices)
    for (const auto row : t)
      if (row < 0 || row >= embeddings.shape()[0])
        throw std::invalid_argument("triplet index out of bounds");
  std::vector<std::int64_t> a, p, n;
  for (const auto &t : indices)
    a.push_back(t[0]), p.push_back(t[1]), n.push_back(t[2]);
  const nn::Tensor ea = nn::gather(embeddings, nn::from_vector(a)),
                   ep = nn::gather(embeddings, nn::from_vector(p)),
                   en = nn::gather(embeddings, nn::from_vector(n));
  return nn::relu(nn::sum(nn::squared_difference(ea, ep), {1}) -
                  nn::sum(nn::squared_difference(ea, en), {1}) + double(margin));
}

/// @brief Average requested triplet hinges with a graph-connected empty zero.
///
/// All selected triplets,including inactive hinges,enter the denominator.
/// There is no active-only reweighting or hidden optimization.
///
/// @param embeddings Float32/64[B,D],D positive.
/// @param indices Valid batch-row triplets,empty allowed.
/// @param margin Finite positive squared-distance margin,default.2.
/// @return Same-type scalar mean hinge;empty selection has zero gradients.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// auto x = n::tensor({0., 1., 2.}, {3, 1}).requires_grad_();
/// const auto loss = n::models::MeanTriplet(x, {{0, 1, 2}});
/// const auto g = n::grad(loss, {x});
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
inline nn::Tensor MeanTriplet(const nn::Tensor &embeddings,
                              const std::vector<TripletIndex> &indices, float margin = .2f) {
  const auto values = TripletHinge(embeddings, indices, margin);
  return indices.empty() ? nn::sum(values) : nn::sum(values) / double(indices.size());
}
/// @brief Own selected triplet row metadata and positive-pair mining counts.
///
/// Triplets are anchor-major,then positive-major;each eligible ordered
/// positive pair yields one closest strict semi-hard negative. No learned state.
///
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto mining = n::models::MineSemiHard(n::tensor({0., 1., 1.5}, {3, 1}), {0, 0, 1}, 2.f);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
struct TripletMining {
  std::vector<TripletIndex> triplets; ///< Selected anchor/positive/negative row metadata.
  int positive_pairs = 0;             ///< Number of distinct-row ordered same-class pairs.
  int without_semihard = 0;           ///< Ordered positive pairs without an eligible negative.
};
/// @brief Compute detached all-pairs squared distances with Float64 arithmetic.
///
/// Float32 coordinates are promoted before subtraction,matching the source
/// host-double distance oracle. No matrix-product cancellation or fitting.
/// Time and temporary storage are O(B*B*D).
///
/// @param features Finite Float32/64[B,D],D positive,empty B allowed.
/// @return Float64[B,B] sum of squared coordinate differences,without input gradients.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto distances = n::models::MetricDistanceMatrix(n::tensor({0., 1., 2.}, {3, 1}));
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see MineSemiHard(), MetricPairDistance()
inline nn::Tensor MetricDistanceMatrix(const nn::Tensor &features) {
  nn::NoGradGuard guard;
  if (features.dim() != 2 || features.shape()[1] < 1 ||
      (features.dtype() != nn::DType::Float32 && features.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(features), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("finite floating embedding matrix required");
  const auto x = features.to(nn::DType::Float64);
  return nn::sum(nn::square(nn::expand_dims(x, 1) - nn::expand_dims(x, 0)), {2});
}
/// @brief Select the closest strictly semi-hard negative for each ordered positive pair.
///
/// Frozen Float64 tensor distances and masks require dAP<dAN<dAP+margin.
/// Negatives have a different class;ties select the lowest negative row. No
/// hard/easy fallback. Host iteration only serializes tensor-selected indices
/// and counters. O(B^3) mask storage makes this a bounded-batch teaching miner;
/// no parameter/input gradients or fitting occur here.
///
/// @param features Finite Float32/64[B,D],D positive.
/// @param labels B integer class IDs as discrete metadata.
/// @param margin Finite positive squared-distance margin,default.2.
/// @return Selected row triplets plus positive-pair and missing-negative counts.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto mining = n::models::MineSemiHard(n::tensor({0., 1., 1.5}, {3, 1}), {0, 0, 1}, 2.f);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
inline TripletMining MineSemiHard(const nn::Tensor &features, const std::vector<int> &labels,
                                  float margin = .2f) {
  nn::NoGradGuard guard;
  if (features.dim() != 2 || features.shape()[0] != std::int64_t(labels.size()) ||
      !std::isfinite(margin) || margin <= 0 ||
      labels.size() > std::size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("invalid mining labels or margin");
  const auto distances = MetricDistanceMatrix(features);
  if (labels.empty())
    return {};
  const auto batch = features.shape()[0];
  const auto y = nn::from_vector(std::vector<std::int64_t>(labels.begin(), labels.end()));
  const auto same = nn::equal(nn::expand_dims(y, 1), nn::expand_dims(y, 0));
  const auto row = nn::arange(0, batch, 1, nn::DType::Int64);
  const auto distinct = nn::not_equal(nn::expand_dims(row, 1), nn::expand_dims(row, 0));
  const auto positive = nn::logical_and(same, distinct);
  const auto ap = nn::expand_dims(distances, 2), an = nn::expand_dims(distances, 1);
  const auto eligible =
      nn::logical_and(nn::logical_and(nn::greater(an, ap), nn::less(an, ap + double(margin))),
                      nn::expand_dims(nn::logical_not(same), 1));
  const auto candidates = nn::where(
      eligible, an, nn::full({}, std::numeric_limits<double>::infinity(), nn::DType::Float64));
  const auto chosen = nn::argmin(candidates, 2);
  const auto valid = nn::logical_and(positive, nn::any(eligible, {2}));
  TripletMining result;
  result.positive_pairs = int(nn::count_nonzero(positive).item<std::int64_t>());
  result.without_semihard =
      result.positive_pairs - int(nn::count_nonzero(valid).item<std::int64_t>());
  const auto selected = chosen.to_vector<std::int64_t>();
  const auto included = valid.to_vector<bool>();
  for (std::int64_t a = 0; a < batch; ++a)
    for (std::int64_t p = 0; p < batch; ++p)
      if (included[std::size_t(a * batch + p)])
        result.triplets.push_back({int(a), int(p), int(selected[std::size_t(a * batch + p)])});
  return result;
}
/// @brief Compute detached aligned embedding-pair squared distances.
///
/// Float64 promotion occurs before subtraction. No unit-sphere assumption,
/// fitting or gradients;host vectors carry row metadata only.
///
/// @param features Finite Float32/64[B,D] embeddings,D positive.
/// @param left Valid batch-row indices.
/// @param right Same-length valid batch-row indices.
/// @return Float64[pairs] squared distances,empty selection allowed.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto distances =
///     n::models::MetricPairDistance(n::tensor({0., 1., 2.}, {3, 1}), {0, 1}, {1, 2});
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see MetricDistanceMatrix(), CalibrateMetricDistanceThreshold()
inline nn::Tensor MetricPairDistance(const nn::Tensor &features,
                                     const std::vector<std::int64_t> &left,
                                     const std::vector<std::int64_t> &right) {
  nn::NoGradGuard guard;
  if (features.dim() != 2 || features.shape()[1] < 1 || left.size() != right.size() ||
      (features.dtype() != nn::DType::Float32 && features.dtype() != nn::DType::Float64) ||
      !nn::all(nn::less(nn::abs(features), std::numeric_limits<double>::infinity()))
           .item<bool>())
    throw std::invalid_argument("invalid embedding pairs");
  for (const auto &rows : {left, right})
    for (const auto row : rows)
      if (row < 0 || row >= features.shape()[0])
        throw std::invalid_argument("embedding pair index out of bounds");
  const auto x = features.to(nn::DType::Float64);
  return nn::sum(
      nn::square(nn::gather(x, nn::from_vector(left)) - nn::gather(x, nn::from_vector(right))),
      {1});
}
/// @brief Sample distinct class-balanced batch rows.
///
/// Metadata-only P-by-K sampling without replacement within class. No pixel
/// arithmetic or numerical model fitting.
///
/// @param buckets Valid included digit-row buckets.
/// @param each Number of distinct rows per included class,at least two.
/// @param seed std::mt19937 index/shuffle seed.
/// @return Shuffled class-balanced source-row IDs.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// n::models::DigitBuckets b;
/// b[0] = {0, 1, 2};
/// b[1] = {3, 4, 5};
/// const auto rows = n::models::MetricBatchRows(b, 2, 11);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
inline std::vector<std::size_t> MetricBatchRows(const DigitBuckets &buckets, int each,
                                                int seed) {
  metric_data_detail::Validate(buckets);
  if (SiameseClasses(buckets).empty())
    throw std::invalid_argument("nonempty metric classes required");
  std::mt19937 rng(seed);
  std::vector<std::size_t> rows;
  for (int c : SiameseClasses(buckets)) {
    auto candidates = buckets[c];
    if (each < 2 || std::size_t(each) > candidates.size())
      throw std::invalid_argument("unique P-by-K batch size");
    std::shuffle(candidates.begin(), candidates.end(), rng);
    rows.insert(rows.end(), candidates.begin(), candidates.begin() + each);
  }
  std::shuffle(rows.begin(), rows.end(), rng);
  return rows;
}

/// @brief Sample fixed valid triplet row metadata.
///
/// No hard-negative criterion;this fixed metadata probe is independent of
/// training-time mining. Same-class positive differs from its anchor.
///
/// @param labels Batch/pool class ID metadata with positive and negative partners.
/// @param count Positive number of sampled triplets.
/// @param seed std::mt19937 index draw seed.
/// @return Random ordered valid anchor/positive/negative row triplets.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// const auto triplets = dm::nn::models::FixedMetricTriplets({0, 0, 1, 1}, 4, 7);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see TripletHinge(), MineSemiHard()
inline std::vector<TripletIndex> FixedMetricTriplets(const std::vector<int> &labels, int count,
                                                     int seed) {
  if (labels.size() < 3 || count < 1)
    throw std::invalid_argument(
        "positive fixed triplet count and at least three rows required");
  std::mt19937 rng(seed);
  std::vector<TripletIndex> out;
  for (int i = 0; i < count; ++i) {
    const int a = std::uniform_int_distribution<int>(0, int(labels.size()) - 1)(rng);
    std::vector<int> positives, negatives;
    for (std::size_t j = 0; j < labels.size(); ++j)
      if (int(j) != a)
        (labels[j] == labels[a] ? positives : negatives).push_back(int(j));
    if (positives.empty() || negatives.empty())
      throw std::invalid_argument("valid fixed triplet classes");
    const int p =
        positives[std::uniform_int_distribution<std::size_t>(0, positives.size() - 1)(rng)];
    const int n =
        negatives[std::uniform_int_distribution<std::size_t>(0, negatives.size() - 1)(rng)];
    out.push_back({a, p, n});
  }
  return out;
}

/// @brief Private shape/value checks shared by distance threshold helpers.
namespace metric_evaluation_detail {
/// @brief Require nonempty finite unit-sphere distances and aligned hard targets.
/// @param distances Float64[N] finite values in[0,4.00001],N positive.
/// @param labels Same-shaped Float64 hard0/1 same-class targets.
/// @throws std::invalid_argument For invalid distance/target tensors.
inline void Validate(const nn::Tensor &distances, const nn::Tensor &labels) {
  if (distances.dim() != 1 || distances.numel() < 1 ||
      distances.dtype() != nn::DType::Float64 || labels.dtype() != nn::DType::Float64 ||
      labels.shape() != distances.shape() ||
      !nn::all(nn::logical_and(nn::greater_equal(distances, 0.),
                               nn::less_equal(distances, 4.00001)))
           .item<bool>() ||
      !nn::all(nn::logical_or(nn::equal(labels, 0.), nn::equal(labels, 1.))).item<bool>())
    throw std::invalid_argument("invalid sphere distances or hard labels");
}
} // namespace metric_evaluation_detail
/// @brief Calibrate an accuracy-maximizing squared-distance threshold on validation pairs.
///
/// Candidates are-1,midpoints of distinct sorted distances,and4.00001.
/// Predict same when distance<=threshold;accuracy ties take the first candidate
/// in ascending order. O(N^2) comparisons;validation inputs only,no optimizer.
///
/// @param distances Float64[N] finite squared unit-sphere distances in[0,4.00001].
/// @param labels Float64[N] hard same-class0/1 validation targets.
/// @return Detached Float64 scalar selected threshold.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto threshold = n::models::CalibrateMetricDistanceThreshold(
///     n::from_vector(std::vector<double>{.1, .2, .8, .9}),
///     n::from_vector(std::vector<double>{1, 1, 0, 0}));
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see MetricDistanceMetrics(), MetricPairDistance()
inline nn::Tensor CalibrateMetricDistanceThreshold(const nn::Tensor &distances,
                                                   const nn::Tensor &labels) {
  nn::NoGradGuard guard;
  metric_evaluation_detail::Validate(distances, labels);
  const auto n = distances.numel();
  const auto sorted = nn::gather(distances, nn::argsort(distances));
  const auto lo = nn::slice_axis(sorted, 0, 0, n - 1), hi = nn::slice_axis(sorted, 0, 1, n);
  const auto candidates = nn::concat({nn::full({1}, -1., nn::DType::Float64), (lo + hi) * .5,
                                      nn::full({1}, 4.00001, nn::DType::Float64)},
                                     0);
  const auto valid = nn::concat(
      {nn::ones({1}, nn::DType::Bool), nn::not_equal(lo, hi), nn::ones({1}, nn::DType::Bool)},
      0);
  const auto predicted =
      nn::less_equal(nn::expand_dims(distances, 0), nn::expand_dims(candidates, 1));
  const auto correct = nn::mean(
      nn::equal(predicted, nn::expand_dims(nn::equal(labels, 1.), 0)).to(nn::DType::Float64),
      {1});
  return nn::gather(candidates, nn::argmax(nn::where(valid, correct, -1.), 0));
}
/// @brief Summarize thresholded verification and pairwise rank AUC with tensors.
///
/// AUC compares every same/different pair,credit1 for smaller same distance
/// and.5 at equality. O(N^2) memory/comparisons. Detached,no fitting or
/// threshold selection;threshold must come from a separate validation split.
///
/// @param distances Float64[N] finite unit-sphere squared distances in[0,4.00001].
/// @param labels Float64[N] hard same-class0/1 targets,both classes present.
/// @param threshold Finite calibrated distance cutoff;predict same at or below it.
/// @return Float64[8]:accuracy,mean same distance,mean different distance,AUC,TP,FP,FN,TN.
/// @throws std::invalid_argument For invalid rank,dtype,nonfinite embeddings,margin,index,label or score geometry.
/// @par Example
/// @code
/// namespace n = dm::nn;
/// const auto result =
///     n::models::MetricDistanceMetrics(n::from_vector(std::vector<double>{.1, .2, .8, .9}),
///                                      n::from_vector(std::vector<double>{1, 1, 0, 0}), .5);
/// @endcode
/// @par References
/// - Schroff,Kalenichenko and Philbin(2015),FaceNet:A Unified Embedding for Face Recognition and Clustering, https://arxiv.org/abs/1503.03832. Closest strict semi-hard negatives without fallback and validation threshold midpoints follow the source teaching adaptation.
/// @see CalibrateMetricDistanceThreshold(), MetricPairDistance()
inline nn::Tensor MetricDistanceMetrics(const nn::Tensor &distances, const nn::Tensor &labels,
                                        double threshold) {
  nn::NoGradGuard guard;
  metric_evaluation_detail::Validate(distances, labels);
  if (!std::isfinite(threshold))
    throw std::invalid_argument("finite metric threshold required");
  const auto positive = nn::sum(labels), negative = nn::sum(1 - labels);
  if (positive.item() == 0 || negative.item() == 0)
    throw std::invalid_argument("both verification classes required");
  const auto prediction = nn::less_equal(distances, threshold).to(nn::DType::Float64);
  const auto a = nn::expand_dims(distances, 1), b = nn::expand_dims(distances, 0);
  const auto rank =
      nn::less(a, b).to(nn::DType::Float64) + .5 * nn::equal(a, b).to(nn::DType::Float64);
  const auto auc = nn::sum(rank * nn::expand_dims(labels, 1) * nn::expand_dims(1 - labels, 0)) /
                   (positive * negative);
  return nn::stack(
      {nn::mean(nn::equal(prediction, labels).to(nn::DType::Float64)),
       nn::sum(distances * labels) / positive, nn::sum(distances * (1 - labels)) / negative,
       auc, nn::sum(prediction * labels), nn::sum(prediction * (1 - labels)),
       nn::sum((1 - prediction) * labels), nn::sum((1 - prediction) * (1 - labels))});
}
} // namespace datamunge::nn::models
