#pragma once
/// @file latent_visualization.hpp
/// @brief Frozen thin-SVD tensor PCA and deterministic fitting-neighbor diagnostics.
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/linalg.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn::models {
/// @brief Internal finite tensor and nonnegative integer-index checks for projection studies.
namespace projection_detail {
/// @brief Require finite logical floating CPU values without changing gradient flags.
/// @param value Defined Float32/64 CPU tensor, with positive axes and rank at least one.
/// @throws std::invalid_argument For invalid shape, dtype or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
inline void Floating(const Tensor& value) {
  if (!value.defined() || value.dim() < 1 ||
      (value.dtype() != DType::Float32 && value.dtype() != DType::Float64))
    throw std::invalid_argument("projection studies require finite floating sample tensors");
  for (auto size : value.shape())
    if (size <= 0)
      throw std::invalid_argument("projection sample axes must be positive");
  for (double x : value.contiguous().to_vector<double>())
    if (!std::isfinite(x))
      throw std::invalid_argument("projection studies require finite floating values");
}
/// @brief Read validated nonnegative integer CPU values of a required rank.
/// @param value Int32/64 CPU tensor with positive axes and nonnegative entries.
/// @param rank Required rank, one for labels or two for neighbor tables.
/// @return Owning logical row-order Int64 values.
/// @throws std::invalid_argument For invalid shape/dtype or negative entries.
/// @throws std::logic_error For non-CPU storage.
inline std::vector<std::int64_t> Indices(const Tensor& value, int rank) {
  if (!value.defined() || value.dim() != rank ||
      (value.dtype() != DType::Int32 && value.dtype() != DType::Int64))
    throw std::invalid_argument("projection diagnostics require integer labels or neighbor tables");
  for (auto size : value.shape())
    if (size <= 0)
      throw std::invalid_argument("integer diagnostic axes must be positive");
  auto ids = value.contiguous().to_vector<std::int64_t>();
  for (auto id : ids)
    if (id < 0)
      throw std::invalid_argument("diagnostic indices and labels must be nonnegative");
  return ids;
}
} // namespace projection_detail

/// @brief Copy sample-axis-first floating values into live Float64 row coordinates.
///
/// Finite signed Float32/64 CPU input[N,...], rank>=1 and positive axes, becomes
/// an independently owned contiguous Float64[N,D] matrix, D=product(trailing
/// axes); a vector gives D=1. Logical row order and input gradients survive
/// strides, reshaping, widening and copying. No fitting, centering, clipping,
/// labels or RNG is implicit. Input unchanged; synchronize concurrent mutation.
/// Work/storage O(N*D), including host finite-value validation.
/// @param input Finite signed nonempty floating sample tensor, first axis N.
/// @return Independent contiguous Float64[N,D] rows, with applicable input paths.
/// @throws std::invalid_argument For invalid shape, dtype or nonfinite values.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// const auto rows=dm::nn::models::DoubleSampleRows(dm::nn::ones({3,2,2,1}));
/// // rows is independently owned Float64[3,4], with unchanged unit values.
/// @endcode
/// @see SvdPca, NearestFitRows()
inline Tensor DoubleSampleRows(const Tensor& input) {
  projection_detail::Floating(input);
  return cast(reshape(input, {input.size(0), input.numel() / input.size(0)}), DType::Float64)
      .clone();
}

/// @brief Fit frozen centered thin-SVD PCA to explicit tensor samples, without scaling.
///
/// Like R prcomp(center=TRUE,scale.=FALSE), widens samples to Float64, subtracts
/// only the fitting mean, takes the core thin SVD and retains requested columns
/// of V. Variances are singular-value squares/(N-1); scores=(rows-mean)*V.
/// No variance scaling, whitening, labels, null dropping, validation selection
/// or optimizer is hidden. Fitting is detached; private state owns lifetime
/// and getters export independent detached copies. Transform/Reconstruct keep
/// query/projection gradients while fitting mean/axes remain frozen. Zero
/// singular values are allowed; retained count is not numerical rank. Signs
/// follow nn::svd and repeated/zero singular values allow arbitrary bases.
/// Separate builds/backends can change bases within rounding or degenerate
/// subspaces. Finite CPU inputs accept strides and signed values. Constructor
/// work O(N*D*min(N,D)) and storage O(N*D+min(N,D)*(N+D)); retained state adds
/// O(N*R+D*R). Const methods have no shared mutable caches; callers synchronize
/// mutation of external tensors. Computed nonfinite outputs are rejected.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const n::models::SvdPca fit(n::tensor({-1.,0.,0.,0.,1.,0.},{3,2}),1);
/// const auto score=fit.Transform(n::tensor({2.,1.},{1,2}));
/// const auto reconstruction=fit.Reconstruct(score);
/// // reconstruction is approximately [2,0]; the query did not refit PCA.
/// @endcode
/// @par References
/// - R stats prcomp, https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html .
///   Follows centered unscaled SVD and N-1 variance semantics, with explicit
///   retained count and finite tensor inputs; this is not the DataFrame covariance solver.
/// @see DoubleSampleRows(), Transform(), Reconstruct(), svd()
class SvdPca {
  Tensor mean_;       ///< Private detached Float64[1,D] fitting mean.
  Tensor components_; ///< Private detached Float64[D,R] view owning its SVD storage.
  Tensor variances_;  ///< Private detached Float64[R] sample variances, zero allowed.
  Tensor scores_;     ///< Private detached Float64[N,R] fitting scores.
 public:
  /// @brief Fit a centered unscaled SVD using the supplied fitting rows alone.
  /// @param fitting Finite Float32/64 CPU samples[N,...], N>=2, positive axes.
  /// @param retained Number of axes R in[1,min(N,D)], default2.
  /// @throws std::invalid_argument For invalid samples/count or computed nonfinite values.
  /// @throws std::logic_error For non-CPU storage.
  /// @throws std::runtime_error If the core SVD solver cannot converge.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// const dm::nn::models::SvdPca fit(dm::nn::ones({4,3}),2);
  /// // Constant fitting rows are allowed; retained sample variances are zero.
  /// @endcode
  /// @see SvdPca, Transform()
  SvdPca(const Tensor& fitting, std::int64_t retained = 2) {
    NoGradGuard guard;
    const auto  x = DoubleSampleRows(fitting);
    if (x.size(0) < 2 || retained <= 0 || retained > std::min(x.size(0), x.size(1)))
      throw std::invalid_argument("SVD PCA needs at least two rows and a valid retained count");
    mean_ = nn::mean(x, {0}, true);
    projection_detail::Floating(mean_);
    const auto centered = x - mean_;
    projection_detail::Floating(centered);
    const auto factor = svd(centered);
    components_       = slice(factor.v, {0, 0}, {x.size(1), retained});
    variances_        = square(slice(factor.s, {0}, {retained})) / double(x.size(0) - 1);
    projection_detail::Floating(components_);
    projection_detail::Floating(variances_);
    scores_ = Transform(x);
  }
  /// @brief Export independent detached fitting means.
  /// @return Owning Float64[1,D] copy; modifications leave this model unchanged.
  Tensor mean() const { return mean_.clone(); }
  /// @brief Export independent detached principal-axis columns.
  /// @return Owning contiguous Float64[D,R] copy, with the fitted sign convention.
  Tensor components() const { return components_.clone(); }
  /// @brief Export independent detached sample variances.
  /// @return Owning Float64[R] singular-value squares/(N-1), zero values allowed.
  Tensor variances() const { return variances_.clone(); }
  /// @brief Export independent detached fitting projections.
  /// @return Owning Float64[N,R] fitting scores.
  Tensor scores() const { return scores_.clone(); }

  /// @brief Project query rows with the frozen fitting mean and retained axes.
  ///
  /// Computes (DoubleSampleRows(input)-mean)*components. No refit, per-query
  /// centering, labels, scaling or clipping. Accepts positive sample axes and
  /// strides, with the same flattened D as fitting. The owning Float64[M,R]
  /// result keeps input gradients; fitting state remains stopped. Work
  /// O(M*D*R), activation/storage O(M*(D+R)); inputs and fitted state unchanged.
  /// @param input Finite signed Float32/64 CPU samples[M,...], M>0, flattened width D.
  /// @return Owning live Float64[M,R] scores, retaining input paths only.
  /// @throws std::invalid_argument For invalid samples/width or computed nonfinite scores.
  /// @throws std::logic_error For non-CPU storage.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// const n::models::SvdPca fit(n::tensor({-1.,0.,0.,0.,1.,0.},{3,2}),1);
  /// const auto query=fit.Transform(n::tensor({4.,3.},{1,2}));
  /// // query has one retained coordinate, using the original fitting origin.
  /// @endcode
  /// @see SvdPca, Reconstruct()
  Tensor Transform(const Tensor& input) const {
    const auto x = DoubleSampleRows(input);
    if (x.size(1) != mean_.size(1))
      throw std::invalid_argument("PCA query feature width differs from fitting");
    auto out = matmul(x - mean_, components_);
    projection_detail::Floating(out);
    return out;
  }
  /// @brief Reconstruct retained-score rows with a frozen leading-axis prefix.
  ///
  /// Computes mean+projection*V_K^T for finite Float32/64 CPU projection[M,K],
  /// M>0 and1<=K<=R. A shorter projection selects the first K retained axes,
  /// with no refit, whitening, clipping or restoration of discarded coordinates.
  /// Output remains Float64 in original feature units; it may leave the input
  /// pixel range. Strides and projection gradients survive widening; fitting
  /// state is frozen. Inputs unchanged. Work O(M*D*K), storage O(M*(D+K)+D*K).
  /// @param projection Finite signed score matrix[M,K], positive M and1<=K<=R.
  /// @return Owning live Float64[M,D] inverse projection, without clipping.
  /// @throws std::invalid_argument For invalid score shape/width or nonfinite computation.
  /// @throws std::logic_error For non-CPU storage.
  /// @par Example
  /// @code
  /// #include <dm/dm.hpp>
  /// namespace n=dm::nn;
  /// const n::models::SvdPca fit(n::tensor({-1.,0.,0.,0.,1.,0.},{3,2}),1);
  /// const auto origin=fit.Reconstruct(n::zeros({2,1},n::DType::Float64));
  /// // Both reconstructed rows equal the frozen fitting mean [0,0].
  /// @endcode
  /// @see SvdPca, Transform()
  Tensor Reconstruct(const Tensor& projection) const {
    if (!projection.defined() || projection.dim() != 2)
      throw std::invalid_argument("PCA inverse needs score rows");
    const auto score = DoubleSampleRows(projection);
    if (score.size(1) > components_.size(1))
      throw std::invalid_argument("PCA inverse has too many coordinates");
    auto axes = score.size(1) == components_.size(1)
                    ? components_
                    : slice(components_, {0, 0}, {components_.size(0), score.size(1)}).contiguous();
    auto out  = mean_ + matmul(score, axes, false, true);
    projection_detail::Floating(out);
    return out;
  }
};

/// @brief Return deterministic Euclidean nearest fitting-row indices for explicit queries.
///
/// Widens finite signed Float32/64 CPU sample tensors with DoubleSampleRows;
/// flattened widths must match. For each pair, subtracts and sums squared
/// differences in feature order in double, then sorts(distance,fitting row).
/// Exact distance ties use smaller fitting rows. No normalization, label,
/// metric fitting or query-dependent reference change is implicit. k distinct
/// fitting rows are returned, nearest first, as owning detached Int32[M,k].
/// Strides are accepted; inputs unchanged. Nonfinite computed distances are
/// rejected. Work O(M*N*D+M*N*log(k+1)), storage O((N+M)*D+N+M*k), without a
/// full M-by-N distance matrix. Synchronize concurrent input mutation.
/// @param fitting Finite floating CPU samples[N,...], 1<=N<=Int32 max.
/// @param query Finite floating CPU samples[M,...], M>0, same flattened width D.
/// @param neighbors Positive k<=N, default10.
/// @return Owning detached Int32[M,k] fitting-row indices, ordered by distance then row.
/// @throws std::invalid_argument For invalid inputs/options/products or distance overflow.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto rows=n::models::NearestFitRows(n::tensor({-1.,1.,4.},{3,1}),n::zeros({1,1}),2);
/// // rows[1,2] is [0,1]: equal-distance fitting rows use the smaller index first.
/// @endcode
/// @par References
/// - Cover and Hart (1967), Nearest Neighbor Pattern Classification,
///   https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf . This utility
///   returns fixed Euclidean neighbors for a descriptive probe, without risk-bound claims.
/// @see NeighborOverlap(), NeighborLabelAccuracy(), DoubleSampleRows()
inline Tensor
NearestFitRows(const Tensor& fitting, const Tensor& query, std::int64_t neighbors = 10) {
  NoGradGuard guard;
  const auto  fit = DoubleSampleRows(fitting), q = DoubleSampleRows(query);
  if (fit.size(1) != q.size(1) || neighbors <= 0 || neighbors > fit.size(0) ||
      fit.size(0) > std::numeric_limits<std::int32_t>::max() ||
      q.size(0) > std::numeric_limits<std::int64_t>::max() / neighbors)
    throw std::invalid_argument(
        "nearest fitting rows need matching widths and a representable positive k");
  const auto                f = fit.to_vector<double>(), x = q.to_vector<double>();
  const auto                width = fit.size(1);
  std::vector<std::int32_t> out(q.size(0) * neighbors);
  for (std::int64_t row = 0; row < q.size(0); ++row) {
    std::vector<std::pair<double, std::int32_t>> distances;
    distances.reserve(fit.size(0));
    for (std::int64_t other = 0; other < fit.size(0); ++other) {
      double distance = 0;
      for (std::int64_t j = 0; j < width; ++j) {
        const double delta = x[row * width + j] - f[other * width + j];
        distance += delta * delta;
      }
      if (!std::isfinite(distance))
        throw std::invalid_argument("Euclidean distance accumulation overflowed");
      distances.emplace_back(distance, std::int32_t(other));
    }
    std::partial_sort(distances.begin(), distances.begin() + neighbors, distances.end());
    for (std::int64_t j = 0; j < neighbors; ++j)
      out[row * neighbors + j] = distances[j].second;
  }
  return from_vector(out, {q.size(0), neighbors});
}

/// @brief Measure row-aligned reference-neighbor membership in candidate lists.
///
/// Both positive [M,k] tables contain nonnegative Int32/64 CPU row IDs, with
/// matching shapes; integer dtypes may differ. Each reference entry counts
/// once if its ID occurs anywhere in the matching candidate row. Repeated
/// reference IDs count separately; repeated candidate IDs add no extra hits.
/// No ranking, labels, refit or gradients enter. Returns hits/(M*k) in[0,1].
/// Strides accepted, inputs unchanged. Work O(M*k*k), host storage O(M*k).
/// @param reference Nonnegative integer reference-neighbor table[M,k].
/// @param candidate Nonnegative integer candidate-neighbor table[M,k].
/// @return Descriptive scalar fraction of reference entries retained by candidates.
/// @throws std::invalid_argument For invalid tables, values or shape mismatch.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// overlap=n::models::NeighborOverlap(n::from_vector(std::vector<int>{0,1},{1,2}),n::from_vector(std::vector<int>{1,2},{1,2}));
/// // overlap is .5; order within each neighbor row does not affect membership.
/// @endcode
/// @see NearestFitRows(), NeighborLabelAccuracy()
inline double NeighborOverlap(const Tensor& reference, const Tensor& candidate) {
  const auto r = projection_detail::Indices(reference, 2),
             c = projection_detail::Indices(candidate, 2);
  if (reference.shape() != candidate.shape())
    throw std::invalid_argument("neighbor overlap needs matching table shapes");
  std::int64_t hits  = 0;
  const auto   width = reference.size(1);
  for (std::int64_t row = 0; row < reference.size(0); ++row)
    for (std::int64_t j = 0; j < width; ++j)
      if (std::find(c.begin() + row * width, c.begin() + (row + 1) * width, r[row * width + j]) !=
          c.begin() + (row + 1) * width)
        ++hits;
  return double(hits) / double(reference.numel());
}

/// @brief Score the first fitting neighbor's label against explicit posthoc query labels.
///
/// Nonnegative Int32/64 CPU neighbors[M,k], fit_labels[N], query_labels[M], all
/// positive axes, may use different integer dtypes/strides. Every neighbor ID
/// must be in[0,N), including unused later columns. Only column0 predicts the
/// label; k>1 does not introduce majority voting. Query labels may name classes
/// absent from fitting and simply score incorrect. There is no label-based
/// distance selection, fitting, normalization, RNG or mutation. Work/storage
/// O(M*k+N), including validation; output is a detached descriptive ratio.
/// @param neighbors Ordered fitting-row indices[M,k], nearest row in column0.
/// @param fit_labels Nonnegative integer fitting labels[N].
/// @param query_labels Aligned nonnegative integer diagnostic labels[M].
/// @return Fraction of queries whose first neighbor has the same label.
/// @throws std::invalid_argument For invalid tables/labels, alignment or out-of-range IDs.
/// @throws std::logic_error For non-CPU storage.
/// @par Example
/// @code
/// #include <dm/dm.hpp>
/// namespace n=dm::nn;
/// const auto
/// accuracy=n::models::NeighborLabelAccuracy(n::from_vector(std::vector<int>{1},{1,1}),n::from_vector(std::vector<int>{3,7}),n::from_vector(std::vector<int>{7}));
/// // accuracy is 1; the query label was used only after the neighbor was supplied.
/// @endcode
/// @par References
/// - Cover and Hart (1967), https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf .
///   Uses a fixed1NN diagnostic, without an asymptotic error-bound claim.
/// @see NearestFitRows(), NeighborOverlap()
inline double NeighborLabelAccuracy(
    const Tensor& neighbors,
    const Tensor& fit_labels,
    const Tensor& query_labels) {
  const auto rows = projection_detail::Indices(neighbors, 2),
             f    = projection_detail::Indices(fit_labels, 1),
             q    = projection_detail::Indices(query_labels, 1);
  if (query_labels.size(0) != neighbors.size(0))
    throw std::invalid_argument("query labels must align with neighbor rows");
  for (auto row : rows)
    if (row >= fit_labels.size(0))
      throw std::invalid_argument("fitting neighbor index outside label pool");
  std::int64_t hits = 0;
  for (std::int64_t row = 0; row < neighbors.size(0); ++row)
    hits += f[rows[row * neighbors.size(1)]] == q[row];
  return double(hits) / double(neighbors.size(0));
}
} // namespace datamunge::nn::models
