#pragma once
/// @file ica.hpp
/// @brief Inspectable deflationary FastICA, full-step diagnostics and source-recovery benchmarks.
#include <cstddef>
#include <cstdint>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <iosfwd>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Controls retained PCA dimensions, contrast, iterations, angle tolerance and
/// initialization. Whitening uses population covariance (divisor n), matching the fixed-point
/// expectations.
/// @par Example
/// @code
/// const dm::stats::ICAOptions o{2,"cube",500,1e-8,7};
/// std::print("{}\n",o.n_components); // 2
/// @endcode
/// @par References
/// - A. Hyvarinen (1999), Fast and Robust Fixed-Point Algorithms for Independent
///   Component Analysis. IEEE Transactions on Neural Networks 10(3), 626-634.
///   https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99_reprint.pdf
/// @see ICA
struct ICAOptions {
  std::size_t n_components{
      2}; ///< Positive retained PCA rank, at most features and complete rows minus one.
  std::string   contrast{"logcosh"}; ///< logcosh (tanh), exp (u*exp(-u*u/2)), or cube (u^3).
  std::size_t   max_iter{200}; ///< Positive maximum fixed-point updates per deflated component.
  double        tol{1e-6};     ///< Finite angle criterion in (0,1): 1-abs(new dot old) < tol.
  std::uint64_t seed{
      42}; ///< mt19937_64/normal_distribution initialization; repeatable within the same STL.
};
/// @brief Owned raw projected FastICA update, normalized direction and sign-invariant angle change.
/// raw_update is E[z*g(z dot w)]-E[g']*w after projection off previous rows.
/// A zero/nonfinite update is an error, not a convergence certificate.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> z(4,2,{-1,-1,-1,1,1,-1,1,1});
/// const auto a=dm::stats::ica_fixed_point_step(z,{.8,.6},"cube");
/// std::print("{:.3f}\n",a.raw_update[0]); // -1.024
/// @endcode
/// @par References
/// - A. Hyvarinen (1999), Fast and Robust Fixed-Point Algorithms for Independent
///   Component Analysis. IEEE Transactions on Neural Networks 10(3), 626-634.
///   https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99_reprint.pdf
/// @see ica_fixed_point_step()
struct ICAFixedPointStep {
  std::vector<double> raw_update; ///< Owned projected, unnormalized k-vector.
  std::vector<double>
         direction;          ///< Owned unit k-vector, before final fitted-source sign convention.
  double update_norm{0};     ///< Norm of raw_update, strictly positive for a returned step.
  double alignment_error{0}; ///< max(0,1-abs(direction dot old)), with unit old direction.
};
/// @brief Computes one deflationary FastICA update on caller-supplied whitened observations.
/// Rows are observations; assumes population covariance identity. This statistical
/// precondition is not inferred or verified. Previous directions must be orthonormal
/// rows, and w must be unit and orthogonal to them, within 1e-8. Two Gram-Schmidt passes
/// enforce deflation before normalization. No model is fitted or mutated.
/// Work O(n*k+k^2), storage O(k), excluding input validation.
/// @param whitened Finite n-by-k matrix, n>=2, k>=1.
/// @param w Finite unit k-vector.
/// @param contrast logcosh, exp or cube, default logcosh.
/// @param previous Optional finite r-by-k orthonormal prior rows, r<k; default empty.
/// @return Owned projected raw update, normalized direction, norm and angle change.
/// @throws std::invalid_argument Invalid dimensions/contrast/directions or nonfinite arithmetic.
/// @throws std::runtime_error The projected update has zero norm.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> z(4,2,{-1,-1,-1,1,1,-1,1,1});
/// const auto a=dm::stats::ica_fixed_point_step(z,{.8,.6},"cube");
/// std::print("{:.6f}\n",a.direction[1]/a.direction[0]); // 0.421875
/// @endcode
/// @par References
/// - A. Hyvarinen (1999), Fast and Robust Fixed-Point Algorithms for Independent
///   Component Analysis. IEEE Transactions on Neural Networks 10(3), 626-634.
///   https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99_reprint.pdf
/// @see ICA::convergence_history()
[[nodiscard]] ICAFixedPointStep ica_fixed_point_step(
    const linalg::DenseMatrix<double>& whitened,
    const std::vector<double>&         w,
    const std::string&                 contrast = "logcosh",
    const linalg::DenseMatrix<double>& previous = {});
/// @brief Deflationary FastICA with population whitening and frozen new-row transforms.
/// Fits centered x=A*s under the modelling assumption of mutually independent sources
/// and a full-column-rank instantaneous linear mixture. At most one source may be
/// Gaussian for identifiability up to sign/scale/permutation; Gaussian subspaces remain
/// rotationally unidentifiable. Finite-sample contrast optimization does not certify
/// statistical independence, non-Gaussianity or true-source recovery. PCA truncation
/// discards information before ICA. No standardization of original feature units occurs.
/// Null feature rows are dropped jointly during fitting; retained nonfinite values are
/// rejected. Uses 1/n covariance and matching 1/n fixed-point averages. Eigenvalue rank
/// threshold is max(n,p)*epsilon*largest eigenvalue after a common feature-scale factor.
/// Returned sources have population covariance identity to numerical precision, even
/// when a finite iteration budget is exhausted. Final signs make the first largest-
/// absolute training value positive; component order is extraction order, not variance.
/// Owns means, retained rows, PCA whitening, rotation, training arrays and diagnostics.
/// Inputs remain unchanged; const inference has no shared mutable caches. Same seed/STL
/// reproduces initialization, but normal_distribution is not bit-identical across STLs.
/// Fit work O(n*p^2+p^3+sum(iterations)*n*k+sum(iterations)*k^2); retained storage
/// O(n*(p+k)+p^2+k^2+k*sum(iterations)). Jacobi eigensolver uses at most 100 sweeps.
/// @par Example
/// @code
/// const auto d=dm::stats::ica_benchmark(100,42,"independent");
/// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
/// std::print("{} {}\n",m.observations(),m.n_components()); // 100 3
/// @endcode
/// @par References
/// - A. Hyvarinen (1999), Fast and Robust Fixed-Point Algorithms for Independent
///   Component Analysis. IEEE Transactions on Neural Networks 10(3), 626-634.
///   https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99_reprint.pdf
/// @see diagnostics(), component_diagnostics(), ica_source_matches()
class ICA {
 public:
  /// @brief Fits population-whitened components immediately from numeric feature columns.
  /// @param data Borrowed input; rows with a null selected feature are omitted jointly.
  /// @param feature_columns Unique nonempty numeric names, defining original feature order/units.
  /// @param options Validated retained dimension, contrast, iteration/tolerance/seed controls.
  /// @throws std::invalid_argument Invalid options/schema, fewer than two complete rows,
  /// duplicate names, retained nonfinite values or insufficient numerical covariance rank.
  /// @throws std::runtime_error A deflated initialization or contrast update collapses to zero.
  /// @par Example
  /// See the class example.
  /// @par References
  /// - A. Hyvarinen (1999), Fast and Robust Fixed-Point Algorithms for Independent
  ///   Component Analysis. IEEE Transactions on Neural Networks 10(3), 626-634.
  ///   https://www.cs.helsinki.fi/u/ahyvarin/papers/TNN99_reprint.pdf
  /// @see ICAOptions
  ICA(const dstruct::DataFrame&       data,
      const std::vector<std::string>& feature_columns,
      ICAOptions                      options = {});
  /// @brief Original feature order.
  /// @return Borrowed names, valid for model lifetime.
  [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
  /// @brief Number of complete fitting rows.
  /// @return Retained row count.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Retained component count.
  /// @return Positive source-column count.
  [[nodiscard]] std::size_t n_components() const { return sources_.cols(); }
  /// @brief Original zero-based row indices retained for fitting.
  /// @return Borrowed ordered row map, valid for model lifetime.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {
    return kept_row_indices_;
  }
  /// @brief Training sources with population covariance identity.
  /// @return Borrowed n-by-k matrix in kept-row/extraction order; valid for model lifetime.
  [[nodiscard]] const linalg::DenseMatrix<double>& sources() const { return sources_; }
  /// @brief Original-unit unmixing B, mapping x-mean to sources.
  /// @return Borrowed k-by-p B; valid for model lifetime.
  [[nodiscard]] const linalg::DenseMatrix<double>& unmixing_matrix() const { return unmixing_; }
  /// @brief Moore-Penrose inverse A of the fitted unmixing map.
  /// @return Borrowed p-by-k A; A*B is the retained PCA projector, B*A is identity.
  [[nodiscard]] const linalg::DenseMatrix<double>& mixing_matrix() const { return mixing_; }
  /// @brief Training feature means used unchanged on every query.
  /// @return Borrowed p-vector in feature order, valid for model lifetime.
  [[nodiscard]] const std::vector<double>& mean() const { return mean_; }
  /// @brief Completed updates per component, including exhausted budgets.
  /// @return Borrowed k-vector, valid for model lifetime.
  [[nodiscard]] const std::vector<std::size_t>& iterations() const { return iterations_; }
  /// @brief Whether every direction met the sign-invariant angle tolerance.
  /// @return True only for the numerical stopping rule; no independence or accuracy certificate.
  [[nodiscard]] bool converged() const { return converged_; }
  /// @brief Empirical excess kurtosis using population second/fourth moments.
  /// @return Borrowed k-vector; fourth moment divided by squared second moment minus three.
  [[nodiscard]] const std::vector<double>& source_kurtosis() const { return source_kurtosis_; }
  /// @brief Fitted PCA whitening map K with z=(x-mean)*K for row vectors.
  /// @return Borrowed p-by-k K, valid for model lifetime.
  [[nodiscard]] const linalg::DenseMatrix<double>& whitening_matrix() const { return whitening_; }
  /// @brief Complete centered training rows after PCA whitening.
  /// @return Borrowed n-by-k matrix with Z transpose Z / n identity.
  [[nodiscard]] const linalg::DenseMatrix<double>& whitened_data() const { return whitened_; }
  /// @brief Sign-fixed orthonormal extraction directions in whitened coordinates.
  /// @return Borrowed k-by-k R; sources=Z*R transpose, B=R*K transpose.
  [[nodiscard]] const linalg::DenseMatrix<double>& rotation_matrix() const { return rotation_; }
  /// @brief Actual unit initial vectors after projection off preceding extracted directions.
  /// @return Borrowed k-by-k rows before final source-sign fix; replay needs actual preceding
  /// directions.
  [[nodiscard]] const linalg::DenseMatrix<double>& initial_directions() const { return initial_; }
  /// @brief Original-unit population covariance spectrum in descending order.
  /// @return Borrowed p-vector; PCA retains the first k entries.
  [[nodiscard]] const std::vector<double>& whitening_eigenvalues() const { return eigenvalues_; }
  /// @brief Copies one training source in kept-row order.
  /// @param index Zero-based component index.
  /// @return Owned n-vector.
  /// @throws std::out_of_range index>=n_components().
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",m.component(0).size()); // 100
  /// @endcode
  /// @see sources()
  [[nodiscard]] std::vector<double> component(std::size_t index) const;
  /// @brief Applies the frozen training mean and unmixing matrix to new rows.
  /// No query centring/refitting or null dropping. Work O(rows*p*k).
  /// @param newdata Borrowed frame with all fitted numeric columns. Empty rows are allowed
  /// if the numeric schema exists; null/nonfinite query values are rejected.
  /// @return Owned rows-by-k scores in original query order.
  /// @throws std::invalid_argument Missing/non-numeric columns, nulls or nonfinite arithmetic.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",m.transform(d).rows()); // 100
  /// @endcode
  /// @see inverse_transform()
  [[nodiscard]] linalg::DenseMatrix<double> transform(const dstruct::DataFrame& newdata) const;
  /// @brief Maps finite source rows to original feature units using the fitted mixing matrix.
  /// With fewer components this reconstructs only the retained PCA subspace. Work O(rows*p*k).
  /// @param scores Borrowed finite rows-by-k matrix, allowing zero rows.
  /// @return Owned rows-by-p reconstruction, including training means.
  /// @throws std::invalid_argument Wrong column count or nonfinite arithmetic.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",m.inverse_transform(m.sources()).cols()); // 3
  /// @endcode
  /// @see transform()
  [[nodiscard]] linalg::DenseMatrix<double> inverse_transform(
      const linalg::DenseMatrix<double>& scores) const;
  /// @brief Summarizes whitening, orthogonality and retained-space reconstruction checks.
  /// No fitting or mutation; work O(n*(p*k+k*k)+p*p).
  /// @return Owned one-row table: observations, features, components, converged (0/1),
  /// whitening_error, source_covariance_error, rotation_error (maximum absolute identity
  /// errors), reconstruction_rmse per original-unit feature cell, retained_variance_fraction
  /// and rank_threshold in original covariance units.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",m.diagnostics().nrows()); // 1
  /// @endcode
  /// @see component_diagnostics()
  [[nodiscard]] dstruct::DataFrame diagnostics() const;
  /// @brief Reports each component's numerical stopping and empirical moments.
  /// Converged flags describe the angle rule, not independence. Work O(n*k).
  /// @return Owned k rows: component (one-based), iterations, converged (0/1),
  /// alignment_error, mean, population_variance, excess_kurtosis.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",m.component_diagnostics().nrows()); // 3
  /// @endcode
  /// @see convergence_history()
  [[nodiscard]] dstruct::DataFrame component_diagnostics() const;
  /// @brief Returns every completed unsigned fixed-point update, before final sign fixing.
  /// History is grouped by component, with iterations starting at one. No refit.
  /// @return Owned component, iteration, alignment_error, update_norm and w0..w(k-1)
  /// columns. Initial directions are returned separately by initial_directions().
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",m.convergence_history().nrows()>0); // true
  /// @endcode
  /// @see initial_directions()
  [[nodiscard]] dstruct::DataFrame convergence_history() const;
  /// @brief Formats counts, convergence and empirical kurtosis as readable text.
  /// @return Owned summary string.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::print("{}\n",!m.summary().empty()); // true
  /// @endcode
  /// @see print_summary()
  [[nodiscard]] std::string summary() const;
  /// @brief Writes the summary to a borrowed stream.
  /// @param os Borrowed output stream; formatting flags may change.
  /// @return Nothing.
  /// @throws std::ios_base::failure Stream failure if exceptions are enabled.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// std::ostringstream out;m.print_summary(out);
  /// std::print("{}\n",!out.str().empty()); // true
  /// @endcode
  /// @see summary()
  void print_summary(std::ostream& os) const;
  /// @brief Writes the summary to standard output.
  /// @return Nothing.
  /// @par Example
  /// See the stream overload, or call m.print_summary().
  /// @see summary()
  void print_summary() const;
  /// @brief Plots training sources coloured by external labels that are not used in fitting.
  /// @param group_labels One label per complete training row, in kept-row order.
  /// @param component_x Zero-based horizontal component, default zero.
  /// @param component_y Zero-based vertical component, default one.
  /// @return Owned plot with lexically ordered groups.
  /// @throws std::invalid_argument Wrong label count.
  /// @throws std::out_of_range Invalid component index.
  /// @par Example
  /// @code
  /// const auto d=dm::stats::ica_benchmark(100,42,"independent");
  /// const dm::stats::ICA m(d,{"x1","x2","x3"},{3});
  /// const auto p=m.plot_sources(std::vector<std::string>(100,"all"));
  /// std::print("{}\n",p.series().size()); // 1
  /// @endcode
  /// @see kept_row_indices()
  [[nodiscard]] graphics::RPlot plot_sources(
      const std::vector<std::string>& group_labels,
      std::size_t                     component_x = 0,
      std::size_t                     component_y = 1) const;

 private:
  /// @brief Validates complete rows, whitens, extracts directions and reconstructs maps.
  /// @param data Borrowed constructor frame.
  /// @throws std::invalid_argument Invalid schema/options/rank or nonfinite arithmetic.
  /// @throws std::runtime_error Degenerate initialization or contrast update.
  void                        fit(const dstruct::DataFrame& data);
  std::vector<std::string>    feature_columns_;     ///< Frozen original feature order.
  ICAOptions                  options_;             ///< Validated fitting controls.
  std::vector<std::size_t>    kept_row_indices_;    ///< Complete original training row map.
  std::vector<double>         mean_;                ///< Original-unit training means.
  std::vector<double>         source_kurtosis_;     ///< Population empirical excess kurtoses.
  std::vector<std::size_t>    iterations_;          ///< Completed updates per component.
  std::vector<bool>           component_converged_; ///< Per-component angle stopping flags.
  std::vector<double>         alignment_errors_;    ///< Last per-component angle changes.
  std::vector<double>         eigenvalues_; ///< Original-unit descending covariance spectrum.
  linalg::DenseMatrix<double> sources_;     ///< Population-standardized n-by-k source estimates.
  linalg::DenseMatrix<double> unmixing_;    ///< k-by-p original-unit map B.
  linalg::DenseMatrix<double> mixing_;      ///< p-by-k pseudo-inverse A.
  linalg::DenseMatrix<double> whitening_;   ///< p-by-k population PCA whitening K.
  linalg::DenseMatrix<double> whitened_;    ///< Complete n-by-k training whitened rows.
  linalg::DenseMatrix<double> rotation_;    ///< Sign-fixed orthonormal k-by-k rows R.
  linalg::DenseMatrix<double> initial_;     ///< Unsigned initial directions after deflation.
  linalg::DenseMatrix<double> centered_;    ///< Complete n-by-p original-unit centered rows.
  dstruct::DataFrame          history_;     ///< Owned completed-update diagnostics and directions.
  std::size_t                 observations_{0}; ///< Complete training rows.
  bool   converged_{false};                     ///< Whether all components met the angle criterion.
  double retained_variance_fraction_{0}; ///< Fraction of scaled covariance trace retained by PCA.
  double rank_threshold_{0};             ///< Original-unit numerical covariance cutoff.
};
/// @brief Generates known-source mixtures for ICA accuracy and limitation checks.
/// kind=independent draws iid unit-population-variance Laplace, uniform and Rademacher
/// sources; gaussian draws three iid standard normals; waveforms uses deterministic
/// sine(.060*i), sign(sin(.093*i)) and fmod(.050*i,2)-1 (not independent random sources).
/// These three-source cases use A rows (.8,.3,.2),(.4,.9,.5),(.6,.2,.7).
/// kind=uniform uses two iid uniform(-sqrt(3),sqrt(3)) sources and A rows (1,.8),(.3,1).
/// Xoshiro256** generates draws; finite samples need not have exactly identity moments.
/// No real observations or noise are added, and truth is never needed by ICA fitting.
/// @param n At least two rows, default 1000.
/// @param seed Local random seed, default 42; waveforms ignores it.
/// @param kind independent (default), gaussian, waveforms or uniform.
/// @return Owned t,s1..sk,x1..xk frame, t=zero-based sample index, with k=2 or 3.
/// @throws std::invalid_argument n<2 or unknown kind.
/// @par Example
/// @code
/// const auto d=dm::stats::ica_benchmark(20,7,"uniform");
/// std::print("{} {}\n",d.nrows(),d.ncols()); // 20 5
/// @endcode
/// @see ICA, ica_source_matches()
[[nodiscard]] dstruct::DataFrame ica_benchmark(
    std::size_t        n    = 1000,
    std::uint64_t      seed = 42,
    const std::string& kind = "independent");
/// @brief Matches known reference sources to distinct recovered components by absolute correlation.
/// Maximizes the sum of absolute Pearson correlations with the Hungarian assignment;
/// one recovered component cannot be reused for multiple true sources. This is a
/// supervised benchmark diagnostic, not something available for unknown real sources.
/// Work O(n*k^2+k^3), storage O(k^2+n*k). No mutation or model fitting occurs.
/// @param reference Finite n-by-k known source rows, n>=2, k>=1; no constant columns.
/// @param recovered Finite matching n-by-k estimates; no constant columns.
/// @return Owned k rows: source, component (both one-based), correlation, absolute_correlation,
/// offset and multiplier. offset+multiplier*recovered[:,component] aligns empirical
/// means/scales and sign to the reference; it does not make an imperfect match exact.
/// @throws std::invalid_argument Shape mismatch, constants or nonfinite arithmetic.
/// @par Example
/// @code
/// const dm::linalg::DenseMatrix<double> s(4,2,{-1,-1,-1,1,1,-1,1,1});
/// const auto table=dm::stats::ica_source_matches(s,s);
/// std::print("{:.0f}\n",table.double_at("absolute_correlation",0)); // 1
/// @endcode
/// @par References
/// - H. W. Kuhn (1955), The Hungarian method for the assignment problem.
///   Naval Research Logistics Quarterly 2, 83-97. https://doi.org/10.1002/nav.3800020109
/// @see ICA::sources(), algorithms::hungarian()
[[nodiscard]] dstruct::DataFrame ica_source_matches(
    const linalg::DenseMatrix<double>& reference,
    const linalg::DenseMatrix<double>& recovered);
} // namespace datamunge::stats
