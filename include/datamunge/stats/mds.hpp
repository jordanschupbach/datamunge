#pragma once

/// @file mds.hpp
/// @brief Classical multidimensional scaling from features or dissimilarities, with spectral and distance diagnostics.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/knn_classifier.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Number of embedding axes and feature-distance metric; features are not standardised.
///
/// @par Example
/// @code
/// const dm::stats::MDSOptions options{2, dm::stats::DistanceMetric::Manhattan};
/// const dm::stats::MDS model(dm::datasets::iris(), {"Petal.Length", "Petal.Width"}, options);
/// @endcode
/// @see MDS
struct MDSOptions {
    std::size_t n_components{2}; ///< Requested axes, at least one and fewer than complete observations.
    DistanceMetric metric{DistanceMetric::Euclidean}; ///< Euclidean or Manhattan distance on raw numeric features.
};

/// @brief Classical MDS (principal coordinates), similar to R's cmdscale without an additive correction.
///
/// Double-centres squared dissimilarities to B=-J D^2 J/2, then retains the leading
/// positive eigenpairs. This minimises squared Gram error (strain), not squared distance
/// error (stress). Negative eigenvalues are exposed and clamped to zero for coordinates.
/// Euclidean feature distances recover unscaled, centred PCA scores up to an orthogonal
/// transformation. No predictor standardisation is performed. Dense fitting takes O(n^2)
/// storage and O(n^3) work per Jacobi sweep; feature distances additionally cost O(n^2*p).
/// Returned matrices are borrowed for the model lifetime unless documented as owned.
/// Const methods do not mutate the fit; signs and axes within repeated eigenspaces are arbitrary.
///
/// @par Example
/// @code
/// const dm::stats::MDS model(dm::datasets::iris(),
///     {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});
/// std::print("{:.4f}\n", model.goodness_of_fit()); // 0.9777
/// @endcode
///
/// @par References
/// - W. S. Torgerson (1952). Multidimensional scaling: I. Theory and method.
///   Psychometrika 17, 401-419. https://doi.org/10.1007/BF02288916
/// - J. C. Gower (1966). Some distance properties of latent root and vector methods used in
///   multivariate analysis. Biometrika 53, 325-338. https://doi.org/10.1093/biomet/53.3-4.325
/// @see from_distances(), dimension_diagnostics(), PCA
class MDS {
public:
    /// @brief Fits classical MDS on distances between unscaled numeric feature rows.
    /// @param data Input frame; rows with a null in any selected feature are dropped.
    /// @param feature_columns Non-empty list of numeric columns defining the metric.
    /// @param options Number of axes and Euclidean/Manhattan metric, as MDSOptions defaults.
    /// @throws std::invalid_argument For missing/non-numeric features, non-finite retained values,
    ///         invalid metric, invalid component count, or distances/Gram entries outside finite range.
    ///
    /// @par Example
    /// See the class example; kept_row_indices() aligns labels after null dropping.
    /// @see MDS, from_distances()
    MDS(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, MDSOptions options = {});

    /// @brief Classical scaling of an explicit dissimilarity matrix, as in R's cmdscale(d).
    ///
    /// No triangle inequality or Euclidean assumption is required. All entries must be finite
    /// and non-negative. Symmetry and zero diagonal are checked with tolerance 1e-12 times
    /// the largest dissimilarity; accepted asymmetric entries are averaged and diagonal set
    /// to zero. Input is copied; no null-dropping, standardisation or additive correction.
    /// feature_names() is empty and kept_row_indices() is 0,...,n-1. Axes beyond positive
    /// rank are zero or rounding-sized; exactly n_components columns are always returned.
    /// @param dissimilarities Square n by n matrix of distances, not squared distances, n >= 2.
    /// @param n_components Requested axes, default two, in [1,n-1].
    /// @return Owned fitted model; the input is unchanged.
    /// @throws std::invalid_argument For invalid shape, entries, diagonal, symmetry, component
    ///         count, or squared distances/centred Gram entries outside finite range.
    ///
    /// @par Example
    /// @code
    /// const dm::linalg::DenseMatrix<double> distances(3,3,{0,1,2, 1,0,1, 2,1,0});
    /// const auto model = dm::stats::MDS::from_distances(distances,1);
    /// std::print("{:.3f}\n", model.goodness_of_fit()); // 1.000
    /// @endcode
    ///
    /// @par References
    /// - W. S. Torgerson (1952). Multidimensional scaling: I. Theory and method.
    ///   https://doi.org/10.1007/BF02288916
    /// @see MDS, gram_matrix(), spectrum()
    [[nodiscard]] static MDS from_distances(const linalg::DenseMatrix<double>& dissimilarities,
                                           std::size_t n_components = 2);

    /// @brief Selected feature names, empty for precomputed dissimilarities.
    /// @return Borrowed names in input order.
    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    /// @brief Number of retained observations.
    /// @return Fitted row count.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of requested axes, including zero axes beyond positive rank.
    /// @return Coordinate column count.
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }
    /// @brief Original input row indices surviving null-dropping, in embedding order.
    /// @return Borrowed zero-based indices, or 0,...,n-1 for explicit distances.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }
    /// @brief Full descending spectrum of the centred Gram matrix, including negative values.
    /// @return Borrowed eigenvalues, length observations().
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }
    /// @brief Retained positive eigenvalue sum divided by the sum of all positive eigenvalues.
    ///
    /// Matches R cmdscale's second GOF component (up to eigenvalue round-off), not stress.
    /// A value of one can coexist with distance distortion when there are negative eigenvalues.
    /// @return Positive-inertia fraction, or zero if the positive sum is zero.
    [[nodiscard]] double goodness_of_fit() const { return goodness_of_fit_; }
    /// @brief Fitted coordinates with n_components columns, centred up to round-off.
    /// @return Borrowed n by k matrix.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    /// @brief Accepted original dissimilarities, before squaring or centring.
    /// @return Borrowed n by n matrix in retained-row order.
    [[nodiscard]] const linalg::DenseMatrix<double>& dissimilarities() const { return dissimilarities_; }
    /// @brief Double-centred squared dissimilarities B=-J D^2 J/2, before spectral truncation.
    /// @return Borrowed symmetric n by n matrix, with row sums zero up to round-off.
    [[nodiscard]] const linalg::DenseMatrix<double>& gram_matrix() const { return gram_; }
    /// @brief Copies one coordinate axis; see the class example for a fitted model.
    /// @param index Zero-based axis, less than n_components().
    /// @return Owned n-vector of coordinates.
    /// @throws std::out_of_range For an invalid index.
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    /// @brief Spectral summary exposing both positive and negative inertia.
    ///
    /// Columns: dimension (one-based), eigenvalue, positive_share, absolute_share,
    /// cumulative_positive_share, retained (1 if within requested axes), significant_negative.
    /// The last flag uses lambda < -1e-10*max(abs(lambda)) to distinguish round-off from
    /// meaningful negative eigenvalues. Shares use the raw spectrum and zero denominators give zero.
    /// @return Owned n-row table, O(n) time and storage.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::MDS model(dm::datasets::iris(), {"Petal.Length", "Petal.Width"});
    /// std::print("{}\n", model.spectrum().head(5).to_md(4));
    /// @endcode
    /// @see eigenvalues(), dimension_diagnostics()
    [[nodiscard]] dstruct::DataFrame spectrum() const;

    /// @brief Distance distortion and optimal Gram residual for every fitted prefix of axes.
    ///
    /// Row k describes the first k axes, k=1,...,n_components(). Columns: dimensions,
    /// gof_positive (goodness_of_fit definition), gof_absolute (retained positive sum divided
    /// by sum(abs(lambda))), strain_squared (sum of discarded positive and all negative
    /// eigenvalues squared), relative_strain (sqrt(strain_squared/sum(lambda^2))),
    /// distance_rmse (sqrt(mean((D-dhat)^2))) and relative_distance_error (sqrt(sum((D-dhat)^2)/sum(D^2))).
    /// Distance sums are over unordered pairs. Zero denominators give zero. Strain is the
    /// spectral optimum for a positive-semidefinite Gram matrix of rank at most k; it does
    /// not establish that stress is minimised. No refits; O(n^2*k) work, O(n^2+k) temporary storage.
    /// @return Owned k-row table in increasing dimension order.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::MDS model(dm::datasets::iris(),
    ///     {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}, {4});
    /// std::print("{}\n", model.dimension_diagnostics().to_md(6));
    /// @endcode
    ///
    /// @par References
    /// - J. C. Gower (1966). Some distance properties of latent root and vector methods used in
    ///   multivariate analysis. https://doi.org/10.1093/biomet/53.3-4.325
    /// - R stats::cmdscale, GOF definitions: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/cmdscale.html
    /// @see goodness_of_fit(), pair_distances(), plot_shepard()
    [[nodiscard]] dstruct::DataFrame dimension_diagnostics() const;

    /// @brief Original and embedded distances for every unordered pair, as used in a Shepard plot.
    /// @param dimensions Prefix length, default zero means all fitted axes; otherwise 1..n_components().
    /// @return Owned n*(n-1)/2-row table: row_i, row_j (original zero-based input indices),
    ///         dissimilarity, embedded_distance, residual (embedded minus original). O(n^2*k) work/storage O(n^2).
    /// @throws std::out_of_range If dimensions exceeds n_components().
    ///
    /// @par Example
    /// @code
    /// const dm::stats::MDS model(dm::datasets::iris(), {"Petal.Length", "Petal.Width"});
    /// std::print("{}\n", model.pair_distances().head(5).to_md(4));
    /// @endcode
    /// @see plot_shepard(), dimension_diagnostics()
    [[nodiscard]] dstruct::DataFrame pair_distances(std::size_t dimensions = 0) const;

    /// @brief Text summary of settings, retained positive inertia and leading eigenvalues.
    /// @return Owned summary text; see the class example for fitting.
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary diagnostics to a stream.
    /// @param os Destination stream; its numeric formatting is changed to fixed precision four.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary diagnostics to standard output, changing its numeric formatting.
    void print_summary() const;

    /// @brief Scatter of two fitted axes; see the grouped overload for an example.
    /// @param dimension_x Horizontal axis, zero-based, default zero.
    /// @param dimension_y Vertical axis, zero-based, default one.
    /// @return Owned plot; aspect ratio follows the plotting canvas, not necessarily equal geometric units.
    /// @throws std::out_of_range For axes outside n_components().
    [[nodiscard]] graphics::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    /// @brief Scatter coloured by external labels, which do not participate in fitting.
    /// @param group_labels One label per retained row, in kept_row_indices() order.
    /// @param dimension_x Horizontal axis, default zero.
    /// @param dimension_y Vertical axis, default one.
    /// @return Owned plot, colours in sorted group order; allow legend headroom through y_limits().
    /// @throws std::out_of_range For axes outside n_components().
    /// @throws std::invalid_argument If group_labels does not match observations().
    ///
    /// @par Example
    /// @code
    /// const auto iris = dm::datasets::iris();
    /// const dm::stats::MDS model(iris, {"Petal.Length", "Petal.Width"});
    /// std::vector<std::string> labels;
    /// for (auto i : model.kept_row_indices()) labels.push_back(iris.string_at("Species",i));
    /// model.plot_embedding(labels).save_svg("mds.svg");
    /// @endcode
    /// @see kept_row_indices(), plot_shepard()
    [[nodiscard]] graphics::RPlot plot_embedding(const std::vector<std::string>& group_labels,
        std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;

    /// @brief Shepard plot: embedded distance versus input dissimilarity, with an equality line.
    /// @param dimensions Prefix length, zero (default) means all fitted axes.
    /// @return Owned plot of all unordered pairs, with equal numeric limits on both axes.
    /// @throws std::out_of_range If dimensions exceeds n_components().
    ///
    /// @par Example
    /// @code
    /// const dm::stats::MDS model(dm::datasets::iris(), {"Petal.Length", "Petal.Width"});
    /// model.plot_shepard().save_svg("mds_shepard.svg");
    /// @endcode
    /// @see pair_distances(), dimension_diagnostics()
    [[nodiscard]] graphics::RPlot plot_shepard(std::size_t dimensions = 0) const;

private:
    /// @brief Temporary empty model used only by from_distances(), which completes fitting.
    MDS() = default;
    /// @brief Builds feature distances after validating columns and dropping null rows.
    /// @param data Source frame.
    void fit(const dstruct::DataFrame& data);
    /// @brief Validates distances, double-centres their squares and computes coordinates.
    /// @param distances Square dissimilarity matrix; copied and symmetrised within tolerance.
    void fit_distances(const linalg::DenseMatrix<double>& distances);

    std::vector<std::string> feature_columns_; ///< Selected numeric features, empty for explicit distances.
    MDSOptions options_; ///< Metric and number of axes.
    std::vector<std::size_t> kept_row_indices_; ///< Original input rows in fitted order.
    std::vector<double> eigenvalues_; ///< Full unmodified descending Gram spectrum.
    double goodness_of_fit_{0.0}; ///< Retained fraction of positive inertia, zero if undefined.
    linalg::DenseMatrix<double> embedding_; ///< Fitted coordinates, zero-clamped negative eigenvalues.
    linalg::DenseMatrix<double> dissimilarities_; ///< Accepted original pairwise dissimilarities.
    linalg::DenseMatrix<double> gram_; ///< Double-centred squared dissimilarities before truncation.
    std::size_t observations_{0}; ///< Number of fitted rows.
};

} // namespace datamunge::stats
