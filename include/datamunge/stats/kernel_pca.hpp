#pragma once
/// @file kernel_pca.hpp
/// @brief Centred feature-space PCA, spectral diagnostics and frozen-training projections.
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {
/// @brief Kernel-PCA geometry, without automatic predictor standardization or gamma selection.
/// @par Example
/// @code
/// dm::stats::KernelPCAOptions o;o.kernel="linear";o.n_components=2;
/// const dm::stats::KernelPCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"},o);
/// std::print("{}\n",m.n_components()); // 2
/// @endcode
/// @see KernelPCA
struct KernelPCAOptions {
    std::size_t n_components{2}; ///< Positive retained dimension, less than n and no greater than numerical rank.
    std::string kernel{"rbf"}; ///< Exactly linear, rbf or polynomial.
    double gamma{1.0}; ///< Finite positive RBF inverse squared length scale/polynomial dot scale; ignored by linear.
    double degree{3.0}; ///< Finite positive integer polynomial degree; ignored by other kernels.
    double coef0{1.0}; ///< Finite nonnegative polynomial offset, ensuring a PSD kernel; ignored otherwise.
};
/// @brief Kernel PCA in the training-centred feature space, as sklearn.decomposition.KernelPCA.
///
/// Numeric predictors are used in their supplied units, without input scaling. Required
/// training nulls are omitted jointly; retained nonfinite values are rejected. At least
/// two complete rows are required. Kc=H*K*H, H=I-11'/n, is diagonalized with a scaled
/// symmetric eigensolver. Scores are V*sqrt(lambda); the historical eigenvalues() returns
/// retained Gram eigenvalues, not covariance variances (those are lambda/(n-1)).
/// Signs are fixed by a largest-magnitude entry, but repeated eigenspaces admit arbitrary
/// orthogonal rotations. Rank uses 64*n*epsilon*largest_eigenvalue. Requested dimensions
/// exceeding that rank throw; small signed eigenvalues remain visible in all_eigenvalues().
/// Materially negative eigenvalues throw instead of silently clipping an indefinite kernel.
///
/// Linear centering uses centred raw predictors; RBF centering uses expm1 of negative
/// squared distance, cancelling its common constant before subtraction. This preserves
/// small-gamma structure that can disappear from a rounded all-ones raw kernel matrix.
/// Fit costs O(n^2*p+n^3) time and O(n^2+n*p) space. Models own all matrices and the
/// training centre. Const operations use no shared mutable caches. No inverse transform,
/// pre-image estimator, automatic gamma choice or classifier is fitted.
/// @par Example
/// @code
/// const auto d=dm::datasets::iris();
/// const dm::stats::KernelPCA m(d,{"Petal.Length","Petal.Width"},{.gamma=.13});
/// std::print("{}\n",m.observations()); // 150
/// @endcode
/// @par References
/// - B. Schölkopf, A. Smola and K.-R. Müller (1998), Nonlinear Component Analysis as a
///   Kernel Eigenvalue Problem. Neural Computation 10(5), 1299-1319.
///   https://doi.org/10.1162/089976698300017467
/// - https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
/// @see spectrum(), transform(), projection_details()
class KernelPCA {
public:
    /// @brief Fits immediately, owning the complete training predictors and centred geometry.
    /// @param data Training frame; null required rows are omitted jointly.
    /// @param feature_columns Nonempty distinct numeric names, in metric order.
    /// @param options Kernel and retained rank, with defaults above.
    /// @throws std::invalid_argument Invalid controls/schema, too few rows, nonfinite data,
    /// unrepresentable kernel/centred geometry, materially negative spectrum or insufficient rank.
    /// @throws std::runtime_error The symmetric eigensolver fails to converge.
    /// @par Example
    /// See the class example.
    KernelPCA(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
              KernelPCAOptions options = {});
    /// @brief Predictor names defining the raw-unit metric.
    /// @return Borrowed owned names in training order.
    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    /// @brief Complete training count.
    /// @return Number of retained rows.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Requested and fitted positive dimension count.
    /// @return Number of score columns.
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }
    /// @brief Original zero-based training row indices in score order.
    /// @return Borrowed increasing indices, for aligning external labels.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }
    /// @brief Training scores V*sqrt(lambda), centred by the training feature-space mean.
    /// @return Borrowed n-by-d matrix; lifetime is the model's.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    /// @brief Copies one zero-based training score column.
    /// @param index Component index less than n_components().
    /// @return Owned vector in retained-row order.
    /// @throws std::out_of_range Invalid component.
    /// @par Example
    /// @code
    /// const dm::stats::KernelPCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"});
    /// std::print("{}\n",m.dimension(0).size()); // 150
    /// @endcode
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;
    /// @brief Retained descending centred-Gram eigenvalues, preserving the original API meaning.
    /// @return Borrowed positive eigenvalues, length n_components().
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }
    /// @brief Full descending signed centred-Gram spectrum, including numerical null directions.
    /// @return Borrowed vector of length observations().
    [[nodiscard]] const std::vector<double>& all_eigenvalues() const { return all_eigenvalues_; }
    /// @brief Raw training kernel, before feature-space centering.
    /// @return Borrowed n-by-n symmetric matrix in retained-row order.
    [[nodiscard]] const linalg::DenseMatrix<double>& kernel_matrix() const { return kernel_; }
    /// @brief Centred training kernel; more accurate than double-centering rounded K at tiny gamma.
    /// @return Borrowed n-by-n matrix, with row sums zero to roundoff.
    [[nodiscard]] const linalg::DenseMatrix<double>& centered_kernel_matrix() const { return centered_; }
    /// @brief Normalized dual coefficients V/sqrt(lambda) for the retained axes.
    /// @return Borrowed n-by-d matrix; Kc*dual_coefficients() reproduces embedding().
    [[nodiscard]] const linalg::DenseMatrix<double>& dual_coefficients() const { return dual_; }
    /// @brief Scale-relative eigenvalue threshold used for PSD/rank decisions.
    /// @return Positive threshold in Gram-eigenvalue units.
    [[nodiscard]] double eigenvalue_tolerance() const { return tolerance_; }
    /// @brief Positive spectrum count above eigenvalue_tolerance().
    /// @return Numerical rank, no greater than n-1.
    [[nodiscard]] std::size_t numerical_rank() const { return rank_; }
    /// @brief Full spectrum, feature-space variance and retained-energy curve.
    /// @return Owned component/eigenvalue/sample_variance/positive_share/cumulative_share/
    /// residual_energy/retained/resolved table. Energy sums only eigenvalues above tolerance;
    /// sample_variance is signed lambda/(n-1), not raw-measurement variance.
    /// @par Example
    /// @code
    /// const dm::stats::KernelPCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"});
    /// std::print("{}\n",m.spectrum().nrows()); // 150
    /// @endcode
    /// @par References
    /// - Schölkopf, Smola and Müller (1998). https://doi.org/10.1162/089976698300017467
    /// @see KernelPCA
    [[nodiscard]] dstruct::DataFrame spectrum() const;
    /// @brief Numerical centering/eigenpair checks and within-kernel retained feature energy.
    /// @return Owned n/p/d/rank/trace/retained_share/mean_residual_energy/max_centered_row_mean/
    /// relative_eigen_residual row. Residual energy is discarded feature-space squared norm/n.
    /// Trace here sums resolved positive eigenvalues above tolerance. The residual check is
    /// ||Kc*dual-Z||_F/||Z||_F. Shares compare dimensions within a fixed kernel/metric,
    /// not quality across kernels.
    /// @par Example
    /// @code
    /// const dm::stats::KernelPCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"});
    /// std::print("{:.0f}\n",m.diagnostics().double_at("n",0)); // 150
    /// @endcode
    /// @par References
    /// - Schölkopf, Smola and Müller (1998). https://doi.org/10.1162/089976698300017467
    /// @see KernelPCA
    [[nodiscard]] dstruct::DataFrame diagnostics() const;
    /// @brief Projects new rows onto frozen training axes using the training feature-space centre.
    /// Each query's cross-kernel mean and the stored training column/grand means centre
    /// its row. Query batches never learn a new centre; adding other queries cannot change
    /// a row's coordinates. Work O(m*n*(p+d)), output O(m*d); no fitting or label use.
    /// @param newdata Numeric training-named columns; extra columns ignored. Empty input allowed.
    /// @return Owned m-by-d scores in original query order; null/nonfinite rows are all NaN.
    /// @throws std::invalid_argument Missing/nonnumeric schema or unrepresentable kernel geometry.
    /// @par Example
    /// @code
    /// const auto d=dm::datasets::iris();
    /// const dm::stats::KernelPCA m(d,{"Petal.Length","Petal.Width"});
    /// std::print("{}\n",m.transform(d.head(3)).rows()); // 3
    /// @endcode
    /// @par References
    /// - Schölkopf, Smola and Müller (1998). https://doi.org/10.1162/089976698300017467
    /// @see KernelPCA
    [[nodiscard]] linalg::DenseMatrix<double> transform(const dstruct::DataFrame& newdata) const;
    /// @brief Feature-space query norm, retained squared norm and orthogonal projection loss.
    /// Residual is ||phi(x)-mu_train||^2 - sum(score^2), rounded below by zero. This is
    /// feature-space reconstruction loss, not original-unit error, density or a calibrated
    /// anomaly score; changing kernel/units changes its meaning. No pre-image is computed.
    /// @param newdata Same schema and row handling as transform().
    /// @return Owned query_row/centered_norm_squared/projected_norm_squared/residual_energy/status;
    /// status is ok, missing_predictor or nonfinite_predictor; invalid rows have NaN norms.
    /// @throws std::invalid_argument As transform().
    /// @par Example
    /// @code
    /// const auto d=dm::datasets::iris();
    /// const dm::stats::KernelPCA m(d,{"Petal.Length","Petal.Width"});
    /// std::print("{}\n",m.projection_details(d.head(3)).nrows()); // 3
    /// @endcode
    /// @par References
    /// - Schölkopf, Smola and Müller (1998). https://doi.org/10.1162/089976698300017467
    /// @see KernelPCA
    [[nodiscard]] dstruct::DataFrame projection_details(const dstruct::DataFrame& newdata) const;
    /// @brief Formats kernel, rank and feature-space retained energy without writing output.
    /// @return Owned summary text.
    /// @par Example
    /// See print_summary(std::ostream&).
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to a borrowed stream without changing its numeric flags.
    /// @param os Stream mutated by insertion.
    /// @par Example
    /// @code
    /// const dm::stats::KernelPCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"});
    /// std::ostringstream out;m.print_summary(out);std::print("{}\n",out.str()==m.summary()); // true
    /// @endcode
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    /// @par Example
    /// See the stream overload.
    void print_summary() const;
    /// @brief Training score scatter, with zero-based component indices.
    /// @param dimension_x Horizontal component, default zero.
    /// @param dimension_y Vertical component, default one.
    /// @return Owned editable plot.
    /// @throws std::out_of_range A requested axis is unavailable.
    /// @par Example
    /// @code
    /// const dm::stats::KernelPCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"});
    /// m.plot_embedding().save_svg("kernel_pca.svg");
    /// @endcode
    [[nodiscard]] graphics::RPlot plot_embedding(std::size_t dimension_x=0,std::size_t dimension_y=1) const;
    /// @brief Training score scatter coloured by external labels; labels never affect fitting.
    /// @param group_labels Exactly one per retained row, aligned using kept_row_indices().
    /// @param dimension_x Horizontal component, default zero.
    /// @param dimension_y Vertical component, default one.
    /// @return Owned editable plot; groups sorted lexically, six colours recycle.
    /// @throws std::invalid_argument Label count mismatch.
    /// @throws std::out_of_range Unavailable axis.
    /// @par Example
    /// @code
    /// const auto d=dm::datasets::iris();
    /// const dm::stats::KernelPCA m(d,{"Petal.Length","Petal.Width"});
    /// m.plot_embedding(d.get_column<std::string>("Species").values()).save_svg("kernel_groups.svg");
    /// @endcode
    [[nodiscard]] graphics::RPlot plot_embedding(const std::vector<std::string>& group_labels,
        std::size_t dimension_x=0,std::size_t dimension_y=1) const;
private:
    /// @brief Owned result of projecting one query row.
    struct Projection {
        std::vector<double> scores; ///< Retained coordinates, NaN for an invalid row.
        double norm_squared; ///< Centred feature norm, NaN for invalid row.
        double projected_squared; ///< Sum of retained squared coordinates.
        std::string status; ///< ok, missing_predictor or nonfinite_predictor.
    };
    /// @brief Validates controls, constructs kernels and diagonalizes their centred geometry.
    /// @param data Training frame.
    void fit(const dstruct::DataFrame& data);
    /// @brief Validates required numeric query columns independently of row nulls.
    /// @param data Query frame; zero-row frames need no schema.
    void validate_query(const dstruct::DataFrame& data) const;
    /// @brief Kernel with an RBF common constant removed for accurate feature centering.
    /// @param a First raw predictor vector.
    /// @param b Second raw predictor vector, same length.
    /// @return Linear/polynomial kernel or expm1(-gamma*distance_squared) for RBF.
    /// @throws std::invalid_argument Result not representable as a finite double.
    long double kernel_value(const std::vector<double>& a,const std::vector<double>& b) const;
    /// @brief Evaluates and projects a single row using stored training centering.
    /// @param data Validated query schema.
    /// @param row Original row index.
    /// @return Owned scores, norms and status.
    Projection project_row(const dstruct::DataFrame& data,std::size_t row) const;
    std::vector<std::string> feature_columns_; ///< Owned predictor names.
    KernelPCAOptions options_; ///< Owned validated controls.
    std::vector<std::size_t> kept_row_indices_; ///< Original retained training rows.
    std::vector<double> eigenvalues_; ///< Retained positive Gram eigenvalues.
    std::vector<double> all_eigenvalues_; ///< Full signed Gram spectrum.
    std::vector<std::vector<double>> training_; ///< Owned n-by-p raw finite predictors.
    std::vector<long double> feature_mean_; ///< Raw means for stable linear centering.
    std::vector<long double> kernel_mean_; ///< Per-training-row means of kernel_value().
    long double kernel_grand_mean_{0}; ///< Grand mean of kernel_value().
    linalg::DenseMatrix<double> kernel_; ///< Raw kernel matrix.
    linalg::DenseMatrix<double> centered_; ///< Numerically centred kernel.
    linalg::DenseMatrix<double> dual_; ///< Retained V/sqrt(lambda).
    linalg::DenseMatrix<double> embedding_; ///< Retained V*sqrt(lambda).
    std::size_t observations_{0}; ///< Complete training count.
    std::size_t rank_{0}; ///< Positive spectrum count above tolerance.
    double tolerance_{0}; ///< Scale-relative eigenvalue threshold.
};
} // namespace datamunge::stats
