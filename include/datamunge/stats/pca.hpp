#pragma once
/// @file pca.hpp
/// @brief Inspectable covariance/correlation PCA, frozen projections and reconstruction error.
#include <cstddef>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <iosfwd>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief Preprocessing for PCA; scaling defaults true, unlike R prcomp's false.
/// With center=true, scale=true uses sample standard deviations (n-1 divisor).
/// With center=false it uses sqrt(sum(x^2)/(n-1)), not centred standard deviation.
/// Exactly zero scales become one. With centring, constant columns remain null directions.
/// @par Example
/// @code
/// const dm::stats::PCAOptions o{true,false};
/// std::print("{} {}\n",o.center,o.scale); // true false
/// @endcode
/// @par References
/// - R prcomp: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html
/// @see PCA
struct PCAOptions {
  bool center{true}; ///< Subtract training feature means; false uses a zero origin.
  bool scale{true};  ///< Divide by training sample RMS about that origin; zero RMS becomes one.
};
/// @brief PCA from a named numeric DataFrame schema, retaining a full p-axis basis.
/// Like R prcomp in centred/scaled score semantics, but uses a covariance eigensolver,
/// defaults scale=true, permits constant columns and always returns p components,
/// including null directions when p>=n. Rank is separate from num_components().
/// Fitting drops rows with any null selected feature and rejects retained nonfinite
/// values. Training means/scales and axes are frozen for new rows. With centring,
/// eigenvalues are sample variances; without it they are second-moment energies.
/// Signs are arbitrary, and repeated eigenvalues permit rotations of their eigenspace.
/// Gram formation squares the data condition number; this is not a high-relative-accuracy
/// SVD for near-null directions. Work O(np^2+p^3) for bounded Jacobi sweeps, storage
/// O(np+p^2). Model owns all state; const operations have no shared mutable caches.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
/// const dm::stats::PCA fit(d,{"x"},dm::stats::PCAOptions{true,false});
/// std::print("{:.0f}\n",fit.explained_variance()[0]); // 1
/// @endcode
/// @par References
/// - Pearson (1901), On lines and planes of closest fit to systems of points in space.
///   https://doi.org/10.1080/14786440109462720
/// - R prcomp: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html
/// @see reconstruct(), reconstruction_curve(), spectrum()
class PCA {
 public:
  /// @brief Fits all p principal axes to complete rows of selected numeric features.
  /// @param data Borrowed frame; selected-feature nulls drop rows, including their other values.
  /// @param feature_columns Nonempty ordered numeric schema, with no duplicate names.
  /// @param options Center/scale switches, both true by default; see PCAOptions.
  /// @throws std::invalid_argument Bad schema, fewer than two complete rows, retained nonfinite
  /// values, or unrepresentable preprocessing, covariance, eigenvalues or scores.
  /// @par Example
  /// See the class example.
  /// @par References
  /// - Pearson (1901). https://doi.org/10.1080/14786440109462720
  PCA(const dstruct::DataFrame&       data,
      const std::vector<std::string>& feature_columns,
      PCAOptions                      options = {});
  /// @brief Selected numeric schema in fitted feature order.
  /// @return Borrowed names, valid for the model's lifetime.
  [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
  /// @brief Complete training-row count after null dropping.
  /// @return n>=2.
  [[nodiscard]] std::size_t observations() const { return observations_; }
  /// @brief Number of returned axes, including zero-variance directions.
  /// @return p, the selected feature count; not numerical_rank().
  [[nodiscard]] std::size_t num_components() const { return loadings_.cols(); }
  /// @brief Maps fitted score rows back to the constructor frame.
  /// @return Borrowed original zero-based row indices in fitted order.
  [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const {
    return kept_row_indices_;
  }
  /// @brief Training offsets used in preprocessing and reconstruction.
  /// @return Borrowed p-vector of means if centered, otherwise zeros.
  [[nodiscard]] const std::vector<double>& means() const { return mean_; }
  /// @brief Training divisors used in preprocessing and reconstruction.
  /// @return Borrowed p-vector, all positive finite; ones if scaling is off or a column has zero
  /// RMS.
  [[nodiscard]] const std::vector<double>& scales() const { return scale_; }
  /// @brief The actual preprocessed Gram matrix divided by n-1.
  /// @return Borrowed p-by-p covariance/correlation matrix when centred; second moment otherwise.
  [[nodiscard]] const linalg::DenseMatrix<double>& covariance_matrix() const { return covariance_; }
  /// @brief Eigenvalues exceeding 64*max(n,p)*epsilon*largest eigenvalue.
  /// @return Scale-relative numerical Gram rank, zero for an all-zero matrix; not an SVD rank
  /// threshold.
  [[nodiscard]] std::size_t numerical_rank() const { return rank_; }
  /// @brief Descending Gram eigenvalues, with negative roundoff clipped to zero.
  /// @return Borrowed p-vector; sample variances only when center=true.
  [[nodiscard]] const std::vector<double>& explained_variance() const {
    return explained_variance_;
  }
  /// @brief Per-axis shares of total preprocessed training energy.
  /// @return Owned p-vector summing approximately to one, or all zeros for zero total energy.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{:.0f}\n",m.explained_variance_ratio()[0]); // 1
  /// @endcode
  /// @par References
  /// - R prcomp: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html
  [[nodiscard]] std::vector<double> explained_variance_ratio() const;
  /// @brief Cumulative shares in descending component order.
  /// @return Owned p-vector, all zeros for zero energy.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{:.0f}\n",m.cumulative_explained_variance_ratio()[0]); // 1
  /// @endcode
  /// @see explained_variance_ratio()
  [[nodiscard]] std::vector<double> cumulative_explained_variance_ratio() const;
  /// @brief Orthonormal principal axes, also called rotation loadings rather than variable
  /// correlations.
  /// @return Borrowed p-by-p matrix with axes in columns, expressed in the preprocessed
  /// coordinates.
  [[nodiscard]] const linalg::DenseMatrix<double>& loadings() const { return loadings_; }
  /// @brief Complete training rows expressed in the principal-axis basis.
  /// @return Borrowed n-by-p score matrix; with centring, T^T*T/(n-1) is diagonal up to roundoff.
  [[nodiscard]] const linalg::DenseMatrix<double>& scores() const { return scores_; }
  /// @brief Copies one principal-axis loading vector.
  /// @param component_index Zero-based component index in [0,p).
  /// @return Owned length-p column.
  /// @throws std::out_of_range Invalid index.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{:.0f}\n",std::abs(m.component_loadings(0)[0])); // 1
  /// @endcode
  [[nodiscard]] std::vector<double> component_loadings(std::size_t component_index) const;
  /// @brief Copies the training scores for one axis.
  /// @param component_index Zero-based component index in [0,p).
  /// @return Owned length-n column, aligned with kept_row_indices().
  /// @throws std::out_of_range Invalid index.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{}\n",m.component_scores(0).size()); // 3
  /// @endcode
  [[nodiscard]] std::vector<double> component_scores(std::size_t component_index) const;
  /// @brief Spectrum and remaining training error after retaining each component prefix.
  /// @return Owned p-row component/variance/share/cumulative_share/tail_sse table. Tail SSE
  /// equals (n-1) times the remaining eigenvalue sum, in preprocessed units.
  /// @throws std::invalid_argument An output energy is not representable.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});std::print("{}\n",m.spectrum().nrows()); // 1
  /// @endcode
  /// @par References
  /// - Eckart and Young (1936). https://doi.org/10.1007/BF02288367
  /// @see reconstruction_curve()
  [[nodiscard]] dstruct::DataFrame spectrum() const;
  /// @brief Formats the principal-axis matrix in a feature-labelled table.
  /// @return Owned frame with feature and numeric PC1..PCp columns; no sign reinterpretation.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{}\n",m.loadings_frame().nrows()); // 1
  /// @endcode
  /// @see loadings()
  [[nodiscard]] dstruct::DataFrame loadings_frame() const;
  /// @brief Projects query rows using the frozen training offsets, divisors and axes.
  /// No row dropping, refit, whitening or centring on the query batch. Work O(mp^2).
  /// @param newdata Frame containing all selected numeric features; extra columns ignored.
  /// Every query coordinate must be nonnull/finite. Zero rows are allowed.
  /// @return Owned m-by-p scores in query row order, even for m=1.
  /// @throws std::invalid_argument Bad schema, missing/nonfinite values or unrepresentable
  /// arithmetic.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"},dm::stats::PCAOptions{true,false});
  /// dm::dstruct::DataFrame q;q.add_column("x",std::vector<double>{2});
  /// std::print("{:.0f}\n",std::abs(m.transform(q)(0,0))); // 2
  /// @endcode
  /// @par References
  /// - R predict.prcomp: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html
  [[nodiscard]] linalg::DenseMatrix<double> transform(const dstruct::DataFrame& newdata) const;
  /// @brief Projects onto a leading-axis prefix and returns coordinates in original feature units.
  /// Computes mean + scale * ((preprocess(newdata)*V_k)*V_k^T). The rank-k approximation
  /// minimizes preprocessed squared error, not generally original-unit error when scaled.
  /// k=0 returns the fitted origin; all p axes reconstruct to floating-point precision.
  /// @param newdata Query frame with the same requirements as transform().
  /// @param components Number of leading axes in [0,p].
  /// @return Owned m-by-p matrix in feature_names() order and original feature units.
  /// @throws std::invalid_argument Bad k/schema, missing/nonfinite values or unrepresentable
  /// arithmetic.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{1,2,3});
  /// const dm::stats::PCA m(d,{"x"});std::print("{:.0f}\n",m.reconstruct(d,0)(0,0)); // 2
  /// @endcode
  /// @par References
  /// - Eckart and Young (1936). https://doi.org/10.1007/BF02288367
  /// @see reconstruction_curve()
  [[nodiscard]] linalg::DenseMatrix<double> reconstruct(
      const dstruct::DataFrame& newdata,
      std::size_t               components) const;
  /// @brief Measures query reconstruction error for every leading prefix, including zero axes.
  /// Uses the actual preprocessed query entries, not a full-score reconstruction as the
  /// reference. Fitting state is unchanged. Error decreases in preprocessed norm; original-
  /// unit error need not decrease at every prefix when scaling is active. Work O(mp^2),
  /// storage O(mp+p). Input labels are not inspected and this is not a calibrated anomaly score.
  /// @param newdata Nonempty query frame accepted by transform(); use complete fitted rows
  /// for the training eigenvalue-tail identity. Independent query rows need not satisfy that
  /// identity.
  /// @return Owned p+1-row components/preprocessed_sse/preprocessed_rmse/original_rmse table,
  /// RMS over m*p entries; units of the last column mix supplied feature units if they differ.
  /// @throws std::invalid_argument Empty query, invalid transform input or unrepresentable errors.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});std::print("{}\n",m.reconstruction_curve(d).nrows()); // 2
  /// @endcode
  /// @par References
  /// - Eckart and Young (1936). https://doi.org/10.1007/BF02288367
  [[nodiscard]] dstruct::DataFrame reconstruction_curve(const dstruct::DataFrame& newdata) const;
  /// @brief Formats the feature schema, options, dimensions and spectrum as a text summary.
  /// @return Owned text; does not mutate the model.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{}\n",!m.summary().empty()); // true
  /// @endcode
  [[nodiscard]] std::string summary() const;
  /// @brief Writes summary()'s content to a supplied stream, setting its numeric formatting.
  /// @param os Borrowed output stream; standard stream error handling applies.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::ostringstream text;m.print_summary(text);
  /// std::print("{}\n",!text.str().empty()); // true
  /// @endcode
  void print_summary(std::ostream& os) const;
  /// @brief Writes a summary to standard output.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// m.print_summary(); // prints the fitted one-axis summary
  /// @endcode
  void print_summary() const;
  /// @brief Draws training scores on two selected components, without grouping.
  /// @param component_x Zero-based horizontal component, default zero.
  /// @param component_y Zero-based vertical component, default one; p must exceed it.
  /// @return Owned plot; no file I/O.
  /// @throws std::out_of_range Either component index is outside [0,p).
  /// @par Example
  /// @code
  /// const dm::stats::PCA m(dm::datasets::iris(),{"Petal.Length","Petal.Width"});
  /// const auto p=m.plot_scores();std::print("{}\n",p.series().size()); // 1
  /// @endcode
  /// @see plot_scores(const std::vector<std::string>&,std::size_t,std::size_t)
  [[nodiscard]] graphics::RPlot plot_scores(
      std::size_t component_x = 0,
      std::size_t component_y = 1) const;
  /// @brief Draws scores coloured by external labels, without using those labels in fitting.
  /// @param group_labels One class/string per complete fitted row, aligned using
  /// kept_row_indices().
  /// @param component_x Horizontal component, default zero.
  /// @param component_y Vertical component, default one.
  /// @return Owned plot with recycled colours in lexicographic group order; no file I/O.
  /// @throws std::invalid_argument Group count differs from observations().
  /// @throws std::out_of_range Bad component index.
  /// @par Example
  /// @code
  /// const auto d=dm::datasets::iris();const dm::stats::PCA m(d,{"Petal.Length","Petal.Width"});
  /// std::vector<std::string> labels;for(auto
  /// i:m.kept_row_indices())labels.push_back(d.string_at("Species",i));
  /// std::print("{}\n",m.plot_scores(labels).series().size()); // 3
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_scores(
      const std::vector<std::string>& group_labels,
      std::size_t                     component_x = 0,
      std::size_t                     component_y = 1) const;
  /// @brief Plots every component's percentage of preprocessed training energy.
  /// @return Owned scree bar plot, including zero-energy directions; no file I/O.
  /// @par Example
  /// @code
  /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{-1,0,1});
  /// const dm::stats::PCA m(d,{"x"});
  /// std::print("{}\n",m.plot_scree().series().size()); // 1
  /// @endcode
  [[nodiscard]] graphics::RPlot plot_scree() const;

 private:
  /// @brief Validates/retains complete fit rows, freezes preprocessing and computes the Gram
  /// spectrum.
  /// @param data Borrowed constructor frame.
  /// @throws std::invalid_argument As the constructor.
  void fit(const dstruct::DataFrame& data);
  /// @brief Validates query schema and applies fitted preprocessing without dropping rows.
  /// @param data Borrowed query frame.
  /// @return Owned m-by-p finite preprocessed entries.
  /// @throws std::invalid_argument Invalid input or unrepresentable preprocessing.
  linalg::DenseMatrix<double> preprocess(const dstruct::DataFrame& data) const;
  /// @brief Multiplies preprocessed coordinates by the fitted axis matrix with checked
  /// accumulation.
  /// @param prepared Borrowed finite m-by-p matrix, produced internally.
  /// @return Owned finite m-by-p scores.
  /// @throws std::invalid_argument Unrepresentable score arithmetic.
  linalg::DenseMatrix<double> project(const linalg::DenseMatrix<double>& prepared) const;
  std::vector<std::string>    feature_columns_; ///< Ordered numeric feature schema.
  PCAOptions                  options_;         ///< Frozen preprocessing switches.
  std::vector<std::size_t> kept_row_indices_; ///< Complete fit-row mapping into the original frame.
  std::vector<double>      mean_;             ///< Training offsets, zero when centring is disabled.
  std::vector<double>      scale_; ///< Positive training divisors, one for disabled/zero scaling.
  std::vector<double>      explained_variance_; ///< Descending nonnegative Gram eigenvalues.
  linalg::DenseMatrix<double> covariance_;      ///< Actual preprocessed Gram matrix divided by n-1.
  linalg::DenseMatrix<double> loadings_;        ///< Full p-by-p orthonormal axis basis.
  linalg::DenseMatrix<double> scores_;          ///< Complete training rows in axis coordinates.
  std::size_t                 observations_{0}; ///< Number of complete training rows.
  std::size_t                 rank_{0}; ///< Gram rank at the documented scale-relative threshold.
};
} // namespace datamunge::stats
