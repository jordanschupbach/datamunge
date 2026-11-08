#pragma once
/// @file diffusion_maps.hpp
/// @brief Dense Gaussian diffusion coordinates and owned actual normalization traces.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Dense Gaussian diffusion-map controls, with raw feature units and no automatic scaling.
/// Defaults retain two coordinates, epsilon one, alpha one half and time one.
/// Validated by DiffusionMaps and diffusion_maps_trace().
/// @par Example
/// @code
/// dm::stats::DiffusionMapsOptions o;
/// std::print("{} {:.1f}\n",o.n_components,o.alpha); // 2 0.5
/// @endcode
/// @see DiffusionMaps, diffusion_maps_trace()
struct DiffusionMapsOptions {
    std::size_t n_components{2}; ///< Retained nontrivial modes; positive and strictly below n-1.
    double heat_kernel_epsilon{1.0}; ///< Finite positive denominator of exp(-squared distance/epsilon).
    double alpha{0.5}; ///< Finite exponent in [0,1] of the kernel degree correction.
    double diffusion_time{1.0}; ///< Finite nonnegative spectral time; integer times correspond to Markov steps.
};

/// @brief Value-owned matrices from the actual dense diffusion fit, without replay.
/// All square matrices are n by n in kept-row order. Right eigenvectors have
/// D-weighted norm one, not stationary-probability norm one. Multiply coordinates
/// by sqrt(sum(degrees)) to obtain the latter convention. No input references survive.
/// @par Example
/// See diffusion_maps_trace().
/// @par References
/// - Coifman and Lafon (2006), Diffusion maps. doi:10.1016/j.acha.2006.04.006
/// @see DiffusionMaps
struct DiffusionMapsTrace {
    std::vector<std::size_t> kept_row_indices; ///< Original complete-case row indices.
    linalg::DenseMatrix<double> squared_distances; ///< Raw squared Euclidean distances.
    linalg::DenseMatrix<double> kernel; ///< Gaussian K, including unit diagonal.
    std::vector<double> kernel_degrees; ///< q_i=sum_j K_ij.
    linalg::DenseMatrix<double> normalized_kernel; ///< A_ij=K_ij/(q_i^alpha q_j^alpha).
    std::vector<double> degrees; ///< d_i=sum_j A_ij; all strictly positive.
    std::vector<double> stationary; ///< pi_i=d_i/sum_j d_j.
    linalg::DenseMatrix<double> transition; ///< Row-stochastic P=D^-1 A.
    linalg::DenseMatrix<double> symmetric; ///< S=D^-1/2 A D^-1/2.
    std::vector<double> eigenvalues; ///< Descending eigenvalues, roundoff negatives clipped to zero.
    linalg::DenseMatrix<double> right_eigenvectors; ///< Columns psi=D^-1/2 v, including trivial mode.
    linalg::DenseMatrix<double> embedding; ///< Retained lambda^time psi columns, excluding mode zero.
};

/// @brief Fits dense diffusion coordinates and retains every normalization matrix from that execution.
/// Shares the model kernel. Drops rows with a null feature; rejects nonfinite retained
/// values. Uses raw Euclidean features, no centring or standardization. Time zero uses
/// scale one even for zero eigenvalues. Roundoff-negative eigenvalues within 128*n*eps
/// are clipped to zero; a more negative spectrum is rejected. Numerically disconnected
/// kernels are rejected because discarding a single trivial mode would be ambiguous.
/// Work O(n*n*p+n^3), storage O(n*n+n*p), including six owned square trace matrices.
/// Inputs are borrowed unchanged; calls have no shared mutable state.
/// @param data Input frame; complete-case count n must be at least three.
/// @param feature_columns Nonempty list of existing numeric columns, in metric order.
/// @param options Fit controls, default DiffusionMapsOptions{}; 0<components<n-1.
/// @param max_observations Complete-case row cap, default 1000; zero disables only this cap.
/// @return Owned trace; eigenvector signs and repeated-eigenspace bases are arbitrary.
/// @throws std::invalid_argument Invalid schema, counts, options or nonfinite retained features.
/// @throws std::length_error Row cap or square-matrix capacity exceeded before dense allocation.
/// @throws std::overflow_error Squared distance cannot be represented as finite double.
/// @throws std::runtime_error Disconnected floating-point kernel or materially negative spectrum.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2});
/// const auto t=dm::stats::diffusion_maps_trace(d,{"x"},{1,1,0,1});
/// std::print("{} {:.0f} {:.0f}\n",t.kept_row_indices.size(),t.kernel(0,0),t.squared_distances(0,2)); // 3 1 4
/// @endcode
/// @par References
/// - Coifman and Lafon (2006), Diffusion maps. Applied and Computational Harmonic Analysis
///   21(1), 5-30. https://doi.org/10.1016/j.acha.2006.04.006
/// @see DiffusionMaps, DiffusionMapsTrace
[[nodiscard]] DiffusionMapsTrace diffusion_maps_trace(
    const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
    DiffusionMapsOptions options = {}, std::size_t max_observations = 1000);

/// @brief Dense Gaussian diffusion maps with alpha degree correction and right Markov eigenvectors.
/// Coordinates are lambda^time D^-1/2 v, where v has Euclidean norm one.
/// Degree weighting, feature units, null handling, time and eigenvalue conventions follow
/// diffusion_maps_trace(). Density independence at alpha one is an asymptotic manifold
/// result under regularity and sampling conditions, not an exact finite-data guarantee.
/// Fit work O(n*n*p+n^3), temporary storage O(n*n+n*p), retained storage O(n*k+n+p).
/// The object owns its results; concurrent const reads are safe. No prediction on new rows.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2});
/// dm::stats::DiffusionMaps m(d,{"x"},{1,1,0,1});
/// std::print("{} {}\n",m.observations(),m.n_components()); // 3 1
/// @endcode
/// @par References
/// - Coifman and Lafon (2006), Diffusion maps. doi:10.1016/j.acha.2006.04.006
/// @see diffusion_maps_trace()
class DiffusionMaps {
public:
    /// @brief Fits the complete-case rows, owning results without changing the input.
    /// @param data Borrowed input frame with at least three complete observations.
    /// @param feature_columns Existing numeric feature names, nonempty.
    /// @param options Controls, default {}; 0<components<n-1.
    /// @throws std::invalid_argument Invalid schema, controls or nonfinite retained values.
    /// @throws std::length_error Dense matrix capacity exceeded.
    /// @throws std::overflow_error Nonfinite squared distance.
    /// @throws std::runtime_error Numerically disconnected kernel or invalid spectrum.
    /// @par Example
    /// See DiffusionMaps.
    /// @see diffusion_maps_trace()
    DiffusionMaps(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                  DiffusionMapsOptions options = {});

    /// @brief Feature names in metric order.
    /// @return Borrowed names, valid for this object lifetime.
    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    /// @brief Number of retained complete rows.
    /// @return Fitted row count.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Number of retained nontrivial coordinates.
    /// @return Embedding column count.
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    /// @return Borrowed indices, valid for the object lifetime.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief Every eigenvalue of the symmetrized Markov matrix P_sym (length = observations()),
    ///        descending. eigenvalues()[0] is the trivial (~1) eigenvalue that is discarded when
    ///        building the embedding; eigenvalues()[1 .. n_components()] are the ones actually
    ///        used (each raised to diffusion_time when scaling the embedding).
    /// @return Borrowed spectrum; tiny roundoff-negative values are clipped to zero.
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }

    /// @brief n x n_components matrix of embedded (diffusion) coordinates.
    /// @return Borrowed fitted coordinate matrix; lifetime matches this object.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    /// @brief Copies one fitted coordinate.
    /// @param index Zero-based retained mode index.
    /// @return Owned vector in fitted-row order.
    /// @throws std::out_of_range Index is not below n_components().
    /// @par Example
    /// See DiffusionMaps; m.dimension(0).size() equals m.observations().
    /// @see embedding()
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    /// @brief Formats the fitted options, shape and leading eigenvalues.
    /// @return Owned descriptive string with a trailing newline.
    /// @par Example
    /// See print_summary(std::ostream&) const.
    /// @see print_summary()
    [[nodiscard]] std::string summary() const;
    /// @brief Writes the model summary using fixed precision four.
    /// Leaves the stream precision/float flags changed.
    /// @param os Output stream, borrowed; stream exceptions propagate.
    /// @par Example
    /// @code
    /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2});
    /// dm::stats::DiffusionMaps m(d,{"x"},{1,1,0,1});
    /// std::ostringstream out;m.print_summary(out);
    /// std::print("{}\n",out.str()==m.summary()); // true
    /// @endcode
    /// @see summary()
    void                      print_summary(std::ostream& os) const;
    /// @brief Writes the summary to std::cout.
    /// @par Example
    /// See print_summary(std::ostream&) const; this overload chooses std::cout.
    /// @see summary()
    void                      print_summary() const;

    /// @brief Creates an ungrouped scatter of two fitted coordinates, without file I/O.
    /// @param dimension_x Zero-based horizontal mode, default zero.
    /// @param dimension_y Zero-based vertical mode, default one.
    /// @return Owned plot, independent of model lifetime.
    /// @throws std::out_of_range A dimension is not below n_components().
    /// @par Example
    /// @code
    /// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,1,2,3});
    /// dm::stats::DiffusionMaps m(d,{"x"});
    /// const auto p=m.plot_embedding();std::print("plot ready\n"); // plot ready
    /// @endcode
    /// @see dimension()
    [[nodiscard]] graphics::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    /// @brief Creates a scatter grouped by aligned external labels; colours cycle after six groups.
    /// @param group_labels One string per fitted row, not per original row when nulls were dropped.
    /// @param dimension_x Horizontal mode, default zero.
    /// @param dimension_y Vertical mode, default one.
    /// @return Owned plot; labels are used only for presentation.
    /// @throws std::out_of_range Invalid dimension.
    /// @throws std::invalid_argument Label count differs from observations().
    /// @par Example
    /// See the ungrouped overload; m.plot_embedding({"a","a","b","b"}) groups those four rows.
    /// @see kept_row_indices()
    [[nodiscard]] graphics::RPlot plot_embedding(const std::vector<std::string>& group_labels,
                                             std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;

private:
    /// @brief Runs the shared fit kernel and retains only compact model results.
    /// @param data Borrowed input frame.
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_; ///< Owned feature names.
    DiffusionMapsOptions       options_; ///< Validated fit controls.

    std::vector<std::size_t> kept_row_indices_; ///< Owned retained row mapping.
    std::vector<double>       eigenvalues_; ///< Owned clipped full spectrum.
    linalg::DenseMatrix<double> embedding_; ///< Owned retained coordinates.
    std::size_t                 observations_{0}; ///< Complete-case count.
};

} // namespace datamunge::stats
