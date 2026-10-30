#pragma once

/// @file lle.hpp
/// @brief Locally Linear Embedding (Roweis and Saul 2000): a nonlinear dimension reduction that
///        keeps each point's reconstruction from its nearest neighbours.
///
/// LLE has two steps. First, each point \f$x_i\f$ is written as the affine combination
/// \f$\sum_jw_{ij}x_j\f$ (\f$\sum_jw_{ij}=1\f$) of its \f$k\f$ nearest neighbours that best
/// reconstructs it; the weights are invariant to rotating, scaling and translating the
/// neighbourhood, so they describe its shape alone. Second, low-dimensional points
/// \f$y_i\f$ are chosen to be reconstructed by the *same* weights, by minimising
/// \f$\Phi(Y)=\sum_i\lVert y_i-\sum_jw_{ij}y_j\rVert^2=\operatorname{tr}(Y^\top MY)\f$ with
/// \f$M=(I-W)^\top(I-W)\f$, subject to \f$Y^\top Y=I\f$ and centred columns. The minimisers
/// are the eigenvectors of \f$M\f$ for its smallest eigenvalues after the constant one.
///
/// @par References
/// - S. T. Roweis and L. K. Saul (2000). "Nonlinear dimensionality reduction by locally linear
///   embedding". Science 290(5500), 2323-2326. doi:10.1126/science.290.5500.2323
/// - L. K. Saul and S. T. Roweis (2003). "Think globally, fit locally: unsupervised learning of
///   low dimensional manifolds". Journal of Machine Learning Research 4, 119-155.

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/knn_classifier.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

/// @brief Options for LLE. @ref regularization means the same as scikit-learn's
///        `LocallyLinearEmbedding(reg=...)` and has its default.
struct LLEOptions {
    std::size_t n_components{2};  ///< Embedding dimension \f$d\f$ (at least 1, less than n_neighbors).
    std::size_t n_neighbors{10};  ///< Neighbours \f$k\f$ per point (at least 2).

    /// @brief A ridge added to each local Gram matrix \f$C\f$ before solving for the weights, as a
    /// fraction of its trace: \f$C+r\operatorname{tr}(C)\,I\f$ with \f$r\f$ = regularization
    /// (scikit-learn's `reg`; Saul and Roweis 2003 write it as \f$\delta^2\operatorname{tr}(C)/k\f$,
    /// so \f$r=\delta^2/k\f$). Needed whenever \f$k\f$ exceeds the data dimension, when \f$C\f$
    /// is singular.
    double regularization{1e-3};
    DistanceMetric metric{DistanceMetric::Euclidean};  ///< Distance used to find the neighbours.
};

/// @brief Reconstruction weights of one point from its neighbours: the \f$w\f$ minimising
///        \f$\lVert x-\sum_jw_jn_j\rVert^2\f$ subject to \f$\sum_jw_j=1\f$, LLE's first step.
///
/// With the local Gram matrix \f$C_{ab}=(x-n_a)^\top(x-n_b)\f$, the Lagrangian solution is
/// \f$w=C^{-1}\mathbf 1/(\mathbf 1^\top C^{-1}\mathbf 1)\f$. \f$C\f$ has rank at most the data
/// dimension, so with more neighbours than dimensions it is singular and the ridge
/// \f$r\operatorname{tr}(C)\,I\f$ is added first; it picks, among the many exact
/// reconstructions, one with small weights. Weights can be negative (the point is outside the
/// neighbours' convex hull). If every neighbour coincides with @p point, all weights are \f$1/k\f$.
///
/// @param point  The point \f$x\f$.
/// @param neighbors  Its neighbours \f$n_1,\dots,n_k\f$, each the same length as @p point.
/// @param regularization  \f$r\f$, as LLEOptions::regularization (default 1e-3; 0 for none).
/// @return \f$k\f$ weights summing to 1.
/// @throws std::invalid_argument  If @p neighbors is empty, lengths differ, or the regularization is negative.
///
/// @par Example
/// @code
/// const auto w = dm::stats::lle_reconstruction_weights({0.0, 0.0}, {{1.0, 0.0}, {0.0, 1.0}, {-1.0, -1.0}}, 0.0);
/// std::print("{::.4f}\n", w);  // [0.3333, 0.3333, 0.3333]
/// @endcode
///
/// @par References
/// - L. K. Saul and S. T. Roweis (2003). "Think globally, fit locally". Journal of Machine
///   Learning Research 4, 119-155 (section 3.2 and appendix A).
///
/// @see LLE
std::vector<double> lle_reconstruction_weights(const std::vector<double>&              point,
                                               const std::vector<std::vector<double>>& neighbors,
                                               double                                  regularization = 1e-3);

/// @brief Locally Linear Embedding (Roweis and Saul 2000) of selected numeric columns of a
///        DataFrame.
///
/// Fitting finds each row's @ref LLEOptions::n_neighbors nearest rows (brute force,
/// \f$O(n^2p)\f$), solves for the reconstruction weights with lle_reconstruction_weights()
/// (\f$O(nk^3)\f$), forms \f$M=(I-W)^\top(I-W)\f$ and computes its full eigendecomposition
/// with linalg::symmetric_eigen() (\f$O(n^3)\f$, about one second at \f$n=1000\f$). Rows with
/// a null in any feature column are dropped first.
///
/// The embedding is \f$Y=[v_1,\dots,v_d]\f$, the eigenvectors of the \f$d\f$ smallest
/// non-trivial eigenvalues, smallest first; each column has unit length and sums to zero, as in
/// scikit-learn (Roweis and Saul scale by \f$\sqrt n\f$ instead). Each column's sign is fixed
/// so that its largest-magnitude entry is positive. A connected neighbour graph gives one
/// trivial eigenvalue 0 (the constant vector). A neighbour graph with \f$c\f$ connected
/// components makes \f$M\f$ block diagonal, with one zero eigenvalue per component, so the
/// \f$c\f$ smallest are discarded; the components are then embedded independently and each
/// remaining eigenvector lives on one component only. The trivial eigenvectors are known
/// exactly (component indicators), so they are shifted above the spectrum before solving:
/// an embedding eigenvalue close to zero then cannot mix with them, and every column sums to
/// zero on each component to rounding error (scikit-learn instead drops the first eigenvector
/// as returned, which can be a rotation of the two). connected_components() reports \f$c\f$:
/// when it exceeds 1, increase n_neighbors or embed the components separately.
///
/// @par Example
/// @code
/// const auto roll = dm::datasets::swiss_roll(400, 0.0, 1);
/// const dm::stats::LLE lle(roll, {"x", "y", "z"}, {.n_components = 2, .n_neighbors = 12});
/// std::print("{} {}\n", lle.observations(), lle.connected_components());  // 400 1
/// @endcode
///
/// @par References
/// - S. T. Roweis and L. K. Saul (2000). "Nonlinear dimensionality reduction by locally linear
///   embedding". Science 290(5500), 2323-2326. doi:10.1126/science.290.5500.2323
/// - L. K. Saul and S. T. Roweis (2003). "Think globally, fit locally: unsupervised learning of
///   low dimensional manifolds". Journal of Machine Learning Research 4, 119-155.
/// - scikit-learn `LocallyLinearEmbedding`:
///   https://scikit-learn.org/stable/modules/generated/sklearn.manifold.LocallyLinearEmbedding.html
///
/// @see lle_reconstruction_weights(), Isomap, PCA, trustworthiness()
class LLE {
public:
    /// @brief Fits LLE to @p feature_columns of @p data.
    /// @param data  Data frame.
    /// @param feature_columns  Numeric columns spanning the input space.
    /// @param options  Embedding dimension, neighbours, regularization and metric.
    /// @throws std::invalid_argument  If no columns are given, a column is missing or not numeric,
    ///         n_components is 0 or not below n_neighbors, n_neighbors is below 2, the
    ///         regularization is negative, or fewer than n_neighbors + 1 complete rows remain.
    ///
    /// @par Example
    /// See LLE.
    LLE(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, LLEOptions options = {});

    /// @brief The feature columns used.
    /// @return Column names, in the order given.
    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    /// @brief Number of fitted rows \f$n\f$.
    /// @return \f$n\f$.
    [[nodiscard]] std::size_t observations() const { return observations_; }
    /// @brief Embedding dimension \f$d\f$.
    /// @return \f$d\f$.
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Data-frame row (0-based) of each fitted row; rows with nulls are absent. Use it to
    ///        align labels or a withheld response with embedding().
    /// @return One row index per fitted row, increasing.
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief The \f$k\f$ nearest neighbours of each fitted row, as fitted-row indices, nearest first.
    /// @return \f$n\f$ lists of \f$k\f$ indices.
    [[nodiscard]] const std::vector<std::vector<std::size_t>>& neighbors() const { return neighbors_; }
    /// @brief Reconstruction weights, parallel to neighbors(); each list sums to 1.
    /// @return \f$n\f$ lists of \f$k\f$ weights.
    [[nodiscard]] const std::vector<std::vector<double>>& weights() const { return weights_; }
    /// @brief Squared reconstruction error \f$\lVert x_i-\sum_jw_{ij}x_j\rVert^2\f$ of each fitted
    ///        row in the input space. Large values flag points whose neighbourhood is not locally flat.
    /// @return One error per fitted row.
    [[nodiscard]] const std::vector<double>& reconstruction_errors() const { return reconstruction_errors_; }

    /// @brief Number of connected components of the (symmetrised) \f$k\f$-nearest-neighbour graph,
    ///        and so of zero eigenvalues of \f$M\f$ discarded as trivial.
    /// @return 1 for a connected graph.
    [[nodiscard]] std::size_t connected_components() const { return component_count_; }
    /// @brief Component of each fitted row, numbered 0, 1, ... in order of first appearance.
    /// @return One label per fitted row.
    [[nodiscard]] const std::vector<int>& component_labels() const { return component_labels_; }

    /// @brief All \f$n\f$ eigenvalues of \f$M=(I-W)^\top(I-W)\f$, in decreasing order.
    ///
    /// The last connected_components() are the trivial zeros; the \f$d\f$ before them, read
    /// backwards, are the embedding's costs \f$\Phi(v_1)\le\dots\le\Phi(v_d)\f$.
    /// @return The eigenvalues.
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }

    /// @brief Embedded coordinates, \f$n\times d\f$; column \f$c\f$ is a unit-length, zero-sum
    ///        eigenvector of \f$M\f$.
    /// @return The embedding.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }

    /// @brief Embedding cost \f$\Phi(v)=\sum_i\big(v_i-\sum_jw_{ij}v_j\big)^2=v^\top Mv\f$ of a
    ///        candidate coordinate, after centring it and scaling it to unit length.
    ///
    /// On the same scale as eigenvalues(): every embedding column \f$c\f$ has cost equal to its
    /// eigenvalue, and no unit, zero-sum vector costs less than the smallest non-trivial one.
    /// Comparing the cost of a known coordinate (the arc length of a Swiss roll, say) with that of
    /// the raw input columns shows which of them the embedding will prefer.
    /// @param values  One value per fitted row (align with kept_row_indices()).
    /// @return \f$\Phi\f$ of the centred, unit-length @p values.
    /// @throws std::invalid_argument  If the length is not observations() or @p values is constant.
    ///
    /// @par Example
    /// @code
    /// const auto roll = dm::datasets::swiss_roll(1000, 0.0, 1);
    /// const dm::stats::LLE lle(roll, {"x", "y", "z"}, {.n_neighbors = 12});
    /// std::print("{:.2e}\n", lle.cost(roll.numeric_column("arc_length").values()));  // 1.18e-05
    /// @endcode
    [[nodiscard]] double cost(const std::vector<double>& values) const;

    /// @brief One column of embedding().
    /// @param index  Dimension, 0-based.
    /// @return \f$n\f$ coordinates.
    /// @throws std::out_of_range  If @p index is not below n_components().
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    /// @brief Text summary: features, options, the embedding eigenvalues and the number of
    ///        neighbour-graph components.
    /// @return The summary.
    ///
    /// @par Example
    /// @code
    /// const dm::stats::LLE lle(dm::datasets::iris(), {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"});
    /// std::print("{}", lle.summary());  // Locally Linear Embedding ... components: 2 ...
    /// @endcode
    [[nodiscard]] std::string summary() const;
    /// @brief Writes summary() to @p os.
    /// @param os  Output stream.
    void print_summary(std::ostream& os) const;
    /// @brief Writes summary() to standard output.
    void print_summary() const;

    /// @brief Scatter plot of two embedding dimensions.
    /// @param dimension_x  Dimension on the x axis (default 0).
    /// @param dimension_y  Dimension on the y axis (default 1).
    /// @return The plot.
    /// @throws std::out_of_range  If a dimension is not below n_components().
    ///
    /// @par Example
    /// @code
    /// const auto roll = dm::datasets::swiss_roll(300, 0.0, 1);
    /// dm::stats::LLE(roll, {"x", "y", "z"}, {.n_neighbors = 12}).plot_embedding().save("lle.svg");
    /// @endcode
    [[nodiscard]] graphics::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    /// @brief Scatter plot of two embedding dimensions, one colour per group.
    /// @param group_labels  One label per fitted row (align with kept_row_indices()).
    /// @param dimension_x  Dimension on the x axis (default 0).
    /// @param dimension_y  Dimension on the y axis (default 1).
    /// @return The plot, with a legend of the sorted group labels.
    /// @throws std::out_of_range  If a dimension is not below n_components().
    /// @throws std::invalid_argument  If @p group_labels does not have one entry per fitted row.
    ///
    /// @par Example
    /// See plot_embedding(std::size_t, std::size_t) const.
    [[nodiscard]] graphics::RPlot plot_embedding(const std::vector<std::string>& group_labels,
                                                 std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;

private:
    /// @brief Runs both LLE steps on @p data.
    /// @param data  The data frame given to the constructor.
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_; ///< Feature column names.
    LLEOptions               options_;         ///< Options given to the constructor.

    std::vector<std::size_t>              kept_row_indices_;      ///< Data row of each fitted row.
    std::vector<std::vector<std::size_t>> neighbors_;             ///< k nearest neighbours of each row.
    std::vector<std::vector<double>>      weights_;               ///< Reconstruction weights.
    std::vector<double>                   reconstruction_errors_; ///< Squared reconstruction errors.
    std::vector<int>                      component_labels_;      ///< Neighbour-graph component of each row.
    std::vector<double>                   eigenvalues_;           ///< Eigenvalues of M, decreasing.
    linalg::DenseMatrix<double>           embedding_;             ///< n x d embedding.
    std::size_t                           observations_{0};       ///< Fitted rows.
    std::size_t                           component_count_{1};    ///< Connected components (trivial eigenvalues discarded).
};

} // namespace datamunge::stats
