#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/knn_classifier.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct LLEOptions {
    std::size_t n_components{2};
    std::size_t n_neighbors{10};
    /// @brief Diagonal regularization added to the local Gram matrix before solving for
    ///        reconstruction weights, as a FRACTION of the Gram matrix's trace (standard LLE
    ///        numerical-stability trick, needed whenever n_neighbors exceeds the local intrinsic
    ///        dimensionality) -- not an absolute value.
    double regularization{1e-3};
    DistanceMetric metric{DistanceMetric::Euclidean};
};

/// @brief Locally Linear Embedding (Roweis & Saul, 2000): reconstructs each point as a locally
///        optimal linear combination of its k nearest neighbors, then finds a low-dimensional
///        embedding that best preserves those local reconstruction weights. Unlike PCA (global
///        linear variance) or classical MDS (global pairwise distances), LLE only preserves LOCAL
///        neighborhood geometry, which lets it unfold nonlinear manifolds that PCA/MDS cannot.
///        Rows with a null value in any feature column are dropped before fitting.
class LLE {
public:
    LLE(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, LLEOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief All n eigenvalues of M = (I-W)^T(I-W), descending. The trivial (smallest, ~0)
    ///        eigenvalue is discarded when building the embedding but reported here as-is.
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }

    /// @brief n x n_components matrix of embedded coordinates.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] plot::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    [[nodiscard]] plot::RPlot plot_embedding(const std::vector<std::string>& group_labels,
                                             std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    LLEOptions                  options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       eigenvalues_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
    /// @brief Number of trailing (near-zero) eigenvalues discarded as trivial before selecting
    ///        the n_components() embedding eigenvectors -- normally 1, but equals the number of
    ///        connected components of the k-nearest-neighbor graph when it is disconnected (see
    ///        fit() for why this matters). Used only internally by print_summary().
    std::size_t discarded_trivial_count_{1};
};

} // namespace datamunge::stats
