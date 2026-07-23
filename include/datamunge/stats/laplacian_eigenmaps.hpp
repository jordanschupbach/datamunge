#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct LaplacianEigenmapsOptions {
    std::size_t n_components{2};
    std::size_t n_neighbors{10};
    // Heat kernel bandwidth (often called t or sigma^2 in the literature): edge weight between
    // connected points i,j is exp(-||x_i-x_j||^2 / heat_kernel_t).
    double heat_kernel_t{1.0};
};

/// @brief Laplacian Eigenmaps (Belkin & Niyogi, 2003): builds a SPARSE symmetric k-nearest-
///        neighbor graph with heat-kernel-weighted edges, forms the graph Laplacian L = D - W,
///        and embeds points via the n_components eigenvectors of the generalized eigenproblem
///        L f = lambda D f associated with the smallest non-trivial eigenvalues (the trivial
///        constant/all-ones-in-D-metric eigenvector at eigenvalue 0 is discarded). Solved via the
///        symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2} trick. Distinct from
///        Diffusion Maps (also heat-kernel-based, but uses a dense all-pairs kernel with
///        alpha-normalization and a diffusion-time-scaled Markov-matrix eigendecomposition
///        instead of a sparse k-NN graph Laplacian). Rows with a null value in any feature column
///        are dropped before fitting.
class LaplacianEigenmaps {
public:
    LaplacianEigenmaps(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                        LaplacianEigenmapsOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief The n_components smallest non-trivial eigenvalues of the symmetric normalized
    ///        Laplacian L_sym used to build the embedding, ascending (index 0 is the eigenvalue
    ///        closest to the discarded trivial 0 eigenvalue). Column c of embedding() corresponds
    ///        to eigenvalues()[c].
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

    std::vector<std::string>   feature_columns_;
    LaplacianEigenmapsOptions  options_;

    std::vector<std::size_t>   kept_row_indices_;
    std::vector<double>        eigenvalues_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
