#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct DiffusionMapsOptions {
    std::size_t n_components{2};
    /// @brief Heat kernel bandwidth: K(i,j) = exp(-||x_i - x_j||^2 / heat_kernel_epsilon). Should
    ///        be on the order of the typical squared pairwise distance in the fitted data.
    double heat_kernel_epsilon{1.0};
    /// @brief Density-normalization exponent in [0, 1]. 0 = no density correction (closer to a
    ///        plain graph-Laplacian-style normalization); 1 = full correction, approximating the
    ///        Laplace-Beltrami operator independent of how densely the manifold was sampled
    ///        (Coifman & Lafon's headline choice). 0.5 is a common practical default.
    double alpha{0.5};
    /// @brief Diffusion time t: embedding coordinates are scaled by eigenvalue^diffusion_time,
    ///        controlling how much large-scale (large t) vs. fine-scale (small t) structure the
    ///        embedding emphasizes.
    double diffusion_time{1.0};
};

/// @brief Diffusion Maps (Coifman & Lafon, 2006): builds a DENSE all-pairs Gaussian heat kernel
///        over the fitted rows, applies an alpha-normalization step that removes the influence of
///        sampling density before forming a row-stochastic Markov transition matrix, and embeds
///        rows using the Markov matrix's leading non-trivial eigenvectors, scaled by
///        eigenvalue^diffusion_time. Unlike Laplacian Eigenmaps (a sparse k-NN graph Laplacian
///        eigenproblem), this method uses the full dense kernel and the alpha-normalization is
///        what lets it recover a sampling-density-independent approximation of the manifold's
///        intrinsic (Laplace-Beltrami) geometry. Rows with a null value in any feature column are
///        dropped before fitting.
class DiffusionMaps {
public:
    DiffusionMaps(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                  DiffusionMapsOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief Every eigenvalue of the symmetrized Markov matrix P_sym (length = observations()),
    ///        descending. eigenvalues()[0] is the trivial (~1) eigenvalue that is discarded when
    ///        building the embedding; eigenvalues()[1 .. n_components()] are the ones actually
    ///        used (each raised to diffusion_time when scaling the embedding).
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }

    /// @brief n x n_components matrix of embedded (diffusion) coordinates.
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
    DiffusionMapsOptions       options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       eigenvalues_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
