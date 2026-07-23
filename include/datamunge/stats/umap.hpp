#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/knn_classifier.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct UMAPOptions {
    std::size_t    n_components{2};
    std::size_t    n_neighbors{15};
    double         min_dist{0.1};
    std::size_t    max_iterations{500}; // "epochs" in UMAP terminology
    double         learning_rate{1.0};
    double         negative_sample_rate{5.0}; // negative samples drawn per positive edge per epoch
    DistanceMetric metric{DistanceMetric::Euclidean};
    std::uint64_t  seed{42};
};

/// @brief UMAP (Uniform Manifold Approximation and Projection): builds a fuzzy simplicial set
///        (a weighted k-nearest-neighbor graph with per-point smooth calibration and probabilistic
///        t-conorm symmetrization) from the high-dimensional data, then optimizes a low-dimensional
///        embedding via cross-entropy minimization using attractive forces on graph edges and
///        repulsive forces from negative sampling. McInnes, Healy & Melville (2018),
///        arXiv:1802.03426. Differs from MDS (no eigendecomposition of a distance matrix -- an
///        iterative force-based layout) and from t-SNE elsewhere in this module (smooth k-NN
///        calibration against a fixed log2(k) target rather than perplexity/entropy calibration,
///        fuzzy-union symmetrization rather than plain averaging, and negative-sampled
///        cross-entropy forces rather than a full KL-divergence gradient). This implementation is
///        deliberately simplified in two named respects -- see the comment in umap.cpp's fit().
///        Rows with a null value in any feature column are dropped before fitting.
class UMAP {
public:
    UMAP(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, UMAPOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief n x n_components matrix of embedded coordinates.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    /// @brief Per-point sigma_i found by the smooth k-NN binary search during fitting (length =
    ///        observations()) -- exposed mainly for diagnostics/testing of calibration convergence.
    [[nodiscard]] const std::vector<double>& sigmas() const { return sigmas_; }
    /// @brief Per-point rho_i (distance to the nearest neighbor), length = observations().
    [[nodiscard]] const std::vector<double>& rhos() const { return rhos_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] plot::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    [[nodiscard]] plot::RPlot plot_embedding(const std::vector<std::string>& group_labels, std::size_t dimension_x = 0,
                                             std::size_t dimension_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    UMAPOptions               options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       sigmas_;
    std::vector<double>       rhos_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
