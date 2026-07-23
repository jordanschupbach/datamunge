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

struct IsomapOptions {
    std::size_t n_components{2};
    std::size_t n_neighbors{10};
    DistanceMetric metric{DistanceMetric::Euclidean};
};

/// @brief Isomap (Tenenbaum, de Silva & Langford, 2000): a nonlinear manifold-learning method
///        that approximates geodesic distances along the data manifold with shortest-path
///        distances over a k-nearest-neighbor graph, then applies classical (Torgerson) MDS to
///        those geodesic distances to produce a low-dimensional embedding. Unlike MDS (which
///        embeds using straight-line distances directly), Isomap can "unroll" a curved manifold
///        (e.g. a Swiss roll) that straight-line distances would misrepresent. Rows with a null
///        value in any feature column are dropped before fitting. Throws if the resulting
///        k-nearest-neighbor graph is disconnected (increase n_neighbors in that case).
class Isomap {
public:
    Isomap(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
           IsomapOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief Every eigenvalue of the double-centered geodesic-distance inner-product matrix
    ///        (length = observations()), descending -- the first n_components() were used to
    ///        build the embedding. Negative eigenvalues are clamped to 0 when building the
    ///        embedding but reported here as-is.
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }
    /// @brief Sum of the n_components() (clamped-non-negative) eigenvalues used, divided by the
    ///        sum of all positive eigenvalues -- 1.0 means the embedding reproduces all of the
    ///        genuinely Euclidean-representable variation in the geodesic distances.
    [[nodiscard]] double goodness_of_fit() const { return goodness_of_fit_; }

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
    IsomapOptions              options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       eigenvalues_;
    double                     goodness_of_fit_{0.0};
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
