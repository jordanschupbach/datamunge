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

struct MDSOptions {
    std::size_t n_components{2};
    DistanceMetric metric{DistanceMetric::Euclidean};
};

/// @brief Classical (metric/Torgerson) multidimensional scaling: computes a pairwise distance
///        matrix from the given numeric feature columns, double-centers it, and embeds the rows
///        in n_components dimensions via eigendecomposition of the double-centered matrix -- the
///        coordinates that best reproduce the original pairwise distances in a low-dimensional
///        Euclidean space (in the least-squares sense, for the leading eigenvalues used). Rows
///        with a null value in any feature column are dropped before fitting.
class MDS {
public:
    MDS(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, MDSOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_components() const { return embedding_.cols(); }

    /// @brief Row indices (into the original `data` passed to the constructor) that survived
    ///        null-dropping, in fitted order -- use this to align an externally-held label/
    ///        group vector (e.g. a withheld response column) with embedding()/plot_embedding().
    [[nodiscard]] const std::vector<std::size_t>& kept_row_indices() const { return kept_row_indices_; }

    /// @brief Every eigenvalue of the double-centered inner-product matrix (length =
    ///        observations()), descending -- the first n_components() were used to build the
    ///        embedding. Negative eigenvalues (possible for non-Euclidean dissimilarities) are
    ///        clamped to 0 when building the embedding but reported here as-is.
    [[nodiscard]] const std::vector<double>& eigenvalues() const { return eigenvalues_; }
    /// @brief Sum of the n_components() (clamped-non-negative) eigenvalues used, divided by the
    ///        sum of all positive eigenvalues -- 1.0 means the embedding reproduces all of the
    ///        genuinely Euclidean-representable variation in the pairwise distances.
    [[nodiscard]] double goodness_of_fit() const { return goodness_of_fit_; }

    /// @brief n x n_components matrix of embedded coordinates.
    [[nodiscard]] const linalg::DenseMatrix<double>& embedding() const { return embedding_; }
    [[nodiscard]] std::vector<double> dimension(std::size_t index) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] plot::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    [[nodiscard]] plot::RPlot plot_embedding(const std::vector<std::string>& group_labels, std::size_t dimension_x = 0,
                                             std::size_t dimension_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    MDSOptions                 options_;

    std::vector<std::size_t> kept_row_indices_;
    std::vector<double>       eigenvalues_;
    double                     goodness_of_fit_{0.0};
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
};

} // namespace datamunge::stats
