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

struct SammonMappingOptions {
    std::size_t n_components{2};
    /// @brief Sammon's own "magic factor" from the original paper -- a step-size scale on the
    ///        pseudo-Newton update. The paper recommends 0.3-0.4.
    double learning_rate{0.3};
    std::size_t max_iterations{500};
    /// @brief Stop early once the relative improvement in stress between successive sweeps falls
    ///        below this (and stress is still decreasing).
    double tolerance{1e-9};
    DistanceMetric metric{DistanceMetric::Euclidean};
    std::uint64_t seed{42};
};

/// @brief Sammon mapping (Sammon, 1969): a classic nonlinear-MDS-family method that predates and
///        directly influenced t-SNE. Unlike classical MDS's closed-form eigendecomposition, Sammon
///        mapping iteratively minimizes a weighted sum-of-squared-distance-errors "stress" via a
///        pseudo-Newton update, WEIGHTING errors by 1/Dstar_ij -- so small (nearby) high-
///        dimensional distances are preserved much more faithfully than large ones, unlike
///        classical MDS which weights all pairs equally. Rows with a null value in any feature
///        column are dropped before fitting.
class SammonMapping {
public:
    SammonMapping(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                  SammonMappingOptions options = {});

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

    /// @brief Final Sammon stress E = (1/c) * sum_{i<j} (Dstar_ij - d_ij)^2 / Dstar_ij, where c is
    ///        the sum of all high-dimensional pairwise distances -- lower is better, 0 is a
    ///        perfect distance-preserving embedding.
    [[nodiscard]] double stress() const { return stress_; }
    /// @brief Number of full sweeps actually performed before convergence or max_iterations.
    [[nodiscard]] std::size_t iterations_run() const { return iterations_run_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] plot::RPlot plot_embedding(std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;
    [[nodiscard]] plot::RPlot plot_embedding(const std::vector<std::string>& group_labels,
                                             std::size_t dimension_x = 0, std::size_t dimension_y = 1) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    SammonMappingOptions       options_;

    std::vector<std::size_t> kept_row_indices_;
    linalg::DenseMatrix<double> embedding_;
    std::size_t                 observations_{0};
    double                       stress_{0.0};
    std::size_t                 iterations_run_{0};
};

} // namespace datamunge::stats
