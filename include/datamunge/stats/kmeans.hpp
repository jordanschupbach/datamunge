#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct KMeansOptions {
    std::size_t n_clusters{8};
    std::size_t max_iterations{300};
    /// @brief Number of random restarts (each with a fresh k-means++ initialization); the
    ///        restart with the lowest inertia is kept, matching scikit-learn's default behavior.
    std::size_t n_init{10};
    /// @brief Stops an individual run once every center moves by less than this (Euclidean distance).
    double tolerance{1e-4};
    std::uint64_t seed{42};
};

/// @brief K-means clustering (Lloyd's algorithm with k-means++ initialization) fit from a
///        DataFrame and a list of numeric feature columns. Rows with a null value in any
///        feature column are dropped before fitting.
class KMeans {
public:
    KMeans(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, KMeansOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t n_clusters() const { return options_.n_clusters; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t iterations_used() const { return iterations_used_; }

    /// @brief Cluster index (0-based) assigned to each fitted row, in the same order as the
    ///        (null-dropped) training data.
    [[nodiscard]] const std::vector<std::size_t>& labels() const { return labels_; }
    /// @brief n_clusters x n_features matrix of cluster centers.
    [[nodiscard]] const linalg::DenseMatrix<double>& cluster_centers() const { return centers_; }
    /// @brief Sum of squared Euclidean distances from each point to its assigned center.
    [[nodiscard]] double inertia() const { return inertia_; }

    /// @brief Assigns each row of @p newdata to its nearest fitted cluster center.
    [[nodiscard]] std::vector<std::size_t> predict(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    KMeansOptions             options_;

    linalg::DenseMatrix<double> centers_;
    std::vector<std::size_t>    labels_;
    double                       inertia_{0.0};
    std::size_t                  observations_{0};
    std::size_t                  iterations_used_{0};
};

} // namespace datamunge::stats
