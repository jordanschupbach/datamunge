#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/knn_classifier.hpp> // DistanceMetric

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct DBSCANOptions {
    double eps{0.5};
    std::size_t min_samples{5};
    DistanceMetric metric{DistanceMetric::Euclidean};
};

/// @brief Density-based spatial clustering (Ester et al. 1996) fit from a DataFrame and a
///        list of numeric feature columns. Rows with a null value in any feature column are
///        dropped before fitting. Unlike KMeans/AgglomerativeClustering, the number of
///        clusters is discovered automatically from the density structure, and points in
///        low-density regions are labeled as noise rather than forced into a cluster.
class DBSCAN {
public:
    DBSCAN(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns, DBSCANOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t n_clusters() const { return n_clusters_; }
    [[nodiscard]] std::size_t n_noise() const { return n_noise_; }

    /// @brief Cluster index (0-based) assigned to each fitted row, or -1 for noise, in the
    ///        same order as the (null-dropped) training data.
    [[nodiscard]] const std::vector<int>& labels() const { return labels_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    DBSCANOptions options_;

    std::vector<int> labels_;
    std::size_t observations_{0};
    std::size_t n_clusters_{0};
    std::size_t n_noise_{0};
};

} // namespace datamunge::stats
