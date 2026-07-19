#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/stats/knn_classifier.hpp> // DistanceMetric

#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

enum class LinkageCriterion { Single, Complete, Average, Ward };

struct AgglomerativeClusteringOptions {
    std::size_t n_clusters{2};
    LinkageCriterion linkage{LinkageCriterion::Ward};
    /// @brief Ward linkage is only defined for Euclidean distance; other linkages accept either.
    DistanceMetric metric{DistanceMetric::Euclidean};
};

/// @brief Agglomerative (bottom-up) hierarchical clustering fit from a DataFrame and a list
///        of numeric feature columns. Rows with a null value in any feature column are
///        dropped before fitting. The full merge history (dendrogram) is always built, so
///        the fitted model can be re-cut at a different cluster count without refitting.
class AgglomerativeClustering {
public:
    struct Merge {
        std::size_t cluster_a; // ids: 0..n-1 are the original points; n, n+1, ... are merge results, in merge order
        std::size_t cluster_b;
        double distance;
        std::size_t size; // number of original points in the resulting merged cluster
    };

    AgglomerativeClustering(const dstruct::DataFrame& data, const std::vector<std::string>& feature_columns,
                            AgglomerativeClusteringOptions options = {});

    [[nodiscard]] const std::vector<std::string>& feature_names() const { return feature_columns_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }

    /// @brief Cluster index (0-based, arbitrary numbering) assigned to each fitted row at
    ///        options.n_clusters, in the same order as the (null-dropped) training data.
    [[nodiscard]] const std::vector<std::size_t>& labels() const { return labels_; }

    /// @brief The full sequence of merges performed while building the dendrogram (length
    ///        observations() - 1), in increasing order of merge distance.
    [[nodiscard]] const std::vector<Merge>& merge_history() const { return merges_; }

    /// @brief Re-cuts the already-built dendrogram to produce @p n_clusters clusters,
    ///        without refitting.
    [[nodiscard]] std::vector<std::size_t> cut(std::size_t n_clusters) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

private:
    void fit(const dstruct::DataFrame& data);

    std::vector<std::string> feature_columns_;
    AgglomerativeClusteringOptions options_;

    std::vector<Merge> merges_;
    std::vector<std::size_t> labels_;
    std::size_t observations_{0};
};

} // namespace datamunge::stats
