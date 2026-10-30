#pragma once
/// @file clustering_diagnostics.hpp
/// @brief Label-permutation-invariant clustering summaries and local-file-ready graphics.
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/agglomerative_clustering.hpp>
#include <string>
#include <vector>
namespace datamunge::stats {
/// @brief External partition agreement, with explicit DBSCAN noise coverage.
/// Labels are compared to reference classes after fitting, not used to train or choose
/// parameters. Purity excludes noise; adjusted Rand treats all -1 rows as one category.
/// The latter convention measures a partition including noise, not successful assignment
/// of those rows. Scores are invariant to arbitrary cluster/class renaming. Value-owned
/// result; no shared mutable state. See cluster_agreement() for input validation.
/// @par Example
/// @code
/// const auto s=dm::stats::cluster_agreement({7,7,9,9},{"a","a","b","b"});
/// std::print("{:.0f} {:.0f}\n",s.purity,s.adjusted_rand); // 1 1
/// @endcode
/// @par References
/// - Hubert and Arabie (1985), Comparing partitions. https://doi.org/10.1007/BF01908075
/// - Manning et al. (2008), Evaluation of clustering (purity).
///   https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
/// @see cluster_agreement(), cluster_composition()
struct ClusterAgreement {
  std::size_t observations; ///< Number of aligned observations, including noise.
  std::size_t
      clusters;      ///< Number of distinct nonnegative cluster labels, including singleton groups.
  std::size_t noise; ///< Number of rows labelled -1.
  double      coverage; ///< Assigned rows divided by observations.
  double purity; ///< Sum of assigned-cluster majority counts divided by assigned rows; NaN if none
                 ///< assigned.
  double adjusted_rand; ///< Chance-adjusted Rand index on all rows, with noise as one category; may
                        ///< be negative.
};
/// @brief Computes external cluster agreement without assuming label numbering matches classes.
/// Uses contingency pair counts and the fixed-marginal adjusted Rand index. If n=1,
/// or both partitions are the same degenerate all-singleton/all-one partition, ARI is
/// one. Purity can increase with fragmentation and is conditional on assignment; always
/// read it with coverage and cluster count. Work O(n log n), auxiliary storage O(n).
/// @param labels Nonempty aligned cluster labels, any nonnegative ids; -1 means noise.
/// @param truth Equally sized reference class strings; empty strings are ordinary classes.
/// @return Owned scores. All noise gives coverage zero and NaN purity; ARI still uses all rows.
/// @throws std::invalid_argument Empty/mismatched inputs or a label below -1.
/// @par Example
/// See ClusterAgreement.
/// @par References
/// - Hubert and Arabie (1985), Comparing partitions. https://doi.org/10.1007/BF01908075
/// @see cluster_composition()
[[nodiscard]] ClusterAgreement cluster_agreement(
    const std::vector<int>&         labels,
    const std::vector<std::string>& truth);
/// @brief Summarizes class composition of every observed cluster, including the noise category.
/// Borrows inputs unchanged; output owns its columns. Tied majority classes choose the
/// lexicographically first class. Noise's row describes its composition only; it is not
/// included in cluster_agreement()'s assigned purity. Work/storage O(n log n)/O(n).
/// @param labels Nonempty cluster ids >=0 or -1 for noise; ids need not be contiguous.
/// @param truth Equally sized reference class strings.
/// @return Rows sorted by cluster id: cluster, size, majority_class, majority_count, purity.
/// @throws std::invalid_argument Empty/mismatched inputs or label below -1.
/// @par Example
/// @code
/// const auto t=dm::stats::cluster_composition({0,0,-1},{"b","a","b"});
/// std::print("{}\n",t.string_at("majority_class",1)); // a
/// @endcode
/// @par References
/// - Hubert and Arabie (1985). https://doi.org/10.1007/BF01908075 (contingency counts).
/// @see cluster_agreement()
[[nodiscard]] dstruct::DataFrame cluster_composition(
    const std::vector<int>&         labels,
    const std::vector<std::string>& truth);
/// @brief Plots cluster labels on two named numeric coordinates; noise uses a separate grey series.
/// This is a projection of existing labels, not a refit in two dimensions or an inferred
/// boundary. Any nonnegative ids are supported; colours cycle after eight groups.
/// All rows must align exactly with labels: explicitly remove incomplete fit rows first.
/// No implicit row dropping/standardization. Work/storage O(n log n)/O(n); no file I/O.
/// @param data Borrowed frame with finite, nonnull numeric coordinates.
/// @param x_feature Name of horizontal coordinate.
/// @param y_feature Name of vertical coordinate.
/// @param labels Exactly data.nrows() cluster ids >=0 or -1; nonempty.
/// @param title Owned plot title, default "Cluster projection".
/// @return Owned plot with labelled series and padded bounds, ready for save_svg().
/// @throws std::invalid_argument Bad schema/alignment, missing/nonfinite coordinates or labels
/// below -1.
/// @par Example
/// @code
/// dm::dstruct::DataFrame
/// d;d.add_column("x",std::vector<double>{0,1});d.add_column("y",std::vector<double>{0,1}); const
/// auto p=dm::stats::plot_clusters(d,"x","y",{0,-1}); std::print("plot ready\n"); // plot ready
/// @endcode
/// @see cluster_composition()
[[nodiscard]] graphics::RPlot plot_clusters(
    const dstruct::DataFrame& data,
    const std::string&        x_feature,
    const std::string&        y_feature,
    const std::vector<int>&   labels,
    std::string               title = "Cluster projection");
/// @brief Draws the full fitted merge tree with contiguous subtrees in traversal leaf order.
/// Leaves are ordered by recursively visiting cluster_a before cluster_b; horizontal
/// positions are traversal positions, not original row ids. Heights are the model's
/// reported linkage distances. For Ward, height^2/2 is the within-cluster SSE increment.
/// Borrows the fitted tree unchanged; the plot owns its geometry. Work/storage O(n);
/// no file I/O and no refit. All-zero heights receive a positive plotting range.
/// @param model Complete fitted agglomerative tree, with at least two observations and finite
/// heights.
/// @param title Owned plot title, default "Agglomerative merge tree".
/// @return Owned dendrogram, ready for save_svg().
/// @throws std::invalid_argument Incomplete tree, invalid child ids/heights or unrepresentable plot
/// range.
/// @par Example
/// @code
/// dm::dstruct::DataFrame d;d.add_column("x",std::vector<double>{0,2,8,10});
/// const dm::stats::AgglomerativeClustering fit(d,{"x"});
/// const auto p=dm::stats::plot_dendrogram(fit);
/// std::print("{}\n",fit.merge_history().size()); // 3
/// @endcode
/// @par References
/// - Ward (1963), Hierarchical grouping to optimize an objective function.
///   https://doi.org/10.1080/01621459.1963.10500845
/// @see AgglomerativeClustering::merge_history(), AgglomerativeClustering::cut()
[[nodiscard]] graphics::RPlot plot_dendrogram(
    const AgglomerativeClustering& model,
    std::string                    title = "Agglomerative merge tree");
} // namespace datamunge::stats
