#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/agglomerative_clustering.hpp>
#include <datamunge/stats/dbscan.hpp>
#include <datamunge/stats/kmeans.hpp>

#include <iomanip>
#include <iostream>
#include <map>

using datamunge::stats::AgglomerativeClustering;
using datamunge::stats::AgglomerativeClusteringOptions;
using datamunge::stats::DBSCAN;
using datamunge::stats::DBSCANOptions;
using datamunge::stats::KMeans;
using datamunge::stats::KMeansOptions;
using datamunge::stats::LinkageCriterion;

namespace {

// Reports, for each fitted cluster, which true species is most common in it and how "pure"
// that cluster is -- a quick, intuitive way to sanity-check unsupervised clusters against
// known labels without needing the clustering algorithm to have ever seen those labels.
template <typename Labels>
void report_purity(const Labels& cluster_labels, const std::vector<std::string>& species, std::size_t n_clusters) {
    std::vector<std::map<std::string, int>> votes(n_clusters);
    for (std::size_t i = 0; i < species.size(); ++i) {
        const auto l = cluster_labels[i];
        if (l < 0) continue; // DBSCAN noise
        ++votes[static_cast<std::size_t>(l)][species[i]];
    }
    for (std::size_t c = 0; c < n_clusters; ++c) {
        int total = 0, best = 0;
        std::string best_species;
        for (const auto& [name, count] : votes[c]) {
            total += count;
            if (count > best) {
                best = count;
                best_species = name;
            }
        }
        if (total == 0) continue;
        std::cout << "  cluster " << c << ": " << total << " points, majority " << best_species << " (" << best
                  << "/" << total << ")\n";
    }
}

} // namespace

int main() {
    std::cout << std::fixed << std::setprecision(4);

    const auto iris = datamunge::datasets::iris();
    std::vector<std::string> species(iris.nrows());
    for (std::size_t i = 0; i < iris.nrows(); ++i) species[i] = iris.string_at("Species", i);
    const std::vector<std::string> features{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

    std::cout << "=================== K-means (k=3) on iris ===================\n";
    KMeansOptions kmeans_options;
    kmeans_options.n_clusters = 3;
    const KMeans kmeans(iris, features, kmeans_options);
    std::cout << "Inertia: " << kmeans.inertia() << ", iterations (best run): " << kmeans.iterations_used() << "\n";
    report_purity(kmeans.labels(), species, 3);

    std::cout << "\n=================== Agglomerative clustering on iris ===================\n";
    for (const auto linkage : {LinkageCriterion::Ward, LinkageCriterion::Complete, LinkageCriterion::Average}) {
        AgglomerativeClusteringOptions options;
        options.n_clusters = 3;
        options.linkage = linkage;
        const AgglomerativeClustering model(iris, features, options);
        const std::string name = linkage == LinkageCriterion::Ward     ? "ward"
                                  : linkage == LinkageCriterion::Complete ? "complete"
                                                                          : "average";
        std::cout << "--- linkage=" << name << " ---\n";
        report_purity(model.labels(), species, 3);
    }

    std::cout << "\n--- Re-cutting the ward dendrogram at k=2 without refitting ---\n";
    AgglomerativeClusteringOptions ward_options;
    ward_options.n_clusters = 3;
    ward_options.linkage = LinkageCriterion::Ward;
    const AgglomerativeClustering ward_model(iris, features, ward_options);
    report_purity(ward_model.cut(2), species, 2);

    std::cout << "\n=================== DBSCAN on iris ===================\n";
    DBSCANOptions dbscan_options;
    dbscan_options.eps = 0.6;
    dbscan_options.min_samples = 5;
    const DBSCAN dbscan(iris, features, dbscan_options);
    std::cout << "Clusters found: " << dbscan.n_clusters() << ", noise points: " << dbscan.n_noise() << " (of "
              << dbscan.observations() << ")\n";
    report_purity(dbscan.labels(), species, dbscan.n_clusters());

    return 0;
}
