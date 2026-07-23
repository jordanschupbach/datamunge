#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/pca.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::PCA;
using datamunge::stats::PCAOptions;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    const auto iris = datamunge::datasets::iris();
    const std::vector<std::string> features{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};
    std::vector<std::string> species(iris.nrows());
    for (std::size_t i = 0; i < iris.nrows(); ++i) species[i] = iris.string_at("Species", i);

    std::cout << "=================== PCA on iris (scaled) ===================\n";
    PCA pca(iris, features, PCAOptions{true, true});
    pca.print_summary();

    std::cout << "\nFirst 5 rows of scores:\n";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << "  " << species[i] << ": PC1=" << pca.scores()(i, 0) << ", PC2=" << pca.scores()(i, 1) << "\n";
    }

    std::cout << "\nPC1 loadings (which original features drive it):\n";
    const auto pc1_loadings = pca.component_loadings(0);
    for (std::size_t j = 0; j < features.size(); ++j) std::cout << "  " << features[j] << ": " << pc1_loadings[j] << "\n";

    const auto scatter = pca.plot_scores(species);
    scatter.save("pca_iris_scores.svg");
    std::cout << "\nscores scatter (colored by species) saved to pca_iris_scores.svg\n";

    const auto scree = pca.plot_scree();
    scree.save("pca_iris_scree.svg");
    std::cout << "scree plot saved to pca_iris_scree.svg\n";

    std::cout << "\n=================== PCA on iris (unscaled) ===================\n";
    PCA unscaled(iris, features, PCAOptions{true, false});
    std::cout << "PC1 explains " << unscaled.explained_variance_ratio()[0] * 100.0
              << "% of variance (vs. " << pca.explained_variance_ratio()[0] * 100.0
              << "% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.\n";

    return 0;
}
