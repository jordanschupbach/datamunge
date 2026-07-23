#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/mds.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::DistanceMetric;
using datamunge::stats::MDS;
using datamunge::stats::MDSOptions;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    const auto iris = datamunge::datasets::iris();
    const std::vector<std::string> features{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};
    std::vector<std::string> species(iris.nrows());
    for (std::size_t i = 0; i < iris.nrows(); ++i) species[i] = iris.string_at("Species", i);

    std::cout << "=================== Classical MDS on iris (euclidean) ===================\n";
    MDS mds(iris, features, MDSOptions{2, DistanceMetric::Euclidean});
    mds.print_summary();

    std::cout << "\nFirst 5 rows of the embedding:\n";
    for (std::size_t i = 0; i < 5; ++i) {
        std::cout << "  " << species[i] << ": Dim1=" << mds.embedding()(i, 0) << ", Dim2=" << mds.embedding()(i, 1) << "\n";
    }

    const auto scatter = mds.plot_embedding(species);
    scatter.save("mds_iris_embedding.svg");
    std::cout << "\nembedding scatter (colored by species) saved to mds_iris_embedding.svg\n";

    std::cout << "\n=================== Classical MDS on iris (manhattan) ===================\n";
    MDS manhattan_mds(iris, features, MDSOptions{2, DistanceMetric::Manhattan});
    std::cout << "Goodness of fit: euclidean=" << mds.goodness_of_fit() << ", manhattan=" << manhattan_mds.goodness_of_fit()
              << "\n";

    return 0;
}
