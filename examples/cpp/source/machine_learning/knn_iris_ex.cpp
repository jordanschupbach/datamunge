#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::KNNClassifier;
using datamunge::stats::KNNClassifierOptions;

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n\n";

    KNNClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
    model.print_summary(std::cout);

    std::cout << "\nConfusion matrix (leave-one-out; rows = actual, cols = predicted):\n";
    const auto cm = model.confusion_matrix();
    std::cout << std::setw(14) << "";
    for (const auto& c : model.classes()) std::cout << std::setw(12) << c;
    std::cout << "\n";
    for (std::size_t i = 0; i < cm.rows(); ++i) {
        std::cout << std::setw(14) << model.classes()[i];
        for (std::size_t j = 0; j < cm.cols(); ++j) std::cout << std::setw(12) << cm(i, j);
        std::cout << "\n";
    }

    std::cout << "\nLeave-one-out misclassified rows:\n";
    const auto& fitted = model.fitted_classes();
    std::size_t misclassified = 0;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        const auto actual = iris.string_at("Species", i);
        if (fitted[i] == actual) continue;
        ++misclassified;
        std::cout << "  row " << i << ": Petal.Length=" << iris.double_at("Petal.Length", i)
                   << " Petal.Width=" << iris.double_at("Petal.Width", i) << "  actual=" << actual
                   << "  predicted=" << fitted[i] << "\n";
    }
    std::cout << misclassified << " of " << iris.nrows() << " misclassified ("
               << (100.0 * static_cast<double>(misclassified) / static_cast<double>(iris.nrows())) << "%)\n";

    model.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k5.svg");
    std::cout << "\nSaved knn_iris_decision_regions_k5.svg\n";

    // k=1 memorizes every training point exactly (jagged, overfit boundary
    // with an island around every point, including noise); k=25 averages
    // over a much larger neighborhood (very smooth, underfit boundary).
    // Neither extreme is visible from resubstitution accuracy alone -- only
    // the decision-region shape (or leave-one-out accuracy) reveals it.
    KNNClassifierOptions k1_options;
    k1_options.k = 1;
    KNNClassifier k1(iris, "Species ~ Petal.Length + Petal.Width", k1_options);
    std::cout << "\nk=1  leave-one-out accuracy: " << k1.training_accuracy() * 100.0 << "%\n";
    k1.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k1.svg");
    std::cout << "Saved knn_iris_decision_regions_k1.svg\n";

    KNNClassifierOptions k25_options;
    k25_options.k = 25;
    KNNClassifier k25(iris, "Species ~ Petal.Length + Petal.Width", k25_options);
    std::cout << "\nk=25 leave-one-out accuracy: " << k25.training_accuracy() * 100.0 << "%\n";
    k25.plot_decision_regions("Petal.Length", "Petal.Width").save("knn_iris_decision_regions_k25.svg");
    std::cout << "Saved knn_iris_decision_regions_k25.svg\n";

    return 0;
}
