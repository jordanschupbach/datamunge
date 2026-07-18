#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::DecisionTreeClassifier;

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n\n";

    DecisionTreeClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
    model.print_summary(std::cout);

    std::cout << "\nConfusion matrix (rows = actual, cols = predicted):\n";
    const auto cm = model.confusion_matrix();
    std::cout << std::setw(14) << "";
    for (const auto& c : model.classes()) std::cout << std::setw(12) << c;
    std::cout << "\n";
    for (std::size_t i = 0; i < cm.rows(); ++i) {
        std::cout << std::setw(14) << model.classes()[i];
        for (std::size_t j = 0; j < cm.cols(); ++j) std::cout << std::setw(12) << cm(i, j);
        std::cout << "\n";
    }

    std::cout << "\nMisclassified rows:\n";
    const auto predictions = model.predict(iris);
    std::size_t misclassified = 0;
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        const auto actual = iris.string_at("Species", i);
        if (predictions[i] == actual) continue;
        ++misclassified;
        std::cout << "  row " << i << ": Petal.Length=" << iris.double_at("Petal.Length", i)
                   << " Petal.Width=" << iris.double_at("Petal.Width", i) << "  actual=" << actual
                   << "  predicted=" << predictions[i] << "\n";
    }
    std::cout << misclassified << " of " << iris.nrows() << " misclassified ("
               << (100.0 * static_cast<double>(misclassified) / static_cast<double>(iris.nrows())) << "%)\n";

    model.plot_classification(iris, "Petal.Length", "Petal.Width").save("tree_iris_classification.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions.svg");
    std::cout << "\nSaved tree_iris_classification.svg and tree_iris_decision_regions.svg\n";

    // A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
    DecisionTreeClassifier shallow(iris, "Species ~ Petal.Length + Petal.Width",
                                  [] {
                                      datamunge::stats::DecisionTreeClassifierOptions options;
                                      options.max_depth = 2;
                                      return options;
                                  }());
    std::cout << "\nDepth-2 tree training accuracy: " << shallow.training_accuracy() * 100.0 << "% ("
               << shallow.leaf_count() << " leaves)\n";
    shallow.plot_decision_regions("Petal.Length", "Petal.Width").save("tree_iris_decision_regions_depth2.svg");
    std::cout << "Saved tree_iris_decision_regions_depth2.svg\n";

    return 0;
}
