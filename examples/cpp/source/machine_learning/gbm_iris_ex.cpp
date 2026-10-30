#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::GBMClassifier;
using datamunge::stats::GBMClassifierOptions;

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n\n";

    GBMClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_training_deviance().save("gbm_iris_training_deviance.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions.svg");
    std::cout << "\nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg\n";

    // A handful of boosting rounds vs a well-boosted ensemble, showing the
    // defining GBM effect: each additional round chips away at the
    // training loss, gradually sharpening the decision boundary.
    GBMClassifierOptions few_options;
    few_options.n_trees = 5;
    GBMClassifier few(iris, "Species ~ Petal.Length + Petal.Width", few_options);
    std::cout << "\n5-round ensemble:   training accuracy=" << few.training_accuracy() * 100.0
               << "%  deviance=" << few.training_deviance().back() << "\n";
    std::cout << "100-round ensemble: training accuracy=" << model.training_accuracy() * 100.0
               << "%  deviance=" << model.training_deviance().back() << "\n";
    few.plot_decision_regions("Petal.Length", "Petal.Width").save("gbm_iris_decision_regions_5rounds.svg");
    std::cout << "Saved gbm_iris_decision_regions_5rounds.svg\n";

    return 0;
}
