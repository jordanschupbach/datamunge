#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::XGBoostClassifier;
using datamunge::stats::XGBoostClassifierOptions;

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n\n";

    XGBoostClassifier model(iris, "Species ~ Petal.Length + Petal.Width");
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

    model.plot_training_deviance().save("xgboost_iris_training_deviance.svg");
    model.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions.svg");
    std::cout << "\nSaved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg\n";

    // XGBoost's defining lever is regularization rather than shrinkage or
    // depth alone: a heavily L2-regularized model (large lambda) grows the
    // same deep trees but keeps every leaf weight small, producing a much
    // smoother decision boundary than the lightly-regularized default.
    XGBoostClassifierOptions heavy_options;
    heavy_options.lambda = 50.0;
    XGBoostClassifier heavy(iris, "Species ~ Petal.Length + Petal.Width", heavy_options);
    std::cout << "\nlambda=1 (default):   training accuracy=" << model.training_accuracy() * 100.0
               << "%  deviance=" << model.training_deviance().back() << "\n";
    std::cout << "lambda=50 (heavy L2): training accuracy=" << heavy.training_accuracy() * 100.0
               << "%  deviance=" << heavy.training_deviance().back() << "\n";
    heavy.plot_decision_regions("Petal.Length", "Petal.Width").save("xgboost_iris_decision_regions_heavy_lambda.svg");
    std::cout << "Saved xgboost_iris_decision_regions_heavy_lambda.svg\n";

    return 0;
}
