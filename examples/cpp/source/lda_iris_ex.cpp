#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::dstruct::DataFrame;
using datamunge::stats::LDA;

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n\n";

    LDA model(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
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

    // Classify a few new flowers, including one right on the versicolor/virginica boundary.
    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.1, 6.0, 6.5, 6.2});
    newdata.add_column("Sepal.Width", std::vector<double>{3.5, 2.7, 3.0, 2.8});
    newdata.add_column("Petal.Length", std::vector<double>{1.4, 4.5, 5.5, 4.8});
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});

    const auto prediction = model.predict_detail(newdata);
    std::cout << "\nPredictions for new flowers:\n";
    std::cout << std::setw(14) << "predicted";
    for (const auto& c : model.classes()) std::cout << std::setw(14) << ("P(" + c + ")");
    std::cout << std::setw(10) << "LD1" << std::setw(10) << "LD2\n";
    for (std::size_t i = 0; i < prediction.class_label.size(); ++i) {
        std::cout << std::setw(14) << prediction.class_label[i];
        for (const double p : prediction.posterior[i]) std::cout << std::setw(14) << std::fixed << std::setprecision(4) << p;
        for (const double ld : prediction.discriminants[i]) std::cout << std::setw(10) << std::fixed << std::setprecision(3) << ld;
        std::cout << "\n";
    }

    model.save_discriminant_plot("lda_iris_discriminants.svg");
    std::cout << "\nSaved discriminant plot as lda_iris_discriminants.svg\n";

    return 0;
}
