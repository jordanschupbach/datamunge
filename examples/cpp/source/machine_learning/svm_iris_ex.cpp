#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::dstruct::DataFrame;
using datamunge::stats::SVM;
using datamunge::stats::SVMKernel;
using datamunge::stats::SVMOptions;

namespace {

void print_confusion_matrix(const SVM& model) {
    const auto cm = model.confusion_matrix();
    std::cout << std::setw(14) << "";
    for (const auto& c : model.classes()) std::cout << std::setw(12) << c;
    std::cout << "\n";
    for (std::size_t i = 0; i < cm.rows(); ++i) {
        std::cout << std::setw(14) << model.classes()[i];
        for (std::size_t j = 0; j < cm.cols(); ++j) std::cout << std::setw(12) << cm(i, j);
        std::cout << "\n";
    }
}

} // namespace

int main() {
    const auto iris = datamunge::datasets::iris();
    std::cout << "iris: " << iris.nrows() << " rows x " << iris.ncols() << " cols\n\n";

    std::cout << "=== RBF kernel (default) ===\n";
    SVM rbf_model(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    rbf_model.print_summary(std::cout);
    std::cout << "\nConfusion matrix (rows = actual, cols = predicted):\n";
    print_confusion_matrix(rbf_model);

    std::cout << "\n=== Linear kernel, for comparison ===\n";
    SVMOptions linear_options;
    linear_options.kernel = SVMKernel::Linear;
    SVM linear_model(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", linear_options);
    std::cout << "Training accuracy: " << linear_model.training_accuracy() * 100.0 << "%\n";
    std::cout << "Support vectors: " << linear_model.num_support_vectors() << "\n";

    // Classify a few new flowers, including one right on the versicolor/virginica boundary.
    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.1, 6.0, 6.5, 6.2});
    newdata.add_column("Sepal.Width", std::vector<double>{3.5, 2.7, 3.0, 2.8});
    newdata.add_column("Petal.Length", std::vector<double>{1.4, 4.5, 5.5, 4.8});
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});

    const auto prediction = rbf_model.predict_detail(newdata);
    std::cout << "\nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):\n";
    std::cout << std::setw(14) << "predicted";
    for (const auto& c : rbf_model.classes()) std::cout << std::setw(18) << ("votes(" + c + ")");
    std::cout << "\n";
    for (std::size_t i = 0; i < prediction.class_label.size(); ++i) {
        std::cout << std::setw(14) << prediction.class_label[i];
        for (const double v : prediction.votes[i]) std::cout << std::setw(18) << v;
        std::cout << "\n";
    }

    return 0;
}
