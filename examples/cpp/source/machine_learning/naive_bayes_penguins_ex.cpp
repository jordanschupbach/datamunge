#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/stats.hpp>

#include <iomanip>
#include <iostream>

using datamunge::stats::NaiveBayesClassifier;

int main() {
    auto penguins = datamunge::datasets::penguins().drop_nulls(
        {"species", "island", "sex", "bill_length_mm", "bill_depth_mm"});
    std::cout << "penguins: " << penguins.nrows() << " rows x " << penguins.ncols() << " cols\n\n";

    // A mix of numeric (Gaussian likelihood) and categorical (frequency-table
    // likelihood) predictors in one formula -- each modeled independently
    // given the class, per the naive Bayes assumption.
    NaiveBayesClassifier model(penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex");
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
    const auto predictions = model.predict(penguins);
    std::size_t misclassified = 0;
    for (std::size_t i = 0; i < penguins.nrows(); ++i) {
        const auto actual = penguins.string_at("species", i);
        if (predictions[i] == actual) continue;
        ++misclassified;
        std::cout << "  row " << i << ": bill_length=" << penguins.double_at("bill_length_mm", i)
                   << " bill_depth=" << penguins.double_at("bill_depth_mm", i)
                   << " island=" << penguins.string_at("island", i) << " sex=" << penguins.string_at("sex", i)
                   << "  actual=" << actual << "  predicted=" << predictions[i] << "\n";
    }
    std::cout << misclassified << " of " << penguins.nrows() << " misclassified ("
               << (100.0 * static_cast<double>(misclassified) / static_cast<double>(penguins.nrows())) << "%)\n";

    // plot_decision_regions requires exactly two NUMERIC predictors, so
    // build a separate two-predictor model (bill measurements alone) just
    // for visualization -- the same Gaussian-per-class idea, just without
    // the categorical predictors mixed in.
    NaiveBayesClassifier bill_only(penguins, "species ~ bill_length_mm + bill_depth_mm");
    std::cout << "\nbill-measurements-only model training accuracy: " << bill_only.training_accuracy() * 100.0
               << "%\n";
    bill_only.plot_classification(penguins, "bill_length_mm", "bill_depth_mm")
        .save("naive_bayes_penguins_classification.svg");
    bill_only.plot_decision_regions("bill_length_mm", "bill_depth_mm")
        .save("naive_bayes_penguins_decision_regions.svg");
    std::cout << "Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg\n";

    return 0;
}
