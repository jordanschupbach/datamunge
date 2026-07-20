#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/random/random.hpp>
#include <datamunge/stats/decision_tree_regressor.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct GBMRegressorOptions {
    std::size_t   n_trees{100};
    double        learning_rate{0.1};
    std::size_t   max_depth{3}; // shallow "weak learner" trees, unlike RandomForest's deeper defaults
    std::size_t   min_samples_split{2};
    std::size_t   min_samples_leaf{1};
    // Fraction of training rows drawn (without replacement) to fit each
    // tree -- "stochastic gradient boosting" (Friedman 2002). 1.0 uses
    // every row for every tree.
    double        subsample{1.0};
    std::uint64_t seed{42};
};

// Gradient boosting regressor: a sequence of shallow regression trees, each
// fit to the residuals of the current ensemble (functional gradient descent
// on squared error), using the same formula + DataFrame constructor style
// as the other stats classes:
//
//   GBMRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Unlike RandomForest, there is no natural out-of-bag estimate here, so
// fitted_values()/r_squared()/rmse() are ordinary (in-sample) training
// metrics; use training_deviance() to see how they evolved round by round
// and judge over/underfitting from the shape of that curve.
class GBMRegressor {
 public:
    GBMRegressor(const dstruct::DataFrame& data, const std::string& formula, GBMRegressorOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }

    // SSE-decrease importance averaged across all trees, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;
    // Mean squared error on the training set after each boosting round (length n_trees()).
    [[nodiscard]] const std::vector<double>& training_deviance() const { return training_deviance_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::RPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::RPlot plot_residuals_vs_fitted() const;
    [[nodiscard]] plot::RPlot plot_training_deviance() const;

 private:
    void fit(const dstruct::DataFrame& data);

    Formula              formula_;
    DesignInfo           design_;
    GBMRegressorOptions  options_;

    std::vector<std::string>            predictor_names_;
    std::vector<DecisionTreeRegressor>  trees_;
    double                               initial_prediction_{0.0};

    std::vector<double> training_y_;
    std::vector<double> fitted_;
    std::vector<double> training_deviance_;
    std::size_t          observations_{0};
};

} // namespace datamunge::stats
