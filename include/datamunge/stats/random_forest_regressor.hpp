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

struct RandomForestRegressorOptions {
    std::size_t   n_trees{100};
    std::size_t   max_depth{10};
    std::size_t   min_samples_split{2};
    std::size_t   min_samples_leaf{1};
    // 0 = auto: floor(number of predictors / 3), at least 1 (the usual
    // regression default). Each tree considers this many randomly chosen
    // predictors at every split.
    std::size_t   max_features{0};
    bool          bootstrap{true};
    double        sample_fraction{1.0};
    std::uint64_t seed{42};
};

// Bagged ensemble of DecisionTreeRegressor trees, each fit on a bootstrap
// resample of the training rows and restricted to a random subset of
// predictors at every split. Same formula + DataFrame constructor style as
// the other stats classes:
//
//   RandomForestRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Predictions are the mean across trees. Out-of-bag rows give a built-in
// R-squared/RMSE estimate via oob_r_squared()/oob_rmse() without needing a
// held-out test set.
class RandomForestRegressor {
 public:
    RandomForestRegressor(const dstruct::DataFrame& data, const std::string& formula,
                          RandomForestRegressorOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }
    [[nodiscard]] std::size_t                     max_features_used() const { return max_features_used_; }

    // Per-tree SSE-decrease importance, averaged across trees and
    // normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_values_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;
    [[nodiscard]] double                     oob_r_squared() const { return oob_r_squared_; }
    [[nodiscard]] double                     oob_rmse() const { return oob_rmse_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    [[nodiscard]] plot::RPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::RPlot plot_residuals_vs_fitted() const;

 private:
    void fit(const dstruct::DataFrame& data, RandomForestRegressorOptions options);

    Formula    formula_;
    DesignInfo design_;
    RandomForestRegressorOptions options_;
    std::size_t max_features_used_{0};

    std::vector<DecisionTreeRegressor> trees_;
    std::vector<std::vector<bool>>     in_bag_;

    std::vector<std::string> predictor_names_;
    std::vector<double>      training_y_;
    std::vector<double>      fitted_values_;
    double                    oob_r_squared_{0.0};
    double                    oob_rmse_{0.0};
    std::size_t               observations_{0};
};

} // namespace datamunge::stats
