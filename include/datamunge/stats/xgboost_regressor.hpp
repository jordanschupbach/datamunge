#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/detail/xgboost_tree.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct XGBoostRegressorOptions {
    std::size_t   n_trees{100};
    double        learning_rate{0.3}; // XGBoost's default eta -- higher than plain GBM's, since regularization
                                       // (rather than shrinkage alone) controls overfitting
    std::size_t   max_depth{6};       // XGBoost's default -- deeper than GBM's, for the same reason
    double        lambda{1.0};        // L2 regularization on leaf weights (XGBoost default)
    double        alpha{0.0};         // L1 regularization on leaf weights (XGBoost default)
    double        gamma{0.0};         // minimum gain required to keep a split (XGBoost default)
    double        min_child_weight{1.0};
    std::size_t   min_samples_leaf{1};
    double        subsample{1.0};
    double        colsample_bytree{1.0};
    std::uint64_t seed{42};
};

// XGBoost-style regularized gradient boosting regressor: a sequence of
// trees, each grown by maximizing a second-order (gradient + Hessian)
// regularized gain rather than plain variance reduction, and each leaf set
// to its Newton step rather than a simple mean -- see
// datamunge::stats::detail::XGBoostTree for the exact objective. For
// squared-error loss the Hessian is uniformly 1, so with lambda = alpha =
// gamma = 0 this reduces to (and can be validated against) plain gradient
// boosting. Same formula + DataFrame constructor style as the other stats
// classes:
//
//   XGBoostRegressor model(iris, "Petal.Length ~ Sepal.Length + Sepal.Width");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// As with GBMRegressor, fitted_values()/r_squared()/rmse() are ordinary
// (in-sample) training metrics; use training_deviance() to see the fit
// evolve round by round.
class XGBoostRegressor {
 public:
    XGBoostRegressor(const dstruct::DataFrame& data, const std::string& formula,
                     XGBoostRegressorOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }
    [[nodiscard]] std::size_t                     n_trees() const { return trees_.size(); }

    // Gain-based importance summed across all trees, normalized to sum to 1.
    [[nodiscard]] std::vector<double> feature_importance() const;

    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;
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

    Formula                  formula_;
    DesignInfo               design_;
    XGBoostRegressorOptions  options_;

    std::vector<std::string>            predictor_names_;
    std::vector<detail::XGBoostTree>    trees_;
    double                               base_score_{0.0};

    std::vector<double> training_y_;
    std::vector<double> fitted_;
    std::vector<double> training_deviance_;
    std::size_t          observations_{0};
};

} // namespace datamunge::stats
