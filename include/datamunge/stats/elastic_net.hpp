#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

struct ElasticNetOptions {
    // 0 = ridge (pure L2), 1 = lasso (pure L1), in between = elastic net.
    double alpha{0.5};
    // Regularization strength in glmnet's parameterization: the objective
    // is (1/2n)*RSS + lambda*alpha*||beta||_1 + lambda*(1-alpha)/2*||beta||_2^2.
    // A negative value (the default) means "choose lambda automatically" by
    // k-fold cross-validation over an internally generated geometric path
    // (glmnet's lambda.min), rather than fitting at a single fixed lambda.
    double        lambda{-1.0};
    std::size_t   n_lambda{100};   // path length when auto-selecting lambda
    std::size_t   cv_folds{5};
    std::size_t   max_iter{10000};
    double        tol{1e-7};
    std::uint64_t seed{42};        // fold-shuffling RNG seed
    // Standardize predictors (mean 0, unit variance) before fitting so the
    // penalty applies fairly across differently-scaled columns; reported
    // coefficients are always unstandardized back to the original scale.
    bool standardize{true};
};

// Elastic net regularized linear regression fit by cyclic coordinate
// descent, using the same formula + DataFrame constructor style as LM:
//
//   ElasticNet model(df, "mpg ~ hp + wt + disp");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// alpha=0 is ridge regression, alpha=1 is lasso; see the Ridge and Lasso
// convenience classes below, which are thin wrappers around this engine.
// Rows with a null value in any column used by the formula are dropped
// before fitting. If lambda is not fixed by the caller, it is chosen by
// k-fold cross-validation (see ElasticNetOptions::lambda).
class ElasticNet {
 public:
    ElasticNet(const dstruct::DataFrame& data, const std::string& formula, ElasticNetOptions options = {});

    [[nodiscard]] const std::string&              formula_text() const { return formula_.text(); }
    [[nodiscard]] bool                            has_intercept() const { return design_.has_intercept; }
    [[nodiscard]] const std::vector<std::string>& predictor_names() const { return predictor_names_; }
    [[nodiscard]] std::size_t                     observations() const { return observations_; }

    [[nodiscard]] double alpha() const { return options_.alpha; }
    // The lambda actually used to produce coefficients() -- either the
    // caller-supplied value or the cross-validation-selected one.
    [[nodiscard]] double lambda() const { return lambda_used_; }
    // True if lambda was chosen automatically (options.lambda was negative).
    [[nodiscard]] bool   lambda_was_selected() const { return lambda_was_selected_; }
    // Nonempty only when lambda was selected automatically: the geometric
    // path of lambda values that was swept, and the mean cross-validated
    // squared error at each of them (same length and order as the path).
    [[nodiscard]] const std::vector<double>& lambda_path() const { return lambda_path_; }
    [[nodiscard]] const std::vector<double>& cv_mean_squared_error() const { return cv_mse_path_; }

    [[nodiscard]] const std::vector<double>& coefficients() const { return coefficients_; }
    [[nodiscard]] double                     intercept() const { return intercept_; }
    [[nodiscard]] std::size_t                non_zero_coefficients() const;

    [[nodiscard]] const std::vector<double>& fitted_values() const { return fitted_; }
    [[nodiscard]] const std::vector<double>& residuals() const { return residuals_; }
    [[nodiscard]] double                     r_squared() const;
    [[nodiscard]] double                     rmse() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

    // Coefficient trace (one line per predictor) across the lambda path in
    // log10(lambda) space. Only meaningful when lambda was auto-selected
    // (throws otherwise, since a fixed-lambda fit has no path).
    [[nodiscard]] plot::ScatterPlot plot_coefficient_path() const;
    // Cross-validated MSE across the lambda path, with the selected lambda
    // marked. Only meaningful when lambda was auto-selected.
    [[nodiscard]] plot::ScatterPlot plot_cv_curve() const;
    [[nodiscard]] plot::ScatterPlot plot_predicted_vs_actual() const;
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_fitted() const;

 protected:
    // Used by Ridge/Lasso to fix alpha while still exposing formula-based
    // construction semantics.
    ElasticNet(const dstruct::DataFrame& data, const std::string& formula, double fixed_alpha,
              ElasticNetOptions options);

 private:
    void fit(const dstruct::DataFrame& data);

    Formula            formula_;
    DesignInfo         design_;
    ElasticNetOptions  options_;

    std::vector<std::string> predictor_names_;
    std::vector<double>      feature_mean_;
    std::vector<double>      feature_scale_;
    double                    y_mean_{0.0};

    std::vector<double> coefficients_; // original scale, aligned with predictor_names_
    double               intercept_{0.0};
    double               lambda_used_{0.0};
    bool                 lambda_was_selected_{false};
    std::vector<double> lambda_path_;
    std::vector<double> cv_mse_path_;
    std::size_t          best_lambda_index_{0}; // index into lambda_path_/cv_mse_path_; only meaningful when auto-selected
    linalg::DenseMatrix<double> path_coefficients_; // [lambda_index][predictor], original scale; empty unless auto-selected

    std::vector<double> fitted_;
    std::vector<double> residuals_;
    std::vector<double> training_y_;
    std::size_t          observations_{0};
};

// Ridge regression (pure L2 penalty, alpha = 0): shrinks all coefficients
// toward zero but never sets them exactly to zero. Same formula +
// DataFrame constructor style; see ElasticNet for the full method set.
struct RidgeOptions {
    double        lambda{-1.0};
    std::size_t   n_lambda{100};
    std::size_t   cv_folds{5};
    std::size_t   max_iter{10000};
    double        tol{1e-7};
    std::uint64_t seed{42};
    bool          standardize{true};
};

class Ridge : public ElasticNet {
 public:
    Ridge(const dstruct::DataFrame& data, const std::string& formula, RidgeOptions options = {});
};

// Lasso regression (pure L1 penalty, alpha = 1): shrinks coefficients
// toward zero and can set them exactly to zero, performing variable
// selection. Same formula + DataFrame constructor style; see ElasticNet
// for the full method set.
struct LassoOptions {
    double        lambda{-1.0};
    std::size_t   n_lambda{100};
    std::size_t   cv_folds{5};
    std::size_t   max_iter{10000};
    double        tol{1e-7};
    std::uint64_t seed{42};
    bool          standardize{true};
};

class Lasso : public ElasticNet {
 public:
    Lasso(const dstruct::DataFrame& data, const std::string& formula, LassoOptions options = {});
};

} // namespace datamunge::stats
