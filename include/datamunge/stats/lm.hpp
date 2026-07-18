#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/plot/plot.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

namespace datamunge::stats {

struct LmOptions {
    std::optional<std::string> weights_column; // enables weighted least squares when set
};

enum class PredictionInterval { None, Confidence, Prediction };

struct LmPrediction {
    std::vector<double> fit;
    std::vector<double> se_fit; // empty unless an interval was requested
    std::vector<double> lower;  // empty unless an interval was requested
    std::vector<double> upper;  // empty unless an interval was requested
};

struct AnovaRow {
    std::string term;
    std::size_t degrees_of_freedom{0};
    double      sum_sq{0.0};
    double      mean_sq{0.0};
    double      f_value{0.0};
    double      p_value{0.0};
};

// An R-`lm()`-style ordinary/weighted least-squares linear model fit from a
// DataFrame and a formula string, e.g.:
//
//   LM model(df, "mpg ~ hp + wt + factor_col");
//   model.print_summary();
//   auto preds = model.predict(newdata);
//
// Rows with a null value in any column used by the formula are dropped
// before fitting (matching R's na.omit default).
class LM {
 public:
    LM(const dstruct::DataFrame& data, const std::string& formula, LmOptions options = {});

    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    [[nodiscard]] bool               has_intercept() const { return design_.has_intercept; }
    [[nodiscard]] std::size_t        observations() const { return observations_; }
    [[nodiscard]] std::size_t        rank() const { return coefficients_.size(); }
    [[nodiscard]] std::size_t        degrees_of_freedom() const { return degrees_of_freedom_; }

    [[nodiscard]] const std::vector<double>&      coefficients() const { return coefficients_; }
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    [[nodiscard]] const std::vector<double>&      fitted_values() const { return fitted_; }
    [[nodiscard]] const std::vector<double>&      residuals() const { return residuals_; }
    [[nodiscard]] const std::vector<double>&      standard_errors() const { return standard_errors_; }
    [[nodiscard]] const std::vector<double>&      t_values() const { return t_values_; }
    [[nodiscard]] const std::vector<double>&      p_values() const { return p_values_; }
    [[nodiscard]] const linalg::DenseMatrix<double>& covariance() const { return covariance_; }

    [[nodiscard]] double r_squared() const { return r_squared_; }
    [[nodiscard]] double adjusted_r_squared() const { return adjusted_r_squared_; }
    [[nodiscard]] double sigma() const { return sigma_; }
    [[nodiscard]] double f_statistic() const { return f_statistic_; }
    [[nodiscard]] double f_p_value() const { return f_p_value_; }

    struct Interval { double lower; double upper; };
    [[nodiscard]] std::vector<Interval> confidence_intervals(double level = 0.95) const;

    // Diagnostics
    [[nodiscard]] const std::vector<double>& leverage() const { return leverage_; }
    [[nodiscard]] const std::vector<double>& standardized_residuals() const { return standardized_residuals_; }
    [[nodiscard]] const std::vector<double>& studentized_residuals() const { return studentized_residuals_; }
    [[nodiscard]] const std::vector<double>& cooks_distance() const { return cooks_distance_; }

    [[nodiscard]] std::vector<AnovaRow> anova() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    // Prediction against a new DataFrame using the same resolved formula.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] LmPrediction        predict(const dstruct::DataFrame& newdata,
                                              PredictionInterval        interval,
                                              double                    level = 0.95) const;
    [[nodiscard]] dstruct::DataFrame  predict_frame(const dstruct::DataFrame& newdata,
                                                    PredictionInterval        interval = PredictionInterval::None,
                                                    double                    level = 0.95) const;

    // Diagnostic plots (mirrors R's plot.lm panels 1-4).
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_fitted() const;
    [[nodiscard]] plot::ScatterPlot plot_normal_qq() const;
    [[nodiscard]] plot::ScatterPlot plot_scale_location() const;
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_leverage() const;

    // Saves all four diagnostic plots as "<path_prefix>_<name>.svg".
    void save_diagnostic_plots(const std::string& path_prefix) const;

 private:
    void fit(const dstruct::DataFrame& data, LmOptions options);
    void compute_diagnostics();
    void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

    Formula                     formula_;
    DesignInfo                  design_;
    LmOptions                   options_;

    linalg::DenseMatrix<double> design_matrix_; // weighted design used to fit (Xw)
    std::vector<double>         response_;      // weighted response used to fit (yw)

    std::vector<double>         coefficients_;
    std::vector<double>         fitted_;
    std::vector<double>         residuals_;
    std::vector<double>         standard_errors_;
    std::vector<double>         t_values_;
    std::vector<double>         p_values_;
    linalg::DenseMatrix<double> covariance_; // sigma^2 * (X^T X)^-1 (or weighted equivalent)
    linalg::DenseMatrix<double> xtx_inverse_; // (X^T W X)^-1, unscaled — used for leverage/prediction variance

    double      r_squared_{0.0};
    double      adjusted_r_squared_{0.0};
    double      sigma_{0.0};
    double      sigma2_{0.0};
    double      f_statistic_{0.0};
    double      f_p_value_{0.0};
    std::size_t observations_{0};
    std::size_t degrees_of_freedom_{0};

    std::vector<double> leverage_;
    std::vector<double> standardized_residuals_;
    std::vector<double> studentized_residuals_;
    std::vector<double> cooks_distance_;
    std::vector<double> weights_; // empty for unweighted fits
};

} // namespace datamunge::stats
