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

// Each family is paired with its canonical link, matching R's glm()
// defaults exactly: gaussian -> identity, binomial -> logit,
// poisson -> log, Gamma -> inverse.
enum class GLMFamily { Gaussian, Binomial, Poisson, Gamma };

struct GLMOptions {
    GLMFamily                   family{GLMFamily::Gaussian};
    std::optional<std::string> weights_column; // prior weights, as in LM's weighted least squares
    std::size_t                  max_iter{25};
    double                        tol{1e-8};
};

enum class GLMPredictionInterval { None, Confidence };

struct GLMPrediction {
    std::vector<double> fit;    // response scale (back-transformed through the inverse link)
    std::vector<double> se_fit; // response scale, via the delta method
    std::vector<double> lower;  // response scale; computed on the link scale then back-transformed, so
    std::vector<double> upper;  // e.g. a binomial interval always stays inside [0, 1]
};

// A generalized linear model: extends LM to response distributions from
// the exponential family (gaussian, binomial, poisson, Gamma), connected
// to the linear predictor by a link function, fit by iteratively
// reweighted least squares (IRLS) -- the standard maximum-likelihood
// algorithm for GLMs. Same formula + DataFrame constructor style as LM:
//
//   GLM model(data, "made_purchase ~ age + income", GLMOptions{.family = GLMFamily::Binomial});
//   model.print_summary();
//   auto probabilities = model.predict(newdata);
//
// The response must already be on the scale the family expects: binomial
// requires values in {0, 1}, poisson requires non-negative values, Gamma
// requires strictly positive values (this implementation does not
// auto-encode a categorical response the way LDA/SVM/classifiers do).
// Rows with a null value in any column used by the formula are dropped
// before fitting (matching R's na.omit default).
class GLM {
 public:
    GLM(const dstruct::DataFrame& data, const std::string& formula, GLMOptions options = {});

    [[nodiscard]] const std::string& formula_text() const { return formula_.text(); }
    [[nodiscard]] GLMFamily          family() const { return options_.family; }
    [[nodiscard]] bool               has_intercept() const { return design_.has_intercept; }
    [[nodiscard]] std::size_t        observations() const { return observations_; }
    [[nodiscard]] std::size_t        rank() const { return coefficients_.size(); }
    [[nodiscard]] std::size_t        degrees_of_freedom() const { return degrees_of_freedom_; }

    [[nodiscard]] const std::vector<double>&      coefficients() const { return coefficients_; }
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    [[nodiscard]] const std::vector<double>&      fitted_values() const { return fitted_; }       // response scale
    [[nodiscard]] const std::vector<double>&      linear_predictors() const { return linear_predictor_; }
    [[nodiscard]] const std::vector<double>&      residuals() const { return deviance_residuals_; } // raw deviance
    [[nodiscard]] const std::vector<double>&      pearson_residuals() const { return pearson_residuals_; }
    [[nodiscard]] const std::vector<double>&      standardized_residuals() const { return standardized_residuals_; }
    [[nodiscard]] const std::vector<double>&      leverage() const { return leverage_; }
    [[nodiscard]] const std::vector<double>&      standard_errors() const { return standard_errors_; }
    // coefficient / standard error -- a z statistic for binomial/poisson (fixed dispersion), a t statistic for
    // gaussian/Gamma (estimated dispersion); p_values() uses the matching reference distribution either way.
    [[nodiscard]] const std::vector<double>&      test_statistics() const { return test_statistics_; }
    [[nodiscard]] const std::vector<double>&      p_values() const { return p_values_; }
    [[nodiscard]] const linalg::DenseMatrix<double>& covariance() const { return covariance_; }

    [[nodiscard]] double deviance() const { return deviance_; }
    [[nodiscard]] double null_deviance() const { return null_deviance_; }
    // 1.0 (fixed) for binomial/poisson; the Pearson X^2/df estimate for gaussian/Gamma.
    [[nodiscard]] double dispersion() const { return dispersion_; }
    [[nodiscard]] double aic() const { return aic_; }

    struct Interval { double lower; double upper; };
    [[nodiscard]] std::vector<Interval> confidence_intervals(double level = 0.95) const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    // Response-scale predictions (back-transformed through the inverse link) by default.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;
    [[nodiscard]] GLMPrediction       predict(const dstruct::DataFrame& newdata, GLMPredictionInterval interval,
                                              double level = 0.95) const;
    [[nodiscard]] dstruct::DataFrame  predict_frame(const dstruct::DataFrame& newdata,
                                                    GLMPredictionInterval     interval = GLMPredictionInterval::None,
                                                    double                    level = 0.95) const;

    // Diagnostic plots using standardized deviance residuals (mirrors R's plot.glm, itself close to plot.lm).
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_fitted() const;
    [[nodiscard]] plot::ScatterPlot plot_normal_qq() const;
    [[nodiscard]] plot::ScatterPlot plot_scale_location() const;
    [[nodiscard]] plot::ScatterPlot plot_residuals_vs_leverage() const;
    void                             save_diagnostic_plots(const std::string& path_prefix) const;

 private:
    void fit(const dstruct::DataFrame& data, GLMOptions options);
    void validate_categorical_levels(const dstruct::DataFrame& newdata) const;

    Formula     formula_;
    DesignInfo  design_;
    GLMOptions  options_;

    linalg::DenseMatrix<double> design_matrix_; // unweighted, original scale
    std::vector<double>          response_;      // original scale
    std::vector<double>          prior_weights_;  // empty when no weights_column

    std::vector<double> coefficients_;
    std::vector<double> fitted_;            // mu
    std::vector<double> linear_predictor_;  // eta
    std::vector<double> deviance_residuals_;
    std::vector<double> pearson_residuals_;
    std::vector<double> standardized_residuals_;
    std::vector<double> leverage_;
    std::vector<double> standard_errors_;
    std::vector<double> test_statistics_;
    std::vector<double> p_values_;
    linalg::DenseMatrix<double> covariance_;   // dispersion * (X^T W X)^-1
    linalg::DenseMatrix<double> xtx_inverse_;  // (X^T W X)^-1 (final IRLS weights), unscaled by dispersion
    std::vector<double>          final_weights_; // final IRLS weights (includes prior weights), for prediction SEs

    double      deviance_{0.0};
    double      null_deviance_{0.0};
    double      dispersion_{1.0};
    double      aic_{0.0};
    std::size_t observations_{0};
    std::size_t degrees_of_freedom_{0};
};

} // namespace datamunge::stats
