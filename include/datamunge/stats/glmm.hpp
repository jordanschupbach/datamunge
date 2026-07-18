#pragma once

#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

enum class GLMMFamily { Binomial, Poisson };

struct GLMMOptions {
    GLMMFamily family{GLMMFamily::Binomial};
    /// @brief Outer penalized-quasi-likelihood (IRLS re-linearization) iterations.
    std::size_t max_iterations{20};
    /// @brief Stops the outer loop when the fixed effects change by less than this (in
    ///        infinity norm) between iterations.
    double tol{1e-6};
    /// @brief Population size / generation count for the DifferentialEvolution search over
    ///        the variance-component parameters (theta), run once per outer iteration.
    std::size_t de_population_size{30};
    std::size_t de_max_generations{150};
    /// @brief Box-constraint magnitude for theta's free entries.
    double theta_bound{5.0};
    std::uint64_t seed{42};
};

/// @brief A generalized linear mixed model fit by penalized quasi-likelihood (PQL; Breslow
///        & Clayton 1993) from a DataFrame and an lme4-style formula string with a single
///        grouping factor, e.g.:
///
///   GLMM model(df, "y ~ x1 + x2 + (1 + x1 | group)", {GLMMFamily::Binomial});
///   model.print_summary();
///
/// Supports Binomial (logit link) and Poisson (log link) -- the two families that actually
/// motivate a GLMM over a plain LMM (a Gaussian GLMM is exactly an LMM; use LMM directly for
/// that case). The dispersion parameter is fixed at 1, matching the canonical assumption for
/// both families.
///
/// PQL works by repeatedly linearizing the model around the current fit (exactly as GLM's
/// IRLS does) to form a Gaussian "working response" and per-observation weights, then fitting
/// a weighted linear mixed model to that pseudo-data -- reusing the same block-diagonal
/// per-group Cholesky trick as LMM, generalized to per-observation weights, with theta (the
/// variance-component parameters) re-optimized via datamunge::optim::DifferentialEvolution at
/// every outer iteration. Like all PQL-based fits, this is an approximation (most accurate
/// for reasonably large cluster sizes) rather than exact marginal (RE)ML.
///
/// The fixed-effects side of the formula supports everything Formula does; the random-effects
/// side follows the same grammar as LMM. Rows with a null value in any column used by the
/// fixed formula, the random terms, or the grouping factor are dropped before fitting.
class GLMM {
public:
    GLMM(const dstruct::DataFrame& data, const std::string& formula, GLMMOptions options = {});

    [[nodiscard]] const std::string& formula_text() const { return formula_text_; }
    [[nodiscard]] std::string        family() const;
    [[nodiscard]] const std::string& group_variable() const { return group_variable_; }
    [[nodiscard]] bool                has_random_intercept() const { return random_intercept_; }
    [[nodiscard]] const std::vector<std::string>& random_effect_names() const { return random_effect_names_; }
    [[nodiscard]] std::size_t         observations() const { return observations_; }
    [[nodiscard]] std::size_t         num_groups() const { return group_labels_.size(); }
    [[nodiscard]] std::size_t         rank() const { return coefficients_.size(); }
    [[nodiscard]] std::size_t         iterations() const { return iterations_used_; }

    [[nodiscard]] const std::vector<double>&      coefficients() const { return coefficients_; }
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    [[nodiscard]] const std::vector<double>&      standard_errors() const { return standard_errors_; }
    [[nodiscard]] const std::vector<double>&      z_values() const { return z_values_; }
    [[nodiscard]] const std::vector<double>&      p_values() const { return p_values_; }
    [[nodiscard]] const std::vector<double>&      fitted_values() const { return fitted_; } // response scale

    [[nodiscard]] const linalg::DenseMatrix<double>& random_effect_covariance() const { return random_effect_covariance_; }
    [[nodiscard]] std::vector<double>                random_effect_std_devs() const;
    [[nodiscard]] double                             random_effect_correlation(std::size_t i, std::size_t j) const;

    [[nodiscard]] const std::vector<std::string>&        group_labels() const { return group_labels_; }
    /// @brief BLUP random-effect vectors (on the linear-predictor scale), one per group, in
    ///        the same order as group_labels().
    [[nodiscard]] const std::vector<std::vector<double>>& random_effects() const { return random_effects_; }

    [[nodiscard]] double deviance() const { return deviance_; }
    [[nodiscard]] double aic() const;
    [[nodiscard]] double bic() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Response-scale predictions (back-transformed through the inverse link). Rows
    ///        whose group was seen while fitting get a BLUP-adjusted prediction; rows with an
    ///        unseen (or null) group value fall back to the population-level prediction.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::string formula_text_;
    std::string group_variable_;
    bool        random_intercept_{true};
    std::vector<std::string> random_slope_vars_;
    std::vector<std::string> random_effect_names_;

    Formula     fixed_formula_;
    DesignInfo  design_;
    GLMMOptions options_;

    linalg::DenseMatrix<double> design_matrix_; // X
    std::vector<double>         response_;      // y (raw scale)

    std::vector<std::string>              group_labels_;
    std::vector<std::size_t>              group_index_;
    std::vector<std::vector<std::size_t>> group_rows_;
    linalg::DenseMatrix<double>           random_effect_design_; // Z

    linalg::DenseMatrix<double> lambda_; // q x q relative covariance factor

    std::vector<double> coefficients_;
    std::vector<double> linear_predictors_;
    std::vector<double> fitted_; // response scale (mu)
    std::vector<double> standard_errors_;
    std::vector<double> z_values_;
    std::vector<double> p_values_;
    linalg::DenseMatrix<double> fixed_effect_covariance_;

    linalg::DenseMatrix<double> random_effect_covariance_; // Lambda Lambda' (phi == 1)
    std::vector<std::vector<double>> random_effects_;

    double      deviance_{0.0};
    std::size_t observations_{0};
    std::size_t iterations_used_{0};
};

} // namespace datamunge::stats
