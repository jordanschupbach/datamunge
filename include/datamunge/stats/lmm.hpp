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

struct LMMOptions {
    /// @brief REML (default, matching lme4/nlme) or maximum likelihood.
    bool reml{true};
    /// @brief Population size / generation count for the DifferentialEvolution search over
    ///        the variance-component parameters (theta). The defaults are generous for the
    ///        low dimensionality (q(q+1)/2, q = number of random-effect terms) this search
    ///        always operates in.
    std::size_t de_population_size{40};
    std::size_t de_max_generations{300};
    /// @brief Box-constraint magnitude for theta's free entries (relative to the residual
    ///        standard deviation), passed to the DifferentialEvolution search.
    double theta_bound{5.0};
    std::uint64_t seed{42};
};

/// @brief A linear mixed model fit by (RE)ML from a DataFrame and an lme4-style formula
///        string with a single grouping factor, e.g.:
///
///   LMM model(df, "score ~ x1 + x2 + (1 + x1 | school)");
///   model.print_summary();
///
/// The fixed-effects side of the formula supports everything Formula does (categorical
/// dummy coding, interactions, I(...), poly(...), ...); the random-effects side is limited
/// to an intercept and/or plain numeric predictor names against one grouping factor, e.g.
/// "(1 | group)", "(x1 | group)", "(1 + x1 + x2 | group)", or "(0 + x1 | group)" (no random
/// intercept). Rows with a null value in any column used by the fixed formula, the random
/// terms, or the grouping factor are dropped before fitting.
///
/// Estimation profiles out the fixed effects and residual variance in closed form for a
/// given relative covariance factor Lambda (the free lower-triangular entries of Lambda are
/// "theta"), exploiting that a single grouping factor makes the marginal covariance V
/// block-diagonal by group; theta itself is found by minimizing the profiled (RE)ML
/// criterion with datamunge::optim::DifferentialEvolution.
class LMM {
public:
    LMM(const dstruct::DataFrame& data, const std::string& formula, LMMOptions options = {});

    [[nodiscard]] const std::string& formula_text() const { return formula_text_; }
    [[nodiscard]] const std::string& group_variable() const { return group_variable_; }
    [[nodiscard]] bool                has_random_intercept() const { return random_intercept_; }
    [[nodiscard]] const std::vector<std::string>& random_effect_names() const { return random_effect_names_; }
    [[nodiscard]] bool                is_reml() const { return options_.reml; }
    [[nodiscard]] std::size_t         observations() const { return observations_; }
    [[nodiscard]] std::size_t         num_groups() const { return group_labels_.size(); }
    [[nodiscard]] std::size_t         rank() const { return coefficients_.size(); }

    [[nodiscard]] const std::vector<double>&      coefficients() const { return coefficients_; }
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }
    [[nodiscard]] const std::vector<double>&      standard_errors() const { return standard_errors_; }
    [[nodiscard]] const std::vector<double>&      z_values() const { return z_values_; }
    [[nodiscard]] const std::vector<double>&      p_values() const { return p_values_; }
    [[nodiscard]] const std::vector<double>&      fitted_values() const { return fitted_; }
    [[nodiscard]] const std::vector<double>&      residuals() const { return residuals_; }

    [[nodiscard]] double residual_variance() const { return sigma2_; }
    [[nodiscard]] double residual_std_dev() const { return sigma_; }
    /// @brief Sigma = sigma^2 * Lambda * Lambda' -- the q x q random-effect covariance matrix.
    [[nodiscard]] const linalg::DenseMatrix<double>& random_effect_covariance() const { return random_effect_covariance_; }
    [[nodiscard]] std::vector<double>                random_effect_std_devs() const;
    [[nodiscard]] double                             random_effect_correlation(std::size_t i, std::size_t j) const;

    [[nodiscard]] const std::vector<std::string>&              group_labels() const { return group_labels_; }
    /// @brief BLUP random-effect vectors, one per group, in the same order as group_labels().
    [[nodiscard]] const std::vector<std::vector<double>>&       random_effects() const { return random_effects_; }

    [[nodiscard]] double log_likelihood() const { return -0.5 * deviance_; }
    /// @brief -2 times the (RE)ML criterion actually minimized to fit the model.
    [[nodiscard]] double deviance() const { return deviance_; }
    [[nodiscard]] double aic() const;
    [[nodiscard]] double bic() const;

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Response-scale predictions. Rows whose group was seen while fitting get a
    ///        BLUP-adjusted prediction (X*beta + Z*b_hat); rows with an unseen (or null)
    ///        group value fall back to the population-level fixed-effects-only prediction.
    [[nodiscard]] std::vector<double> predict(const dstruct::DataFrame& newdata) const;

private:
    void fit(const dstruct::DataFrame& data);

    std::string formula_text_;
    std::string group_variable_;
    bool        random_intercept_{true};
    std::vector<std::string> random_slope_vars_;
    std::vector<std::string> random_effect_names_; // "(Intercept)" + slope var names, in Z-column order

    Formula     fixed_formula_;
    DesignInfo  design_;
    LMMOptions  options_;

    linalg::DenseMatrix<double> design_matrix_; // X, rows aligned with response_/group_index_
    std::vector<double>         response_;      // y

    std::vector<std::string>              group_labels_;    // distinct group values, first-seen order
    std::vector<std::size_t>              group_index_;     // per-row index into group_labels_
    std::vector<std::vector<std::size_t>> group_rows_;       // per-group row indices into design_matrix_/response_
    linalg::DenseMatrix<double>           random_effect_design_; // Z, rows aligned with design_matrix_

    std::vector<double>         theta_;       // Lambda's free lower-triangular entries
    linalg::DenseMatrix<double> lambda_;      // q x q relative covariance factor

    std::vector<double> coefficients_;
    std::vector<double> fitted_;
    std::vector<double> residuals_;
    std::vector<double> standard_errors_;
    std::vector<double> z_values_;
    std::vector<double> p_values_;
    linalg::DenseMatrix<double> fixed_effect_covariance_; // sigma^2 * (sum_i X_i' M_i^-1 X_i)^-1

    linalg::DenseMatrix<double> random_effect_covariance_; // sigma^2 * Lambda Lambda'
    std::vector<std::vector<double>> random_effects_;      // BLUPs, one per group

    double      sigma2_{0.0};
    double      sigma_{0.0};
    double      deviance_{0.0};
    std::size_t observations_{0};
};

} // namespace datamunge::stats
