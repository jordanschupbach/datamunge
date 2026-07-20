#pragma once

#include <datamunge/bayes/inla.hpp>
#include <datamunge/dstruct/dataframe.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/stats/formula.hpp>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace datamunge::stats {

enum class INLAMixedModelFamily { Gaussian, Binomial, Poisson };

struct INLAMixedModelOptions {
    INLAMixedModelFamily family{INLAMixedModelFamily::Gaussian};
    bayes::INLAIntegrationStrategy strategy{bayes::INLAIntegrationStrategy::Grid};
    /// @brief Fixed-effects prior std dev (a large, effectively-flat vague prior) -- unlike
    ///        the variance components below, this is a fixed modeling choice, not itself
    ///        integrated over.
    double fixed_effect_prior_sd{1000.0};
    /// @brief Lower bound for the random-effect covariance factor Lambda's diagonal entries
    ///        (must stay strictly positive so Lambda*Lambda' remains invertible -- unlike
    ///        GLMM/LMM's PQL fit, INLA needs a genuine precision matrix at every theta the
    ///        integration visits).
    double lambda_diag_epsilon{1e-3};
    /// @brief Box-constraint magnitude for Lambda's free entries (matches GLMM/LMM's theta_bound).
    double lambda_bound{5.0};
    /// @brief Gaussian family only: half-width, in log-sigma units, of the residual-sigma
    ///        search range around a data-driven initial estimate.
    double log_sigma_search_radius{3.0};
    std::size_t grid_points_per_dim{7};
    double grid_span{4.0};
    std::size_t mode_population_size{40};
    std::size_t mode_max_generations{200};
    std::uint64_t seed{42};
};

/// @brief A mixed model (formula + single grouping factor, matching GLMM/LMM's lme4-style
///        grammar) fit by Integrated Nested Laplace Approximation (bayes::INLA) rather than
///        GLMM's penalized quasi-likelihood or LMM's profiled REML -- a genuinely Bayesian
///        alternative that returns real posterior means/sds (including for the variance
///        components themselves) instead of point estimates + asymptotic standard errors.
///
///   INLAMixedModel model(df, "y ~ x1 + x2 + (1 + x1 | group)", {INLAMixedModelFamily::Binomial});
///   model.print_summary();
///
/// Supports Gaussian (identity link), Binomial (logit link), and Poisson (log link) -- the
/// Gaussian case additionally treats the residual variance as a hyperparameter integrated
/// over like any other, which GLMM/LMM's frequentist fits don't do.
///
/// Internally builds a single latent field x = [fixed effects; one random-effect block per
/// group] and a corresponding block-diagonal precision matrix (fixed effects get a fixed
/// vague-prior precision; each group's random-effect block gets the same covariance,
/// parameterized via the packed Cholesky factor Lambda exactly as GLMM/LMM do), then delegates
/// to bayes::INLA::fit(). See bayes/inla.hpp for the underlying algorithm and its documented
/// limitations (dense linear algebra, Gaussian latent-marginal strategy, practical grid-theta-
/// dimension cap).
class INLAMixedModel {
public:
    INLAMixedModel(const dstruct::DataFrame& data, const std::string& formula, INLAMixedModelOptions options = {});

    [[nodiscard]] const std::string& formula_text() const { return formula_text_; }
    [[nodiscard]] std::string family() const;
    [[nodiscard]] const std::string& group_variable() const { return group_variable_; }
    [[nodiscard]] const std::vector<std::string>& random_effect_names() const { return random_effect_names_; }
    [[nodiscard]] std::size_t observations() const { return observations_; }
    [[nodiscard]] std::size_t num_groups() const { return group_labels_.size(); }
    [[nodiscard]] const std::vector<std::string>& group_labels() const { return group_labels_; }

    [[nodiscard]] const std::vector<double>&      fixed_effects_mean() const { return fixed_effects_mean_; }
    [[nodiscard]] const std::vector<double>&      fixed_effects_sd() const { return fixed_effects_sd_; }
    [[nodiscard]] const std::vector<std::string>& coefficient_names() const { return design_.coefficient_names; }

    /// @brief Posterior mean/sd BLUP-like random-effect vectors, one per group (in
    ///        group_labels() order), on the linear-predictor scale.
    [[nodiscard]] const std::vector<std::vector<double>>& random_effects_mean() const { return random_effects_mean_; }
    [[nodiscard]] const std::vector<std::vector<double>>& random_effects_sd() const { return random_effects_sd_; }

    /// @brief Random-effect standard deviations, evaluated at theta's posterior mean.
    [[nodiscard]] std::vector<double> random_effect_std_devs() const;
    /// @brief Gaussian family only: residual standard deviation at theta's posterior mean.
    ///        Throws std::logic_error for Binomial/Poisson (dispersion fixed at 1).
    [[nodiscard]] double residual_std_dev() const;

    /// @brief log p(y) approximated by the same INLA machinery used to fit the model.
    [[nodiscard]] double log_marginal_likelihood() const { return log_marginal_likelihood_; }

    [[nodiscard]] std::string summary() const;
    void                      print_summary(std::ostream& os) const;
    void                      print_summary() const;

    /// @brief Response-scale predictions (posterior-mean latent field, back-transformed
    ///        through the inverse link). Rows whose group was seen while fitting get a
    ///        random-effect-adjusted prediction; unseen/null groups fall back to the
    ///        population-level (fixed-effects-only) prediction.
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
    INLAMixedModelOptions options_;

    std::vector<std::string>              group_labels_;
    std::vector<std::size_t>              group_index_;
    std::vector<std::vector<std::size_t>> group_rows_;

    std::vector<double> fixed_effects_mean_;
    std::vector<double> fixed_effects_sd_;
    std::vector<std::vector<double>> random_effects_mean_;
    std::vector<std::vector<double>> random_effects_sd_;
    std::vector<double> theta_mean_;
    std::size_t lambda_free_count_{0};

    double      log_marginal_likelihood_{0.0};
    std::size_t observations_{0};
};

} // namespace datamunge::stats
