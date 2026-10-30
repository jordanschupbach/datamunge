#pragma once

/// @file normal_models.hpp
/// @brief Exact posteriors for Normal models with known observation variances: a common mean,
///        and the hierarchical (random-effects) model behind "eight schools" and Bayesian
///        meta-analysis.
///
/// Both rely on conjugacy. For the hierarchical model
/// \f$y_j\sim N(\theta_j,\sigma_j^2)\f$, \f$\theta_j\sim N(\mu,\tau^2)\f$,
/// \f$\mu\sim N(\mu_0,s_0^2)\f$, \f$\tau\sim\text{half-Cauchy}(0,c)\f$, the \f$\theta_j\f$ and
/// \f$\mu\f$ integrate out in closed form (Gelman et al., BDA3 §5.4), leaving a one-dimensional
/// integral over \f$\tau\f$ that is done by quadrature. These serve as ground truth for MCMC.

#include <cstddef>
#include <vector>

namespace datamunge::bayes {

/// Closed-form Normal posterior.
struct NormalPosterior {
    /// @brief Posterior mean.
    double mean{0.0};
    /// @brief Posterior standard deviation.
    double sd{0.0};
};

/// @brief Exact posterior of a common mean \f$\mu\f$ given \f$y_i\sim N(\mu,\sigma_i^2)\f$ with
///        known \f$\sigma_i\f$ and prior \f$\mu\sim N(\mu_0,\tau_0^2)\f$.
///
/// Precisions add: \f$1/s^2=1/\tau_0^2+\sum_i 1/\sigma_i^2\f$ and
/// \f$m=s^2(\mu_0/\tau_0^2+\sum_i y_i/\sigma_i^2)\f$. Pass \f$\tau_0=\infty\f$ for a flat prior.
/// @param y Observations.
/// @param sigma Known standard deviation of each observation (same length as @p y, positive).
/// @param mu0 Prior mean.
/// @param tau0 Prior standard deviation (positive, may be infinite).
/// @return The posterior mean and standard deviation.
/// @throws std::invalid_argument for mismatched lengths, empty data or non-positive scales.
NormalPosterior normal_mean_posterior(const std::vector<double>& y, const std::vector<double>& sigma, double mu0,
                                      double tau0);

/// @brief Log marginal likelihood \f$\log p(y\mid\tau)\f$ of the hierarchical Normal model at a
///        fixed between-group standard deviation, with \f$\theta\f$ and \f$\mu\f$ integrated out.
///
/// Given \f$\tau\f$, \f$y_j\sim N(\mu,V_j)\f$ independently with \f$V_j=\sigma_j^2+\tau^2\f$.
/// Completing the square in \f$\mu\f$ with posterior mean \f$m\f$ and sd \f$s\f$ gives
/// \f$\log(s/s_0)-\tfrac12\sum_j\log(2\pi V_j)-\tfrac12\bigl(\sum_j y_j^2/V_j+\mu_0^2/s_0^2-m^2/s^2\bigr)\f$.
/// \f$\tau=0\f$ is complete pooling.
/// @param y Group estimates.
/// @param sigma Their known standard errors.
/// @param mu0 Prior mean of \f$\mu\f$.
/// @param mu_sd Prior standard deviation of \f$\mu\f$ (positive, finite).
/// @param tau Between-group standard deviation (non-negative).
/// @return \f$\log p(y\mid\tau)\f$.
/// @throws std::invalid_argument for mismatched lengths, empty data or invalid scales.
double hierarchical_normal_log_marginal(const std::vector<double>& y, const std::vector<double>& sigma, double mu0,
                                        double mu_sd, double tau);

/// Exact posterior summaries of the hierarchical Normal model.
struct HierarchicalNormalPosterior {
    /// @brief Quadrature nodes for \f$\tau\f$ (increasing; equally spaced in prior probability).
    std::vector<double> tau;
    /// @brief Posterior density \f$p(\tau\mid y)\f$ at each node.
    std::vector<double> tau_density;
    /// @brief Posterior mean of \f$\mu\f$.
    double mu_mean{0.0};
    /// @brief Posterior standard deviation of \f$\mu\f$.
    double mu_sd{0.0};
    /// @brief Posterior mean of \f$\tau\f$.
    double tau_mean{0.0};
    /// @brief Posterior standard deviation of \f$\tau\f$.
    double tau_sd{0.0};
    /// @brief Posterior 5% quantile of \f$\tau\f$.
    double tau_q05{0.0};
    /// @brief Posterior median of \f$\tau\f$.
    double tau_q50{0.0};
    /// @brief Posterior 95% quantile of \f$\tau\f$.
    double tau_q95{0.0};
    /// @brief Posterior mean of each \f$\theta_j\f$.
    std::vector<double> theta_mean;
    /// @brief Posterior standard deviation of each \f$\theta_j\f$.
    std::vector<double> theta_sd;
    /// @brief Posterior mean of each shrinkage factor \f$B_j=\sigma_j^2/(\sigma_j^2+\tau^2)\f$.
    std::vector<double> shrinkage;
    /// @brief Log evidence \f$\log p(y)\f$ under the half-Cauchy prior on \f$\tau\f$.
    double log_evidence{0.0};
};

/// @brief Exact posterior of the hierarchical Normal model with \f$\mu\sim N(\mu_0,s_0^2)\f$ and
///        \f$\tau\sim\text{half-Cauchy}(0,c)\f$.
///
/// Substituting \f$\tau=c\tan(\pi u/2)\f$ turns the half-Cauchy prior into \f$u\sim U(0,1)\f$, so
/// every expectation is \f$\int_0^1 p(y\mid\tau(u))\,g(\tau(u))\,du\big/\int_0^1p(y\mid\tau(u))\,du\f$,
/// evaluated with the midpoint rule. Given \f$\tau\f$, \f$\mu\f$ is Normal (normal_mean_posterior()
/// with variances \f$V_j\f$) and \f$\theta_j\f$ is Normal with mean
/// \f$B_j\mu+(1-B_j)y_j\f$ and variance \f$(1-B_j)\sigma_j^2\f$ given \f$\mu\f$; the moments of
/// \f$\theta_j\f$ follow from the laws of total expectation and variance.
/// @param y Group estimates.
/// @param sigma Their known standard errors.
/// @param mu0 Prior mean of \f$\mu\f$.
/// @param mu_sd Prior standard deviation of \f$\mu\f$ (positive, finite).
/// @param tau_scale Half-Cauchy scale \f$c\f$ (positive).
/// @param grid_points Number of quadrature nodes (at least 10).
/// @return Posterior summaries and the density of \f$\tau\f$.
/// @throws std::invalid_argument for invalid inputs.
HierarchicalNormalPosterior hierarchical_normal_posterior(const std::vector<double>& y,
                                                          const std::vector<double>& sigma, double mu0,
                                                          double mu_sd, double tau_scale,
                                                          std::size_t grid_points = 4000);

} // namespace datamunge::bayes
