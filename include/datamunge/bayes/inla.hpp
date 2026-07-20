#pragma once

#include <datamunge/bayes/inla_integration_strategy.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::bayes {

/// @brief A single observation's likelihood contribution log p(y | eta), where eta is that
///        observation's linear predictor, plus the first two derivatives w.r.t. eta needed
///        for Newton-Raphson mode-finding: score = d/deta log p(y|eta), and curvature =
///        -d^2/deta^2 log p(y|eta) (non-negative for the canonical-link GLM families this is
///        meant for -- Gaussian/Binomial/Poisson -- since their log-likelihoods are concave
///        in eta).
struct INLALikelihoodPoint {
    double log_density{0.0};
    double score{0.0};
    double curvature{0.0};
};

/// @brief Takes theta too (not just y and eta) because some likelihoods have their own
///        hyperparameters -- e.g. Gaussian's residual sigma -- that are themselves part of
///        theta and get integrated over exactly like the latent-field precision parameters.
///        Likelihoods with no such hyperparameter (Binomial, Poisson) simply ignore it.
using INLALikelihoodFn = std::function<INLALikelihoodPoint(double y, double eta, const std::vector<double>& theta)>;
/// @brief Builds the n_latent x n_latent precision matrix Q(theta) of the latent Gaussian
///        field x | theta ~ N(0, Q(theta)^-1). Must be symmetric positive definite for every
///        theta the integration strategy visits.
using INLAPrecisionFn = std::function<linalg::DenseMatrix<double>(const std::vector<double>& theta)>;
/// @brief log pi(theta), the hyperparameter prior density, evaluated on whatever scale the
///        caller chose for theta (e.g. log-precision).
using INLALogPriorFn = std::function<double(const std::vector<double>& theta)>;

struct INLAOptions {
    INLAIntegrationStrategy strategy{INLAIntegrationStrategy::Grid};
    std::size_t newton_max_iterations{100};
    double newton_tolerance{1e-8};
    /// @brief Grid points along each theta axis; must be odd (so the mode itself is a grid
    ///        point). Total grid size is grid_points_per_dim^theta_dim -- only practical for
    ///        small theta_dim (fit() throws above kMaxGridThetaDim, see inla.cpp).
    std::size_t grid_points_per_dim{7};
    /// @brief Half-width of the grid along each (whitened) axis, in posterior-SD units.
    double grid_span{4.0};
    /// @brief Differential-evolution search (matching GLMM/LMM's theta search) used to find
    ///        the mode of theta's Laplace-approximated posterior.
    std::size_t mode_population_size{40};
    std::size_t mode_max_generations{200};
    std::uint64_t seed{42};
};

struct INLAResult {
    /// @brief Posterior mean/sd of every latent field component (fixed and random effects
    ///        alike -- whatever the caller's design matrix columns represent), as a mixture
    ///        over the theta grid of the per-theta Gaussian (Laplace) approximation.
    std::vector<double> latent_mean;
    std::vector<double> latent_sd;
    std::vector<double> theta_mean;
    std::vector<double> theta_sd;
    /// @brief log p(y) approximated by the same grid/Laplace machinery used for theta's
    ///        posterior -- usable for INLA-style model comparison.
    double log_marginal_likelihood{0.0};
    std::size_t theta_dim{0};
    /// @brief Diagnostics: every grid point visited (flattened, theta_dim entries each) and
    ///        its normalized combination weight, in matching order.
    std::vector<double> theta_grid;
    std::vector<double> theta_grid_weights;
};

/// @brief Integrated Nested Laplace Approximation (Rue, Martino & Chopin, 2009) for a latent
///        Gaussian model:
///
///   y_i | eta_i        ~ likelihood (via @p likelihood, canonical-link GLM families)
///   eta = design * x     (x is the full latent field: fixed effects, random effects, ... --
///                          whatever the caller's design matrix columns encode)
///   x | theta           ~ N(0, precision(theta)^-1)
///   theta                ~ exp(log_prior)
///
/// For each theta visited, the latent field's mode x*(theta) is found via Newton-Raphson on
/// the joint log-density (the "Laplace approximation" step), giving a Gaussian approximation
/// to x | y, theta with mean x*(theta) and covariance H*(theta)^-1. theta's own marginal
/// posterior is then approximated via Laplace's method (evaluating the Gaussian
/// approximation at its own mode) and integrated numerically (grid or empirical-Bayes, see
/// INLAIntegrationStrategy) to combine the per-theta Gaussians into final latent-field
/// marginal means/sds.
///
/// This implements INLA's "Gaussian strategy" for the latent field marginals -- i.e. no
/// additional nested skew-correction Laplace step is applied to individual x_i | y, theta
/// marginals beyond the Gaussian approximation at the mode. Real INLA's default ("simplified
/// Laplace") strategy adds that correction for better accuracy on skewed marginals; omitting
/// it here keeps the implementation tractable while still capturing INLA's central idea
/// (integrating over hyperparameter uncertainty via nested Laplace approximations) correctly.
///
/// Uses dense linear algebra throughout (matching this library's DenseMatrix/Cholesky
/// convention elsewhere) rather than R-INLA's sparse GMRF solvers, so it's intended for
/// modest latent-field sizes (up to a few hundred/thousand components), not the huge sparse
/// spatial models R-INLA is built to scale to.
class INLA {
public:
    explicit INLA(INLAOptions options = {});

    /// @brief Fits the model described above. @p design is n_obs x n_latent; @p y has length
    ///        n_obs. @p theta_lower/@p theta_upper box-constrain the differential-evolution
    ///        mode search; @p theta_init seeds it and determines theta's dimensionality.
    [[nodiscard]] INLAResult fit(const linalg::DenseMatrix<double>& design, const std::vector<double>& y,
                                  const INLALikelihoodFn& likelihood, const INLAPrecisionFn& precision,
                                  const INLALogPriorFn& log_prior, const std::vector<double>& theta_lower,
                                  const std::vector<double>& theta_upper, const std::vector<double>& theta_init) const;

private:
    INLAOptions options_;
};

// ---- Convenience likelihood builders for the common canonical-link GLM families ----

/// @brief Gaussian likelihood (identity link), y_i | eta_i ~ N(eta_i, sigma^2), where sigma is
///        extracted from theta by @p sigma_of_theta -- pass e.g. `[](auto&){ return 2.0; }`
///        for a fixed known sigma, or read a dedicated theta entry (log scale recommended for
///        positivity, e.g. `[](auto& theta){ return std::exp(theta.back()); }`) to integrate
///        over residual-variance uncertainty like any other hyperparameter.
INLALikelihoodFn inla_gaussian_likelihood(std::function<double(const std::vector<double>&)> sigma_of_theta);
/// @brief Binomial likelihood (logit link), y_i in {0, 1}.
INLALikelihoodFn inla_binomial_logit_likelihood();
/// @brief Poisson likelihood (log link), y_i a non-negative count.
INLALikelihoodFn inla_poisson_log_likelihood();

} // namespace datamunge::bayes
