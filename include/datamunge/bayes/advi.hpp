#pragma once

/// @file advi.hpp
/// @brief Automatic differentiation variational inference with mean-field and full-rank Gaussians.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

/// @brief Support of one parameter, mapped to the real line before the Gaussian is fitted.
enum class SupportKind {
    real,     ///< No constraint; identity map.
    lower,    ///< theta > lower; theta = lower + exp(zeta).
    upper,    ///< theta < upper; theta = upper - exp(zeta).
    interval, ///< lower < theta < upper; theta = lower + (upper - lower) logistic(zeta).
};

/// @brief Support of one parameter.
/// @par Example
/// @code
/// const auto s = dm::bayes::ParameterSupport::positive();
/// std::println("{}", s.lower); // 0
/// @endcode
struct ParameterSupport {
    SupportKind kind{SupportKind::real}; ///< Constraint type.
    double lower{0.0};                   ///< Lower bound (lower and interval kinds).
    double upper{1.0};                   ///< Upper bound (upper and interval kinds).

    /// @brief Positive parameter (lower bound 0).
    /// @return The support.
    static ParameterSupport positive() { return {SupportKind::lower, 0.0, 0.0}; }

    /// @brief Parameter in (0, 1).
    /// @return The support.
    static ParameterSupport unit_interval() { return {SupportKind::interval, 0.0, 1.0}; }
};

/// @brief Variational family on the unconstrained space.
enum class AdviFamily {
    mean_field, ///< Independent normals: mean mu and log-SD omega.
    full_rank,  ///< Multivariate normal with mean mu and Cholesky factor L.
};

/// @brief Options for advi().
/// @par Example
/// @code
/// const dm::bayes::AdviOptions o{.family = dm::bayes::AdviFamily::full_rank};
/// std::println("{}", o.gradient_samples); // 1
/// @endcode
struct AdviOptions {
    AdviFamily family{AdviFamily::mean_field};  ///< Variational family.
    std::vector<ParameterSupport> supports;     ///< One per parameter; empty means all real.
    std::size_t gradient_samples{1};            ///< Monte Carlo draws per gradient estimate.
    std::size_t elbo_samples{100};              ///< Draws per ELBO evaluation.
    std::size_t max_iterations{10000};          ///< Iteration limit of the stochastic optimisation.
    std::size_t eval_every{100};                ///< Iterations between ELBO evaluations.
    double tolerance{0.01};                     ///< Relative ELBO change that signals convergence.
    double step_size{0.0};                      ///< eta; 0 selects it from {100, 10, 1, 0.1, 0.01}.
    std::size_t adapt_iterations{50};           ///< Iterations per candidate in the step-size search.
    std::size_t output_draws{1000};             ///< Draws returned on the constrained scale.
    std::uint64_t seed{1};                      ///< Seed of all randomness.
};

/// @brief Output of advi().
/// @par Example
/// @code
/// dm::bayes::AdviResult r;
/// r.mean = {0.0, 1.0};
/// std::println("{}", r.mean.size()); // 2
/// @endcode
struct AdviResult {
    std::vector<double> mean;                  ///< Variational mean mu (unconstrained scale).
    std::vector<double> cholesky;              ///< Row-major lower Cholesky factor of the covariance (d x d).
    double elbo{};                             ///< Final ELBO estimate.
    std::vector<double> elbo_history;          ///< ELBO at each evaluation.
    double step_size{};                        ///< eta used.
    std::size_t iterations{};                  ///< Iterations performed.
    bool converged{};                          ///< Relative ELBO change fell below the tolerance.
    std::vector<std::vector<double>> draws;    ///< Approximate posterior draws on the constrained scale.
};

/// @brief Automatic differentiation variational inference (Kucukelbir et al. 2017).
///
/// Each parameter is mapped to the real line by its support transform \f$\theta=T(\zeta)\f$, and a
/// Gaussian \f$q(\zeta)\f$ maximises the evidence lower bound
/// \f$\mathrm{ELBO}=E_q[\log p(T(\zeta))+\log|J_T(\zeta)|]+\mathcal H[q]\f$. Its gradient is
/// estimated by reparameterisation, \f$\zeta=\mu+L\varepsilon\f$ (L diagonal, \f$e^\omega\f$, for
/// mean-field), with the entropy gradient exact. The step sizes follow Stan:
/// \f$\rho_k=\eta k^{-1/2+10^{-16}}/(1+\sqrt{s_k})\f$ with \f$s_k=0.1g_k^2+0.9s_{k-1}\f$ per coordinate.
/// When `step_size` is 0, eta is chosen by short runs. The optimisation stops when the mean or
/// median relative ELBO change over recent evaluations falls below the tolerance. Mean-field
/// approximations ignore posterior correlation and typically understate marginal variances.
/// @param log_density Log posterior density up to a constant, on the constrained scale, with its gradient.
/// @param initial Starting value of the variational mean on the constrained scale (inside the supports).
/// @param options Family, supports, iteration and output settings.
/// @return The fitted Gaussian, ELBO trace and constrained draws.
/// @throws std::invalid_argument For empty or out-of-support initial values, supports of the wrong
/// length or with invalid bounds, or zero sample counts.
/// @throws std::runtime_error When every candidate step size diverges or the ELBO becomes nonfinite.
/// @par Example
/// A standard normal target is recovered by either family:
/// @code
/// struct StdNormal : dm::optim::DifferentiableFunction {
///     double evaluate(const std::vector<double>& t) override { return -0.5 * t[0] * t[0]; }
///     std::vector<double> gradient(const std::vector<double>& t) override { return {-t[0]}; }
/// } target;
/// const auto r = dm::bayes::advi(target, {1.0}, {.seed = 3});
/// std::println("{}", std::abs(r.mean[0]) < 0.1 && std::abs(r.cholesky[0] - 1) < 0.1); // true
/// @endcode
AdviResult advi(optim::DifferentiableFunction& log_density, const std::vector<double>& initial,
                const AdviOptions& options = {});

} // namespace datamunge::bayes
