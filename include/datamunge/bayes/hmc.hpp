#pragma once

/// @file hmc.hpp
/// @brief Hamiltonian Monte Carlo with a fixed number of leapfrog steps, and its exact acceptance
///        rate on a Gaussian target.

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::bayes {

/// Settings for HMC.
struct HMCOptions {
    /// @brief Number of post-warmup draws.
    std::size_t num_samples{1000};
    /// @brief Number of warmup iterations (step-size adaptation; 0 keeps initial_step_size).
    std::size_t num_warmup{1000};
    /// @brief Number of leapfrog steps per proposal (trajectory length = num_leapfrog_steps * step size).
    std::size_t num_leapfrog_steps{10};
    /// @brief Leapfrog step size at the start of warmup.
    double initial_step_size{0.1};
    /// @brief Target Metropolis acceptance rate the dual-averaging adaptation aims for during warmup.
    double target_accept_rate{0.8};
    /// @brief Random seed.
    std::uint64_t seed{42};
};

/// Output of HMC::sample().
struct HMCResult {
    /// @brief Post-warmup draws, one vector<double> per sample.
    std::vector<std::vector<double>> samples;
    /// @brief Fraction of all iterations (warmup included) whose proposal was accepted.
    double accept_rate{0.0};
    /// @brief Step size used after warmup.
    double final_step_size{0.0};
};

/// @brief Hamiltonian Monte Carlo with a fixed leapfrog trajectory length and Nesterov
///        dual-averaging step-size adaptation during warmup (Hoffman & Gelman 2014), using
///        an identity mass matrix. Samples from the distribution proportional to
///        exp(log_posterior) over unconstrained real parameters.
class HMC {
public:
    /// @brief Stores the sampler settings.
    /// @param options Sampler settings.
    explicit HMC(HMCOptions options = {});

    /// @brief Runs warmup (adapting the step size unless num_warmup is 0) then num_samples draws.
    /// @param log_posterior Log density (up to a constant) and its gradient.
    /// @param initial_params Starting point on the unconstrained scale.
    /// @return Draws, acceptance rate and final step size.
    HMCResult sample(optim::DifferentiableFunction& log_posterior, const std::vector<double>& initial_params) const;

    /// @brief The options this HMC was configured with.
    /// @return The stored options.
    [[nodiscard]] const HMCOptions& options() const noexcept { return options_; }

private:
    HMCOptions options_; ///< Sampler settings.
};

/// @brief Exact expected Metropolis acceptance of one HMC proposal (fixed step size and
///        number of leapfrog steps, unit mass) on a one-dimensional \f$N(m,s^2)\f$ target, started
///        in equilibrium.
///
/// In scaled coordinates \f$x=((q-m)/s,\,p)\f$ each leapfrog step is the linear map
/// \f$M=\begin{pmatrix}1&0\\-h/2&1\end{pmatrix}\begin{pmatrix}1&h\\0&1\end{pmatrix}\begin{pmatrix}1&0\\-h/2&1\end{pmatrix}\f$
/// with \f$h=\varepsilon/s\f$, \f$\det M=1\f$ and \f$\operatorname{tr}M=2-h^2\f$, so it is stable
/// exactly when \f$\varepsilon<2s\f$. Writing \f$x_0=r(\cos\phi,\sin\phi)\f$ the energy error is
/// \f$\Delta H=r^2f(\phi)\f$ with \f$f=\tfrac12(|M^L(\cos\phi,\sin\phi)|^2-1)\f$, and because
/// \f$r^2/2\sim\text{Exp}(1)\f$ the radial average of \f$\min(1,e^{-\Delta H})\f$ is
/// \f$1/(1+2f)\f$ for \f$f>0\f$ and 1 otherwise. The angle is integrated numerically.
/// @param step_size Leapfrog step size \f$\varepsilon>0\f$.
/// @param num_steps Number of leapfrog steps \f$L\geq1\f$.
/// @param sd Target standard deviation \f$s>0\f$.
/// @param angles Number of quadrature angles (at least 8).
/// @return The expected acceptance probability.
/// @throws std::invalid_argument for non-positive arguments.
double leapfrog_gaussian_acceptance(double step_size, std::size_t num_steps, double sd, std::size_t angles = 4096);

} // namespace datamunge::bayes
