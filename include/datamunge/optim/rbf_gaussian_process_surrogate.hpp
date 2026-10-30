#pragma once

/// @file rbf_gaussian_process_surrogate.hpp
/// @brief Exact Gaussian-process surrogate with a squared-exponential kernel for
///        BayesianOptimization: posterior predictions, marginal-likelihood hyperparameter
///        fitting, and expected-improvement, probability-of-improvement or lower-confidence-bound
///        acquisition.

#include <datamunge/optim/bayesian_optimization.hpp>

#include <vector>

namespace datamunge::optim {

/// @brief Expected improvement of a Gaussian prediction below an incumbent (minimisation).
///
/// For \f$Y\sim N(\mu,\sigma^2)\f$ and target \f$t=\text{incumbent}-\xi\f$,
/// \f$\operatorname{E}\max(t-Y,0)=(t-\mu)\Phi(u)+\sigma\varphi(u)\f$ with \f$u=(t-\mu)/\sigma\f$.
/// @param mean         Predicted mean \f$\mu\f$.
/// @param sd           Predicted standard deviation \f$\sigma\ge0\f$; 0 gives \f$\max(t-\mu,0)\f$.
/// @param incumbent    Smallest value observed so far.
/// @param exploration  Margin \f$\xi\ge0\f$ an improvement must exceed.
/// @return The expected improvement (never negative).
///
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::optim::expected_improvement(0.0, 1.0, 0.0));  // 0.3989 = phi(0)
/// @endcode
double expected_improvement(double mean, double sd, double incumbent, double exploration = 0.0);

/// @brief Probability that a Gaussian prediction falls below incumbent − exploration.
/// @param mean         Predicted mean \f$\mu\f$.
/// @param sd           Predicted standard deviation; 0 gives 0 or 1.
/// @param incumbent    Smallest value observed so far.
/// @param exploration  Margin \f$\xi\ge0\f$.
/// @return \f$\Phi\big((\text{incumbent}-\xi-\mu)/\sigma\big)\f$.
///
/// @par Example
/// @code
/// std::print("{:.4f}\n", dm::optim::probability_of_improvement(1.0, 1.0, 0.0));  // 0.1587
/// @endcode
double probability_of_improvement(double mean, double sd, double incumbent, double exploration = 0.0);

/// @brief The Gaussian-process posterior of the objective at one point.
struct SurrogatePrediction {
    double mean{0.0};  ///< Posterior mean, in the units of the objective.
    double sd{0.0};    ///< Posterior standard deviation of the noise-free objective.
};

#ifndef SWIG
/// @brief Which acquisition function RBFGaussianProcessSurrogate::acquisition() returns.
enum class Acquisition {
    ExpectedImprovement,       ///< expected_improvement() with margin `exploration`.
    ProbabilityOfImprovement,  ///< probability_of_improvement() with margin `exploration`.
    LowerConfidenceBound       ///< \f$-(\mu-\kappa\sigma)\f$: the most optimistic plausible value, negated.
};

/// @brief Settings of an RBFGaussianProcessSurrogate built from options.
///
/// With `normalize_values`, the values are centred at their mean (the prior mean) and divided
/// by their standard deviation before fitting, so the kernel hyperparameters are on a fixed
/// scale whatever the objective's units. With `fit_hyperparameters`, the length scale (and the
/// signal variance, in closed form) maximise the marginal likelihood at every fit().
struct GaussianProcessSurrogateOptions {
    /// @brief Kernel length scale \f$\ell\f$ (the starting point is ignored when fitting).
    double length_scale{1.0};
    /// @brief Prior variance \f$s\f$ of the (normalised) objective.
    double signal_variance{1.0};
    /// @brief Observation-noise variance added to the kernel diagonal. When fitting, the ratio
    ///        noise / signal_variance (the nugget) is what stays fixed.
    double noise{1e-6};
    /// @brief Centre and scale the values before fitting.
    bool normalize_values{true};
    /// @brief Maximise the marginal likelihood over \f$\ell\f$ (and \f$s\f$) at every fit().
    bool fit_hyperparameters{true};
    /// @brief Also maximise over the nugget ratio, from \f$10^{-8}\f$ to 1 (for noisy objectives).
    bool fit_noise{false};
    /// @brief Smallest length scale searched; 0 means 0.01 times the widest coordinate range of the data.
    double min_length_scale{0.0};
    /// @brief Largest length scale searched; 0 means 10 times the widest coordinate range of the data.
    double max_length_scale{0.0};
    /// @brief Acquisition function.
    Acquisition acquisition{Acquisition::ExpectedImprovement};
    /// @brief Margin \f$\xi\f$ for expected or probability of improvement, in objective units.
    double exploration{0.0};
    /// @brief Width \f$\kappa\f$ of the lower confidence bound.
    double kappa{2.0};
};
#endif

/// @brief Exact Gaussian-process regression surrogate with the squared-exponential kernel
///        \f$s\exp(-\lVert x-x'\rVert^2/2\ell^2)\f$.
///
/// fit() Cholesky-factors \f$K+\sigma_n^2I=LL^\top\f$ once and solves for
/// \f$\alpha=(K+\sigma_n^2I)^{-1}(y-m)\f$; predict() then needs one forward substitution per point:
/// mean \f$m+k_*^\top\alpha\f$ and variance \f$s-\lVert L^{-1}k_*\rVert^2\f$.
///
/// The two-argument constructor keeps the original fixed setup: zero prior mean, unit signal
/// variance, no normalisation or fitting, expected improvement. The options constructor adds
/// value normalisation, marginal-likelihood fitting of the hyperparameters and other acquisitions.
/// The kernel is isotropic, so the inputs should be on comparable scales.
///
/// @par Example
/// @code
/// dm::optim::RBFGaussianProcessSurrogate gp(1.0, 0.0);
/// gp.fit({{0.0}, {2.0}}, {1.0, -1.0});
/// const auto p = gp.predict({1.0});
/// std::print("{:.4f} {:.4f}\n", p.mean, p.sd);  // 0.0000 0.5933
/// @endcode
/// @see BayesianOptimization, expected_improvement()
/// @par References
/// - C. E. Rasmussen and C. K. I. Williams (2006). *Gaussian Processes for Machine Learning*,
///   chapters 2 and 5. MIT Press.
class RBFGaussianProcessSurrogate : public BayesianSurrogate {
public:
    /// @brief Fixed-hyperparameter surrogate with expected improvement.
    /// @param length_scale  Kernel length scale \f$\ell>0\f$.
    /// @param noise         Noise variance added to the diagonal (\f$\ge0\f$).
    /// @throws std::invalid_argument  If @p length_scale is not positive or @p noise is negative.
    explicit RBFGaussianProcessSurrogate(double length_scale = 1.0, double noise = 1e-6);
#ifndef SWIG
    /// @brief Surrogate configured by options.
    /// @param options  Hyperparameters, normalisation, fitting and acquisition.
    /// @throws std::invalid_argument  If a hyperparameter or length-scale bound is out of range.
    explicit RBFGaussianProcessSurrogate(const GaussianProcessSurrogateOptions& options);
    /// @brief The options this surrogate was built with.
    /// @return The stored options.
    [[nodiscard]] const GaussianProcessSurrogateOptions& options() const noexcept { return options_; }
#endif

    /// @brief Fits the GP to the observations (refitting hyperparameters when configured).
    /// @param points  Training points, all of one dimension.
    /// @param values  Objective values.
    /// @throws std::invalid_argument  If the data are empty, mismatched or of mixed dimension.
    void fit(const std::vector<std::vector<double>>& points, const std::vector<double>& values) override;
    /// @brief The configured acquisition at @p point.
    /// @param point      Candidate point.
    /// @param incumbent  Smallest observed value.
    /// @return The acquisition value (0 before fitting).
    /// @throws std::invalid_argument  If @p point has the wrong dimension.
    double acquisition(const std::vector<double>& point, double incumbent) override;
    /// @brief Posterior mean and standard deviation at @p point.
    /// @param point  Query point.
    /// @return The prediction, in the objective's units.
    /// @throws std::logic_error      Before fit().
    /// @throws std::invalid_argument  If @p point has the wrong dimension.
    [[nodiscard]] SurrogatePrediction predict(const std::vector<double>& point) const;
    /// @brief Log marginal likelihood of the values at the hyperparameters in use.
    ///
    /// The density of the raw values under the GP with the prior mean (and scale, when
    /// normalising) treated as known, so it includes the \f$-n\log(\text{scale})\f$ Jacobian.
    /// @return \f$\log p(y)\f$; 0 before fit().
    [[nodiscard]] double log_marginal_likelihood() const noexcept { return log_marginal_likelihood_; }

    /// @brief The length scale in use (fitted, after fit(), when fitting is on).
    /// @return \f$\ell\f$.
    [[nodiscard]] double length_scale() const noexcept { return length_scale_; }
    /// @brief The noise variance in use, in normalised units when normalising.
    /// @return \f$\sigma_n^2\f$.
    [[nodiscard]] double noise() const noexcept { return noise_; }
    /// @brief The signal variance in use, in normalised units when normalising.
    /// @return \f$s\f$.
    [[nodiscard]] double signal_variance() const noexcept { return signal_variance_; }
    /// @brief The prior mean: the mean of the values when normalising, else 0.
    /// @return \f$m\f$.
    [[nodiscard]] double prior_mean() const noexcept { return offset_; }
    /// @brief The value scale: the standard deviation of the values when normalising, else 1.
    /// @return The scale.
    [[nodiscard]] double value_scale() const noexcept { return scale_; }
    /// @brief The training points from the last fit() (empty before fitting).
    /// @return The stored training points.
    [[nodiscard]] const std::vector<std::vector<double>>& points() const noexcept { return points_; }
    /// @brief The training values from the last fit() (empty before fitting).
    /// @return The stored training values.
    [[nodiscard]] const std::vector<double>& values() const noexcept { return values_; }

private:
#ifndef SWIG
    GaussianProcessSurrogateOptions options_;  ///< Configuration.
#endif
    double length_scale_;                      ///< \f$\ell\f$ in use.
    double noise_;                             ///< \f$\sigma_n^2\f$ in use.
    double signal_variance_{1.0};              ///< \f$s\f$ in use.
    double offset_{0.0};                       ///< Prior mean.
    double scale_{1.0};                        ///< Value scale.
    double log_marginal_likelihood_{0.0};      ///< At the hyperparameters in use.
    std::vector<std::vector<double>> points_;  ///< Training points.
    std::vector<double>              values_;  ///< Training values (raw).
    std::vector<double>              alpha_;   ///< \f$(K+\sigma_n^2I)^{-1}(y-m)/\text{scale}\f$.
    std::vector<double>              cholesky_;  ///< Lower Cholesky factor, row-major \f$n\times n\f$.

    /// @brief Normalised latent mean and variance at @p x.
    /// @param x  Query point.
    /// @param mean  Output: \f$k_*^\top\alpha\f$.
    /// @param var   Output: \f$s-\lVert L^{-1}k_*\rVert^2\f$ (may be slightly negative).
    void latent(const std::vector<double>& x, double& mean, double& var) const;
};

} // namespace datamunge::optim
