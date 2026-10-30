#pragma once
/// @file inla.hpp
/// @brief Dense Gaussian-strategy INLA with auditable conditional fits and finite-grid mixtures.
#include <cstddef>
#include <cstdint>
#include <datamunge/bayes/inla_integration_strategy.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <functional>
#include <limits>
#include <vector>
namespace datamunge::bayes {
/// @brief Log likelihood and its first derivative and negative second derivative in eta.
/// Curvature must be finite/nonnegative for the concave conditional model. Owns values.
/// @par Example
/// @code
/// const auto p=dm::bayes::inla_binomial_logit_likelihood()(1,0,{});
/// std::print("{} {}\n",p.score,p.curvature); // 0.5 0.25
/// @endcode
/// @see INLALikelihoodFn
struct INLALikelihoodPoint {
  double log_density{0}; ///< Normalized observation log density, including constants.
  double score{0};       ///< Derivative of log density with respect to eta.
  double curvature{0};   ///< Negative second derivative in eta, nonnegative.
};
/// @brief Callback (y,eta,theta) -> likelihood value/derivatives; parameters are borrowed.
/// @par Example
/// @code
/// const dm::bayes::INLALikelihoodFn f=dm::bayes::inla_poisson_log_likelihood();
/// std::print("{}\n",f(2,0,{}).score); // 1
/// @endcode
/// @see INLALikelihoodPoint
using INLALikelihoodFn =
    std::function<INLALikelihoodPoint(double, double, const std::vector<double>&)>;
/// @brief Callback theta -> owned symmetric positive-definite latent precision Q(theta).
/// @par Example
/// @code
/// dm::bayes::INLAPrecisionFn q=[](const auto&){return dm::linalg::DenseMatrix<double>(1,1,2);};
/// std::print("{}\n",q({})(0,0)); // 2
/// @endcode
/// @see INLA::fit()
using INLAPrecisionFn = std::function<linalg::DenseMatrix<double>(const std::vector<double>&)>;
/// @brief Log hyperprior on the caller's theta scale, including any change-of-variable Jacobian.
/// Return negative infinity for excluded points. Normalize the prior for evidence calculations.
/// @par Example
/// @code
/// dm::bayes::INLALogPriorFn p=[](const auto& t){return dm::bayes::normal_lpdf(t[0],0,1);};
/// std::print("{:.3f}\n",p({0})); // -0.919
/// @endcode
/// @see INLA::fit()
using INLALogPriorFn = std::function<double(const std::vector<double>&)>;
/// @brief Numerical controls for dense Gaussian-strategy INLA; no sparse or skew correction.
/// Grid work grows as points^theta_dim. DE is a seeded mode search, not a global certificate.
/// @par Example
/// @code
/// dm::bayes::INLAOptions o;o.grid_points_per_dim=31;o.grid_span=5;
/// const dm::bayes::INLA solver(o);
/// @endcode
/// @see INLA
struct INLAOptions {
  INLAIntegrationStrategy strategy{INLAIntegrationStrategy::Grid}; ///< Grid or mode-only EB.
  std::size_t newton_max_iterations{100}; ///< Positive conditional Newton update budget.
  double newton_tolerance{1e-8}; ///< Positive proposed-step tolerance times (1+mode infinity norm).
  std::size_t   grid_points_per_dim{7}; ///< Odd >=3 for Grid; total at most one million points.
  double        grid_span{4}; ///< Positive half-span of the whitened grid in curvature-SD units.
  std::size_t   mode_population_size{40};  ///< DE population, at least four.
  std::size_t   mode_max_generations{200}; ///< Positive maximum DE generations.
  std::uint64_t seed{42};                  ///< Reproducible DE seed.
};
/// @brief One converged Gaussian approximation at fixed hyperparameters.
/// Mode equals conditional mean only when the conditional posterior is Gaussian.
/// Covariance is inverse negative Hessian at the mode; all arrays own their storage.
/// @par Example
/// @code
/// const auto f=dm::bayes::INLA().conditional(dm::linalg::DenseMatrix<double>(3,1,1),
///     {1,2,3},dm::bayes::inla_gaussian_likelihood([](const auto&){return 1.;}),
///     dm::linalg::DenseMatrix<double>(1,1,1));
/// std::print("{} {}\n",f.mode[0],f.covariance(0,0)); // 1.5 0.25
/// @endcode
/// @see INLA::conditional()
struct INLAConditional {
  std::vector<double>         mode;       ///< Conditional latent mode, length p.
  linalg::DenseMatrix<double> covariance; ///< Owned p-by-p inverse negative Hessian.
  double log_marginal_likelihood{0}; ///< Conditional Laplace log integral over the latent field.
  std::size_t iterations{0};   ///< Accepted Newton updates, including final convergence update.
  double      gradient_inf{0}; ///< Absolute conditional log-posterior score infinity norm at mode.
};
/// @brief Gaussian-mixture posterior summaries and the full retained hyperparameter grid.
/// These are approximate moments, not exact posterior draws. Within variance is the
/// weighted conditional variance; between variance is variance of conditional modes.
/// Their sum is total variance; neither term alone orders total variance against EB.
/// All members own their data. Conditional covariance storage costs O(K*p^2).
/// @par Example
/// @code
/// dm::bayes::INLAResult r;
/// std::print("{}\n",r.theta_grid_weights.size()); // 0
/// @endcode
/// @see INLA::fit(), inla_latent_summary(), inla_grid_table()
struct INLAResult {
  std::vector<double> latent_mean; ///< Weighted conditional modes, length p.
  std::vector<double> latent_sd;   ///< Square roots of total mixture marginal variances.
  std::vector<double> theta_mean;  ///< Weighted theta-coordinate means, on fitted scale.
  std::vector<double> theta_sd;    ///< Theta-coordinate SDs, zero in EB.
  double              log_marginal_likelihood{
      std::numeric_limits<double>::quiet_NaN()}; ///< Grid Laplace log integral; NaN for EB (no
                                                              ///< theta integration).
  std::size_t         theta_dim{0};       ///< Number of hyperparameters, from theta_init.
  std::vector<double> theta_grid;         ///< K-by-theta_dim coordinates flattened by row.
  std::vector<double> theta_grid_weights; ///< K normalized nonnegative mixture weights.
  std::vector<double> theta_mode;         ///< DE-selected hyperposterior mode within search box.
  double              log_posterior_at_mode{
      0}; ///< Conditional Laplace log evidence plus log hyperprior at mode.
  std::vector<double> theta_log_posterior; ///< Same log integrand at each retained grid point.
  std::vector<std::vector<double>>
      conditional_means; ///< K conditional latent modes, each length p.
  std::vector<linalg::DenseMatrix<double>>
                           conditional_covariances; ///< K inverse negative Hessians, p-by-p.
  std::vector<std::size_t> conditional_iterations; ///< Accepted Newton update count per grid point.
  std::vector<double>      conditional_gradient_inf; ///< Final score norm per grid point.
  linalg::DenseMatrix<double> latent_covariance;     ///< Full p-by-p covariance of the mixture.
  std::vector<double>
      latent_within_variance; ///< E_theta[Var(x_j|theta,y)] under Gaussian strategy.
  std::vector<double> latent_between_variance; ///< Var_theta[mode_j(theta)], nonnegative.
  double              grid_cell_volume{1};     ///< Constant rotated-grid cell volume; EB uses one.
  std::vector<double> theta_grid_cell_volumes; ///< Actual cell measures; scalar boundary cells are
                                               ///< clipped/recentered.
  bool   grid_covariance_fallback{false};      ///< Curvature failed; used diagonal (box width/4)^2.
  double grid_edge_mass{
      0}; ///< Retained weight at outer whitened-grid index layer; not a tail bound.
  std::size_t grid_requested_points{0}; ///< Tensor-grid candidates (one for EB).
  std::size_t grid_skipped_points{0}; ///< Candidates outside box or with unusable conditional fits.
};
/// @brief Gaussian-strategy integrated Laplace inference for a latent Gaussian field.
/// Model: eta=D*x, x|theta~N(0,Q(theta)^-1), with conditionally independent observations.
/// At each theta, Newton finds x's mode and inverse Hessian. The Laplace log integrand
/// is log prior(theta)+.5 log|Q|+log likelihood-.5*x'Q*x-.5 log|H|. Grid integration
/// forms a mixture of these Gaussians; there is no simplified/full marginal Laplace
/// skew correction and no sparse GMRF solver. Non-Gaussian conditional modes are
/// used as approximate means. Use only modest dense latent dimensions.
/// Borrowed callbacks run synchronously; result owns everything. Const calls have no
/// shared mutation, but user callbacks must themselves be thread-safe when shared.
/// @par Example
/// @code
/// dm::bayes::INLAOptions o;o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
/// const dm::bayes::INLA solver(o);
/// const auto r=solver.fit(dm::linalg::DenseMatrix<double>(3,1,1),{1,2,3},
///     dm::bayes::inla_gaussian_likelihood([](const auto&){return 1.;}),
///     [](const auto& t){return dm::linalg::DenseMatrix<double>(1,1,std::exp(t[0]));},
///     [](const auto& t){return dm::bayes::normal_lpdf(t[0],0,1);},{-5},{5},{0});
/// std::print("{}\n",r.theta_grid_weights.size()); // 1
/// @endcode
/// @par References
/// - Rue, Martino and Chopin (2009), JRSS B 71, 319-392.
///   https://doi.org/10.1111/j.1467-9868.2008.00700.x
/// - Tierney and Kadane (1986), JASA 81, 82-86. https://doi.org/10.1080/01621459.1986.10478240
/// @see INLAResult, INLAConditional
class INLA {
 public:
  /// @brief Configure positive Newton/DE/grid controls; no model is fitted yet.
  /// @param options Defaults from INLAOptions.
  /// @throws std::invalid_argument Invalid enum, budget, tolerance, population or grid controls.
  /// @par Example
  /// @code
  /// const dm::bayes::INLA solver;
  /// std::print("{}\n",solver.options().newton_max_iterations); // 100
  /// @endcode
  /// @see INLAOptions
  explicit INLA(INLAOptions options = {});
  /// @brief Fit bounded hyperparameters and combine conditional Gaussian moments.
  /// Search and grid evaluations stay within the supplied box. Finite differences
  /// use inward probes near its boundary. A non-SPD hyper-Hessian triggers the
  /// recorded box-width fallback. For one theta dimension, boundary cells are clipped and
  /// recentered. Higher dimensions use constant rotated cell volumes and skip outside-box centers;
  /// failed conditional fits are also skipped and counted. There is no adaptive grid
  /// refinement or error guarantee. Check density/span/resolution sensitivity.
  /// Evidence requires normalized likelihood/prior and intentional integration
  /// support; a constant improper hyperprior has an arbitrary evidence scale.
  /// EB returns one conditional fit and NaN evidence, not an integrated likelihood.
  /// Work per conditional Newton iteration O(n*p^2+p^3); retained storage O(K*p^2).
  /// @param design Finite nonempty n-by-p observation matrix.
  /// @param y Exactly n finite responses valid for the supplied likelihood.
  /// @param likelihood Concave observation log density/derivatives; callback exceptions propagate.
  /// @param precision Returns p-by-p finite symmetric SPD Q(theta); failed builders are skipped.
  /// @param log_prior Log hyperprior on theta scale; negative infinity excludes a point.
  /// @param theta_lower Finite lower bounds, strictly below upper bounds.
  /// @param theta_upper Finite upper bounds; part of the integration support.
  /// @param theta_init Finite in-box seed; nonempty and determines theta dimension.
  /// @return Owned approximate mixture and all retained conditional fits.
  /// @throws std::invalid_argument Bad shapes/finite values/box, empty callbacks or excessive grid.
  /// @throws std::runtime_error No usable converged approximation at the selected mode/grid.
  /// @par Example
  /// @code
  /// dm::bayes::INLAOptions o;o.strategy=dm::bayes::INLAIntegrationStrategy::EmpiricalBayes;
  /// const auto r=dm::bayes::INLA(o).fit(dm::linalg::DenseMatrix<double>(2,1,1),{0,1},
  ///     dm::bayes::inla_binomial_logit_likelihood(),[](const auto&){return
  ///     dm::linalg::DenseMatrix<double>(1,1,1);},
  ///     [](const auto& t){return dm::bayes::normal_lpdf(t[0],0,1);},{-4},{4},{0});
  /// std::print("{}\n",r.theta_grid_weights.size()); // 1
  /// @endcode
  /// @see conditional(), INLAResult
  [[nodiscard]] INLAResult fit(
      const linalg::DenseMatrix<double>& design,
      const std::vector<double>&         y,
      const INLALikelihoodFn&            likelihood,
      const INLAPrecisionFn&             precision,
      const INLALogPriorFn&              log_prior,
      const std::vector<double>&         theta_lower,
      const std::vector<double>&         theta_upper,
      const std::vector<double>&         theta_init) const;
  /// @brief Approximate only the latent posterior at fixed precision and hyperparameters.
  /// Uses the same safeguarded Newton loop as fit(). Exhaustion is an error, never
  /// silently reported as convergence. Gaussian likelihood gives exact normal moments
  /// up to floating-point error; other families generally have skew approximation error.
  /// @param design Finite nonempty n-by-p design.
  /// @param y n finite responses valid for likelihood.
  /// @param likelihood Concave log density and its eta derivatives.
  /// @param precision Finite symmetric SPD p-by-p prior precision, borrowed.
  /// @param theta Finite callback hyperparameters; default empty for fully fixed models.
  /// @return Owned mode, full covariance, conditional Laplace evidence and convergence evidence.
  /// @throws std::invalid_argument Invalid design/response/callback or nonfinite theta.
  /// @throws std::runtime_error Invalid precision, failed line search or exhausted Newton budget.
  /// @par Example
  /// @code
  /// const auto r=dm::bayes::INLA().conditional(dm::linalg::DenseMatrix<double>(2,1,1),{0,1},
  ///     dm::bayes::inla_binomial_logit_likelihood(),dm::linalg::DenseMatrix<double>(1,1,1));
  /// std::print("{:.1f}\n",r.mode[0]); // 0.0
  /// @endcode
  /// @see INLAConditional, fit()
  [[nodiscard]] INLAConditional conditional(
      const linalg::DenseMatrix<double>& design,
      const std::vector<double>&         y,
      const INLALikelihoodFn&            likelihood,
      const linalg::DenseMatrix<double>& precision,
      const std::vector<double>&         theta = {}) const;
  /// @brief Borrow immutable controls; reference lives with this solver.
  /// @return Stored INLAOptions.
  [[nodiscard]] const INLAOptions& options() const noexcept { return options_; }

 private:
  INLAOptions options_; ///< Owned numerical controls.
};
/// @brief Normal observation likelihood with identity link and callback-selected noise SD.
/// Includes the normal density constant. Returned function borrows no external model
/// state beyond what its owned sigma callback captures; inputs are unchanged.
/// @param sigma_of_theta Returns positive finite sigma with representable finite nonzero variance.
/// @return Callable evaluating log density, (y-eta)/sigma^2 and 1/sigma^2.
/// @throws std::invalid_argument Empty callback, or invalid y/eta/sigma at evaluation.
/// @par Example
/// @code
/// const auto f=dm::bayes::inla_gaussian_likelihood([](const auto&){return 2.;});
/// std::print("{}\n",f(3,1,{}).score); // 0.5
/// @endcode
/// @see INLALikelihoodPoint
INLALikelihoodFn inla_gaussian_likelihood(
    std::function<double(const std::vector<double>&)> sigma_of_theta);
/// @brief Stable Bernoulli logit log density, score y-p and curvature p*(1-p).
/// This accepts individual zero/one outcomes, not fractional proportions or trial totals.
/// @return Owned callback; ignores theta, uses stable exponential tails.
/// @throws std::invalid_argument Nonbinary y or nonfinite eta at evaluation.
/// @par Example
/// @code
/// std::print("{}\n",dm::bayes::inla_binomial_logit_likelihood()(1,0,{}).score); // 0.5
/// @endcode
/// @see inla_gaussian_likelihood()
INLALikelihoodFn inla_binomial_logit_likelihood();
/// @brief Poisson log-link log density including log-factorial, score y-exp(eta).
/// @return Owned callback, ignoring theta; curvature equals exp(eta).
/// Extremely positive eta can yield an unusable infinite evaluation, rejected by INLA.
/// @throws std::invalid_argument Noninteger/negative/nonfinite count or nonfinite eta at
/// evaluation.
/// @par Example
/// @code
/// std::print("{}\n",dm::bayes::inla_poisson_log_likelihood()(2,0,{}).curvature); // 1
/// @endcode
/// @see inla_binomial_logit_likelihood()
INLALikelihoodFn inla_poisson_log_likelihood();
} // namespace datamunge::bayes
