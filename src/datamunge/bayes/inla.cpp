/// @file inla.cpp
/// @brief Dense conditional Gaussian approximations and bounded hyperparameter quadrature.
#include <algorithm>
#include <cmath>
#include <datamunge/bayes/distributions.hpp>
#include <datamunge/bayes/inla.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/optim/differential_evolution.hpp>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::bayes {

/// @brief Internal conditional-mode and grid machinery.
namespace {

/// @brief Dense Cholesky factorization type.
using linalg::CholeskyDecomposition;
/// @brief Owned dense matrix type.
using linalg::DenseMatrix;

/// @brief Maximum integrated hyperparameter dimension (grid cost grows exponentially).
// Grid integration's cost is grid_points_per_dim^theta_dim; beyond this, EmpiricalBayes is
// the only practical strategy (matching R-INLA's own documented advice to switch away from
// "grid" once there are more than a handful of hyperparameters).
constexpr std::size_t kMaxGridThetaDim = 4;

/// @brief Log determinant of an SPD matrix from its Cholesky factor.
/// @param chol Successful factorization with positive diagonal.
/// @return Twice the sum of log diagonal entries.
double log_det_from_chol(const CholeskyDecomposition<double>& chol) {
  double s = 0.0;
  for (std::size_t i = 0; i < chol.L.rows(); ++i)
    s += 2.0 * std::log(chol.L(i, i));
  return s;
}

/// @brief Conditional mode, Hessian and explicit Newton convergence evidence.
struct InnerFit {
  bool                ok{true}; ///< Whether a finite converged approximation is available.
  std::vector<double> mode;     ///< Conditional latent mode.
  CholeskyDecomposition<double> hessian_chol; ///< Conditional negative Hessian factor.
  double      log_joint_at_mode{0.0};         ///< Log likelihood minus latent prior quadratic.
  std::size_t iterations{0};                  ///< Accepted Newton updates.
  double      gradient_inf{0};                ///< Final absolute score infinity norm.
};

/// @brief Dense matrix-vector product without input mutation.
/// @param D Compatible row matrix.
/// @param x Column vector of D.cols() entries.
/// @return Owned D*x.
std::vector<double> mat_vec(const DenseMatrix<double>& D, const std::vector<double>& x) {
  std::vector<double> out(D.rows(), 0.0);
  for (std::size_t i = 0; i < D.rows(); ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < D.cols(); ++j)
      s += D(i, j) * x[j];
    out[i] = s;
  }
  return out;
}

/// @brief Safeguarded Newton mode of a concave latent log posterior.
/// @param D Finite n-by-p design.
/// @param y n finite responses.
/// @param likelihood Finite log density, score and nonnegative curvature callback.
/// @param Q Validated p-by-p SPD prior precision.
/// @param theta Hyperparameter values passed to likelihood.
/// @param options Positive update budget and relative step tolerance.
/// @return Mode/Hessian and score evidence; ok=false on exhaustion or invalid evaluations.
InnerFit newton_raphson_mode(
    const DenseMatrix<double>& D,
    const std::vector<double>& y,
    const INLALikelihoodFn&    likelihood,
    const DenseMatrix<double>& Q,
    const std::vector<double>& theta,
    const INLAOptions&         options) {
  const std::size_t n_obs    = D.rows();
  const std::size_t n_latent = D.cols();

  std::vector<double> x(n_latent, 0.0);
  std::vector<double> eta = mat_vec(D, x);

  // Evaluate the unnormalized conditional log posterior at a proposed latent state.
  auto log_joint = [&](const std::vector<double>& xv, const std::vector<double>& etav) {
    double quad = 0.0;
    for (std::size_t j = 0; j < n_latent; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n_latent; ++k)
        s += Q(j, k) * xv[k];
      quad += xv[j] * s;
    }
    double ll = 0.0;
    for (std::size_t i = 0; i < n_obs; ++i) {
      if (!std::isfinite(etav[i]))
        return -std::numeric_limits<double>::infinity();
      ll += likelihood(y[i], etav[i], theta).log_density;
    }
    return -0.5 * quad + ll;
  };

  // Add the likelihood curvature contribution to the prior precision.
  auto build_A = [&](const std::vector<double>& curvature) {
    DenseMatrix<double> A = Q;
    for (std::size_t i = 0; i < n_obs; ++i) {
      const double c = curvature[i];
      if (c == 0.0)
        continue;
      for (std::size_t j = 0; j < n_latent; ++j) {
        const double dij = D(i, j) * c;
        if (dij == 0.0)
          continue;
        for (std::size_t k = 0; k < n_latent; ++k)
          A(j, k) += dij * D(i, k);
      }
    }
    return A;
  };

  double   cur_lj = log_joint(x, eta);
  InnerFit fit;
  fit.ok = false;
  if (!std::isfinite(cur_lj))
    return fit;
  bool converged = false;

  for (std::size_t iter = 0; iter < options.newton_max_iterations; ++iter) {
    std::vector<double> score(n_obs), curvature(n_obs);
    for (std::size_t i = 0; i < n_obs; ++i) {
      const auto pt = likelihood(y[i], eta[i], theta);
      if (!std::isfinite(pt.log_density) || !std::isfinite(pt.score) ||
          !std::isfinite(pt.curvature) || pt.curvature < 0)
        return fit;
      score[i]     = pt.score;
      curvature[i] = pt.curvature;
    }

    const DenseMatrix<double> A = build_A(curvature);

    std::vector<double> Qx(n_latent, 0.0);
    for (std::size_t j = 0; j < n_latent; ++j) {
      double s = 0.0;
      for (std::size_t k = 0; k < n_latent; ++k)
        s += Q(j, k) * x[k];
      Qx[j] = s;
    }
    std::vector<double> rhs(n_latent, 0.0);
    for (std::size_t i = 0; i < n_obs; ++i) {
      const double si = score[i];
      if (si == 0.0)
        continue;
      for (std::size_t j = 0; j < n_latent; ++j)
        rhs[j] += D(i, j) * si;
    }
    for (std::size_t j = 0; j < n_latent; ++j)
      rhs[j] -= Qx[j];

    const auto chol_a = linalg::cholesky(A);
    if (!chol_a.ok) {
      fit.ok = false;
      return fit;
    }
    const std::vector<double> delta = chol_a.solve(rhs);

    double              step_scale = 1.0;
    std::vector<double> x_new      = x;
    std::vector<double> eta_new    = eta;
    double              new_lj     = cur_lj;
    bool                accepted   = false;
    for (int half = 0; half < 30; ++half) {
      for (std::size_t j = 0; j < n_latent; ++j)
        x_new[j] = x[j] + step_scale * delta[j];
      eta_new = mat_vec(D, x_new);
      new_lj  = log_joint(x_new, eta_new);
      if (std::isfinite(new_lj) && new_lj >= cur_lj - 1e-12 * (1 + std::abs(cur_lj))) {
        accepted = true;
        break;
      }
      step_scale *= 0.5;
    }

    if (!accepted)
      return fit;
    double proposed_norm = 0, x_norm = 0;
    for (std::size_t j = 0; j < n_latent; ++j) {
      proposed_norm = std::max(proposed_norm, std::abs(delta[j]));
      x_norm        = std::max(x_norm, std::abs(x_new[j]));
    }

    x              = x_new;
    eta            = eta_new;
    cur_lj         = new_lj;
    fit.iterations = iter + 1;
    if (proposed_norm < options.newton_tolerance * (1 + x_norm)) {
      converged = true;
      break;
    }
  }

  if (!converged)
    return fit;
  std::vector<double> final_curvature(n_obs);
  std::vector<double> final_gradient(n_latent, 0);
  for (std::size_t i = 0; i < n_obs; ++i) {
    const auto pt = likelihood(y[i], eta[i], theta);
    if (!std::isfinite(pt.log_density) || !std::isfinite(pt.score) ||
        !std::isfinite(pt.curvature) || pt.curvature < 0)
      return fit;
    final_curvature[i] = pt.curvature;
    for (std::size_t j = 0; j < n_latent; ++j)
      final_gradient[j] += D(i, j) * pt.score;
  }
  const auto Qx = mat_vec(Q, x);
  for (std::size_t j = 0; j < n_latent; ++j)
    fit.gradient_inf = std::max(fit.gradient_inf, std::abs(final_gradient[j] - Qx[j]));
  const auto chol_final = linalg::cholesky(build_A(final_curvature));
  if (!chol_final.ok) {
    fit.ok = false;
    return fit;
  }

  fit.ok                = true;
  fit.mode              = x;
  fit.hessian_chol      = chol_final;
  fit.log_joint_at_mode = cur_lj;
  return fit;
}

/// @brief Conditional fit and log hyperposterior, including prior normalization if supplied.
struct LaplaceEval {
  bool     ok{true};           ///< Whether the evaluation is usable.
  double   log_posterior{0.0}; ///< Log prior plus conditional Laplace log evidence.
  InnerFit inner;              ///< Converged latent mode and curvature.
};

/// @brief Evaluate the latent Gaussian approximation at a fixed hyperparameter vector.
/// @param D Validated observation design.
/// @param y Validated responses.
/// @param likelihood Concave likelihood callback.
/// @param precision SPD precision builder.
/// @param log_prior Hyperprior density on the supplied parameter scale.
/// @param theta Finite hyperparameter vector.
/// @param options Newton controls.
/// @return Failed evaluation for excluded priors, invalid precision or unconverged Newton fits.
LaplaceEval evaluate_laplace(
    const DenseMatrix<double>& D,
    const std::vector<double>& y,
    const INLALikelihoodFn&    likelihood,
    const INLAPrecisionFn&     precision,
    const INLALogPriorFn&      log_prior,
    const std::vector<double>& theta,
    const INLAOptions&         options) {
  LaplaceEval  result;
  const double lp = log_prior(theta);
  if (!std::isfinite(lp)) {
    result.ok = false;
    return result;
  }
  DenseMatrix<double> Q;
  try {
    Q = precision(theta);
  } catch (...) {
    result.ok = false;
    return result;
  }
  if (Q.rows() != Q.cols() || Q.rows() != D.cols()) {
    result.ok = false;
    return result;
  }

  for (std::size_t i = 0; i < Q.rows(); ++i)
    for (std::size_t j = 0; j < Q.cols(); ++j)
      if (!std::isfinite(Q(i, j)) ||
          std::abs(Q(i, j) - Q(j, i)) >
              1e-12 * std::max({1., std::abs(Q(i, j)), std::abs(Q(j, i))})) {
        result.ok = false;
        return result;
      }
  const auto chol_q = linalg::cholesky(Q);
  if (!chol_q.ok) {
    result.ok = false;
    return result;
  }

  InnerFit inner = newton_raphson_mode(D, y, likelihood, Q, theta, options);
  if (!inner.ok) {
    result.ok = false;
    return result;
  }

  result.ok            = true;
  result.log_posterior = lp + 0.5 * log_det_from_chol(chol_q) + inner.log_joint_at_mode -
                         0.5 * log_det_from_chol(inner.hessian_chol);
  result.inner = std::move(inner);
  return result;
}

/// @brief Differential-evolution objective for the negative approximate hyperposterior.
class ThetaObjective : public optim::ArbitraryFunction {
 public:
  /// @brief Borrow all model inputs for the duration of mode search.
  /// @param D Observation design.
  /// @param y Responses.
  /// @param likelihood Observation callback.
  /// @param precision Precision callback.
  /// @param log_prior Hyperprior callback.
  /// @param options Newton controls.
  ThetaObjective(
      const DenseMatrix<double>& D,
      const std::vector<double>& y,
      const INLALikelihoodFn&    likelihood,
      const INLAPrecisionFn&     precision,
      const INLALogPriorFn&      log_prior,
      const INLAOptions&         options)
      : D_(D),
        y_(y),
        likelihood_(likelihood),
        precision_(precision),
        log_prior_(log_prior),
        options_(options) {}

  /// @brief Return the negative log hyperposterior, or a large penalty for invalid fits.
  /// @param theta Hyperparameter vector.
  /// @return Scalar objective to minimize.
  double evaluate(const std::vector<double>& theta) override {
    const auto ev = evaluate_laplace(D_, y_, likelihood_, precision_, log_prior_, theta, options_);
    if (!ev.ok || !std::isfinite(ev.log_posterior))
      return std::numeric_limits<double>::max() / 4.0;
    return -ev.log_posterior; // DifferentialEvolution minimizes
  }

 private:
  const DenseMatrix<double>& D_;          ///< Borrowed observation design.
  const std::vector<double>& y_;          ///< Borrowed responses.
  const INLALikelihoodFn&    likelihood_; ///< Borrowed likelihood.
  const INLAPrecisionFn&     precision_;  ///< Borrowed precision builder.
  const INLALogPriorFn&      log_prior_;  ///< Borrowed hyperprior.
  const INLAOptions&         options_;    ///< Borrowed Newton controls.
};

/// @brief Enumerate tensor-grid tuples in odometer order.
/// @param theta_dim Number of axes.
/// @param points_per_dim Validated odd axis size.
/// @return Owned index tuples; first axis changes fastest.
std::vector<std::vector<long>> cartesian_indices(
    std::size_t theta_dim,
    std::size_t points_per_dim) {
  std::size_t total = 1;
  for (std::size_t d = 0; d < theta_dim; ++d)
    total *= points_per_dim;
  std::vector<std::vector<long>> result(total, std::vector<long>(theta_dim, 0));
  std::vector<long>              idx(theta_dim, 0);
  for (std::size_t point = 0; point < total; ++point) {
    result[point] = idx;
    for (std::size_t d = 0; d < theta_dim; ++d) {
      if (++idx[d] < static_cast<long>(points_per_dim))
        break;
      idx[d] = 0;
    }
  }
  return result;
}

} // namespace

INLA::INLA(INLAOptions options) : options_(options) {
  if (options_.strategy != INLAIntegrationStrategy::Grid &&
      options_.strategy != INLAIntegrationStrategy::EmpiricalBayes)
    throw std::invalid_argument("INLA: invalid strategy");
  if (!options_.newton_max_iterations || !std::isfinite(options_.newton_tolerance) ||
      options_.newton_tolerance <= 0 || options_.mode_population_size < 4 ||
      !options_.mode_max_generations || !std::isfinite(options_.grid_span) ||
      options_.grid_span <= 0)
    throw std::invalid_argument("INLA: invalid numerical controls");
  if (options_.strategy == INLAIntegrationStrategy::Grid &&
      (options_.grid_points_per_dim < 3 || options_.grid_points_per_dim % 2 == 0))
    throw std::invalid_argument("INLA: grid points must be odd and at least three");
}

INLAResult INLA::fit(
    const DenseMatrix<double>& design,
    const std::vector<double>& y,
    const INLALikelihoodFn&    likelihood,
    const INLAPrecisionFn&     precision,
    const INLALogPriorFn&      log_prior,
    const std::vector<double>& theta_lower,
    const std::vector<double>& theta_upper,
    const std::vector<double>& theta_init) const {
  if (design.rows() == 0 || design.cols() == 0 || !likelihood || !precision || !log_prior)
    throw std::invalid_argument("INLA::fit: nonempty model and callbacks required");
  for (std::size_t i = 0; i < design.rows(); ++i)
    for (std::size_t j = 0; j < design.cols(); ++j)
      if (!std::isfinite(design(i, j)))
        throw std::invalid_argument("INLA::fit: nonfinite design");
  for (double v : y)
    if (!std::isfinite(v))
      throw std::invalid_argument("INLA::fit: nonfinite response");
  if (design.rows() != y.size())
    throw std::invalid_argument("INLA::fit: design row count must match y size");
  const std::size_t theta_dim = theta_init.size();
  if (theta_dim == 0)
    throw std::invalid_argument("INLA::fit: theta_init must be non-empty");
  if (theta_lower.size() != theta_dim || theta_upper.size() != theta_dim)
    throw std::invalid_argument("INLA::fit: theta_lower/theta_upper must match theta_init's size");
  if (options_.strategy == INLAIntegrationStrategy::Grid && theta_dim > kMaxGridThetaDim)
    throw std::invalid_argument(
        "INLA::fit: theta_dim (" + std::to_string(theta_dim) +
        ") exceeds the practical grid-integration limit (" + std::to_string(kMaxGridThetaDim) +
        "); use INLAIntegrationStrategy::EmpiricalBayes for higher-dimensional theta");
  std::size_t requested = 1;
  for (std::size_t d = 0; d < theta_dim; ++d) {
    if (!std::isfinite(theta_lower[d]) || !std::isfinite(theta_upper[d]) ||
        !std::isfinite(theta_init[d]) || theta_lower[d] >= theta_upper[d] ||
        theta_init[d] < theta_lower[d] || theta_init[d] > theta_upper[d])
      throw std::invalid_argument("INLA::fit: invalid hyperparameter box/initial value");
    if (options_.strategy == INLAIntegrationStrategy::Grid) {
      if (options_.grid_points_per_dim > 1000000 / requested)
        throw std::invalid_argument("INLA::fit: grid exceeds one million points");
      requested *= options_.grid_points_per_dim;
    }
  }

  ThetaObjective      objective(design, y, likelihood, precision, log_prior, options_);
  std::vector<double> theta_mode = theta_init;
  optim::DEOptions    de_options;
  de_options.population_size = options_.mode_population_size;
  de_options.max_generations = options_.mode_max_generations;
  de_options.seed            = options_.seed;
  optim::DifferentialEvolution de(de_options);
  de.optimize(objective, theta_mode, theta_lower, theta_upper);

  const auto mode_eval =
      evaluate_laplace(design, y, likelihood, precision, log_prior, theta_mode, options_);
  if (!mode_eval.ok || !std::isfinite(mode_eval.log_posterior))
    throw std::runtime_error(
        "INLA::fit: failed to evaluate the Laplace approximation at the mode of theta");

  const std::size_t n_latent = design.cols();

  std::vector<std::vector<double>> grid_theta;
  std::vector<LaplaceEval>         grid_evals;
  double                           cell_volume         = 1.0;
  bool                             covariance_fallback = false;
  std::vector<bool>                edge_flags;
  std::vector<double>              cell_volumes;
  std::size_t                      skipped = 0;

  if (options_.strategy == INLAIntegrationStrategy::EmpiricalBayes) {
    grid_theta.push_back(theta_mode);
    grid_evals.push_back(mode_eval);
    edge_flags.push_back(false);
    cell_volumes.push_back(1);
  } else {
    DenseMatrix<double> neg_hess(theta_dim, theta_dim, 0.0);
    std::vector<double> h(theta_dim), direction(theta_dim, 0);
    // Curvature probes stay inside the declared box, including boundary modes.
    for (std::size_t d = 0; d < theta_dim; ++d) {
      h[d] = std::min(1e-3, .2 * (theta_upper[d] - theta_lower[d]));
      if (theta_mode[d] - h[d] < theta_lower[d])
        direction[d] = 1;
      else if (theta_mode[d] + h[d] > theta_upper[d])
        direction[d] = -1;
    }
    auto f = [&](const std::vector<double>& th) {
      const auto ev = evaluate_laplace(design, y, likelihood, precision, log_prior, th, options_);
      return ev.ok && std::isfinite(ev.log_posterior) ? -ev.log_posterior
                                                      : std::numeric_limits<double>::quiet_NaN();
    };
    const double f0 = f(theta_mode);
    for (std::size_t i = 0; i < theta_dim; ++i) {
      auto plus = theta_mode, minus = theta_mode;
      if (direction[i] == 0) {
        plus[i] += h[i];
        minus[i] -= h[i];
        neg_hess(i, i) = (f(plus) - 2 * f0 + f(minus)) / (h[i] * h[i]);
      } else {
        plus[i] += direction[i] * h[i];
        minus[i] += direction[i] * 2 * h[i];
        neg_hess(i, i) = (f0 - 2 * f(plus) + f(minus)) / (h[i] * h[i]);
      }
      for (std::size_t j = 0; j < i; ++j) {
        double       sum = 0;
        const double si  = direction[i] == 0 ? 1 : direction[i],
                     sj  = direction[j] == 0 ? 1 : direction[j];
        const double ai = direction[i] == 0 ? -1 : 0, aj = direction[j] == 0 ? -1 : 0;
        for (int ii = 0; ii < 2; ++ii)
          for (int jj = 0; jj < 2; ++jj) {
            auto th = theta_mode;
            th[i] += (ii ? si : ai) * h[i];
            th[j] += (jj ? sj : aj) * h[j];
            sum += (ii == jj ? 1 : -1) * f(th);
          }
        neg_hess(i, j) = neg_hess(j, i) = sum / ((si - ai) * h[i] * (sj - aj) * h[j]);
      }
    }
    bool finite_hessian = true;
    for (std::size_t i = 0; i < theta_dim; ++i)
      for (std::size_t j = 0; j < theta_dim; ++j)
        finite_hessian &= std::isfinite(neg_hess(i, j));
    const auto          chol_nh = linalg::cholesky(neg_hess);
    DenseMatrix<double> sigma_theta(theta_dim, theta_dim, 0.0);
    if (chol_nh.ok && finite_hessian) {
      sigma_theta = chol_nh.solve(DenseMatrix<double>::identity(theta_dim));
    } else {
      // Explicit fallback: one quarter of each box width per coordinate.
      covariance_fallback = true;
      for (std::size_t i = 0; i < theta_dim; ++i) {
        const double width = (theta_upper[i] - theta_lower[i]) / 4;
        sigma_theta(i, i)  = width * width;
      }
    }

    const auto        eig     = linalg::jacobi_eigen(sigma_theta);
    const std::size_t half    = (options_.grid_points_per_dim - 1) / 2;
    const double      delta_z = half > 0 ? options_.grid_span / static_cast<double>(half) : 0.0;

    // A near-zero (or, from finite-difference/Jacobi numerical noise, slightly negative)
    // eigenvalue is expected when the mode sits right at a variance-component boundary
    // (posterior curvature there is ill-behaved); floor it relative to the largest
    // eigenvalue rather than at literal zero, or det_l -- and every grid weight, via
    // cell_volume -- collapses to exactly zero and poisons the whole integral (a real bug
    // this floor fixes, not just defensive padding).
    double max_eigenvalue = 0.0;
    for (std::size_t d = 0; d < theta_dim; ++d)
      max_eigenvalue = std::max(max_eigenvalue, eig.eigenvalues[d]);
    const double min_eigenvalue     = std::max(max_eigenvalue * 1e-6, 1e-12);
    auto         floored_eigenvalue = [&](std::size_t d) {
      return std::max(eig.eigenvalues[d], min_eigenvalue);
    };

    double det_l = 1.0;
    for (std::size_t d = 0; d < theta_dim; ++d)
      det_l *= std::sqrt(floored_eigenvalue(d));
    cell_volume = std::pow(delta_z, static_cast<double>(theta_dim)) * det_l;

    for (const auto& idx : cartesian_indices(theta_dim, options_.grid_points_per_dim)) {
      bool                is_center = true;
      bool                edge      = false;
      std::vector<double> th        = theta_mode;
      for (std::size_t d = 0; d < theta_dim; ++d) {
        if (idx[d] != static_cast<long>(half))
          is_center = false;
        edge |= idx[d] == 0 || idx[d] == static_cast<long>(options_.grid_points_per_dim - 1);
        const double z     = static_cast<double>(idx[d] - static_cast<long>(half)) * delta_z;
        const double scale = std::sqrt(floored_eigenvalue(d));
        for (std::size_t j = 0; j < theta_dim; ++j)
          th[j] += z * scale * eig.eigenvectors(j, d);
      }

      double local_volume = cell_volume;
      if (theta_dim == 1) {
        const double left  = std::max(theta_lower[0], th[0] - .5 * cell_volume);
        const double right = std::min(theta_upper[0], th[0] + .5 * cell_volume);
        if (right <= left) {
          ++skipped;
          continue;
        }
        th[0]        = left / 2 + right / 2;
        local_volume = right - left;
        is_center    = is_center && th[0] == theta_mode[0];
      }
      bool inside = true;
      for (std::size_t d = 0; d < theta_dim; ++d)
        inside &= th[d] >= theta_lower[d] && th[d] <= theta_upper[d];
      if (!inside) {
        ++skipped;
        continue;
      }
      const LaplaceEval ev =
          is_center ? mode_eval
                    : evaluate_laplace(design, y, likelihood, precision, log_prior, th, options_);
      // Skip unusable grid points: loses positive definiteness near a boundary (ev.ok
      // false), or a caller's precision()/likelihood() produces a technically-PD-looking
      // but numerically degenerate matrix (e.g. from a near-zero variance component)
      // whose Cholesky solve silently yields Inf/NaN rather than failing outright.
      if (!ev.ok || !std::isfinite(ev.log_posterior)) {
        ++skipped;
        continue;
      }
      grid_theta.push_back(th);
      grid_evals.push_back(ev);
      edge_flags.push_back(edge);
      cell_volumes.push_back(local_volume);
    }
    if (grid_theta.empty())
      throw std::runtime_error("INLA::fit: every grid point around the theta mode was unusable");
  }

  double max_lp = -std::numeric_limits<double>::infinity();
  for (const auto& ev : grid_evals)
    max_lp = std::max(max_lp, ev.log_posterior);

  std::vector<double> raw_weights(grid_evals.size());
  double              sum_raw = 0.0;
  for (std::size_t k = 0; k < grid_evals.size(); ++k) {
    raw_weights[k] =
        std::exp(grid_evals[k].log_posterior - max_lp) * (cell_volumes[k] / cell_volume);
    sum_raw += raw_weights[k];
  }
  std::vector<double> weights(grid_evals.size());
  for (std::size_t k = 0; k < grid_evals.size(); ++k)
    weights[k] = raw_weights[k] / sum_raw;

  INLAResult out;
  out.theta_dim                = theta_dim;
  out.log_marginal_likelihood  = options_.strategy == INLAIntegrationStrategy::Grid
                                     ? max_lp + std::log(sum_raw) + std::log(cell_volume)
                                     : std::numeric_limits<double>::quiet_NaN();
  out.theta_mode               = theta_mode;
  out.log_posterior_at_mode    = mode_eval.log_posterior;
  out.grid_cell_volume         = cell_volume;
  out.theta_grid_cell_volumes  = cell_volumes;
  out.grid_covariance_fallback = covariance_fallback;
  out.grid_requested_points    = requested;
  out.grid_skipped_points      = skipped;
  out.latent_covariance        = DenseMatrix<double>(n_latent, n_latent, 0);
  out.latent_within_variance.assign(n_latent, 0);
  out.latent_between_variance.assign(n_latent, 0);
  for (std::size_t k = 0; k < grid_evals.size(); ++k) {
    out.conditional_means.push_back(grid_evals[k].inner.mode);
    out.conditional_covariances.push_back(
        grid_evals[k].inner.hessian_chol.solve(DenseMatrix<double>::identity(n_latent)));
    out.theta_log_posterior.push_back(grid_evals[k].log_posterior);
    out.conditional_iterations.push_back(grid_evals[k].inner.iterations);
    out.conditional_gradient_inf.push_back(grid_evals[k].inner.gradient_inf);
    if (edge_flags[k])
      out.grid_edge_mass += weights[k];
  }

  out.latent_mean.assign(n_latent, 0.0);
  for (std::size_t k = 0; k < grid_evals.size(); ++k)
    for (std::size_t j = 0; j < n_latent; ++j)
      out.latent_mean[j] += weights[k] * grid_evals[k].inner.mode[j];

  std::vector<double> latent_var(n_latent, 0.0);
  for (std::size_t k = 0; k < grid_evals.size(); ++k) {
    const auto& hinv = out.conditional_covariances[k];
    for (std::size_t j = 0; j < n_latent; ++j) {
      const double diff = grid_evals[k].inner.mode[j] - out.latent_mean[j];
      latent_var[j] += weights[k] * (hinv(j, j) + diff * diff);
      out.latent_within_variance[j] += weights[k] * hinv(j, j);
      out.latent_between_variance[j] += weights[k] * diff * diff;
      for (std::size_t l = 0; l < n_latent; ++l)
        out.latent_covariance(j, l) +=
            weights[k] * (hinv(j, l) + diff * (grid_evals[k].inner.mode[l] - out.latent_mean[l]));
    }
  }
  out.latent_sd.resize(n_latent);
  for (std::size_t j = 0; j < n_latent; ++j)
    out.latent_sd[j] = std::sqrt(std::max(0.0, latent_var[j]));

  out.theta_mean.assign(theta_dim, 0.0);
  for (std::size_t k = 0; k < grid_theta.size(); ++k)
    for (std::size_t d = 0; d < theta_dim; ++d)
      out.theta_mean[d] += weights[k] * grid_theta[k][d];
  std::vector<double> theta_var(theta_dim, 0.0);
  for (std::size_t k = 0; k < grid_theta.size(); ++k)
    for (std::size_t d = 0; d < theta_dim; ++d) {
      const double diff = grid_theta[k][d] - out.theta_mean[d];
      theta_var[d] += weights[k] * diff * diff;
    }
  out.theta_sd.resize(theta_dim);
  for (std::size_t d = 0; d < theta_dim; ++d)
    out.theta_sd[d] = std::sqrt(std::max(0.0, theta_var[d]));

  out.theta_grid.reserve(grid_theta.size() * theta_dim);
  for (const auto& th : grid_theta)
    out.theta_grid.insert(out.theta_grid.end(), th.begin(), th.end());
  out.theta_grid_weights = std::move(weights);

  return out;
}

INLAConditional INLA::conditional(
    const DenseMatrix<double>& design,
    const std::vector<double>& y,
    const INLALikelihoodFn&    likelihood,
    const DenseMatrix<double>& precision,
    const std::vector<double>& theta) const {
  if (design.rows() == 0 || design.cols() == 0 || design.rows() != y.size() || !likelihood)
    throw std::invalid_argument(
        "INLA::conditional: nonempty aligned design and likelihood required");
  for (std::size_t i = 0; i < design.rows(); ++i)
    for (std::size_t j = 0; j < design.cols(); ++j)
      if (!std::isfinite(design(i, j)))
        throw std::invalid_argument("INLA::conditional: nonfinite design");
  for (double v : y)
    if (!std::isfinite(v))
      throw std::invalid_argument("INLA::conditional: nonfinite response");
  for (double v : theta)
    if (!std::isfinite(v))
      throw std::invalid_argument("INLA::conditional: nonfinite theta");
  const auto eval = evaluate_laplace(
      design,
      y,
      likelihood,
      [&](const auto&) { return precision; },
      [](const auto&) { return 0.; },
      theta,
      options_);
  if (!eval.ok || !std::isfinite(eval.log_posterior))
    throw std::runtime_error("INLA::conditional: invalid precision or failed Newton convergence");
  return {
      eval.inner.mode,
      eval.inner.hessian_chol.solve(DenseMatrix<double>::identity(design.cols())),
      eval.log_posterior,
      eval.inner.iterations,
      eval.inner.gradient_inf};
}

INLALikelihoodFn inla_gaussian_likelihood(
    std::function<double(const std::vector<double>&)> sigma_of_theta) {
  if (!sigma_of_theta)
    throw std::invalid_argument("inla_gaussian_likelihood: sigma callback required");
  return [sigma_of_theta](double y, double eta, const std::vector<double>& theta) {
    if (!std::isfinite(y) || !std::isfinite(eta))
      throw std::invalid_argument("INLA Gaussian: finite y/eta required");
    const double sigma = sigma_of_theta(theta);
    if (!std::isfinite(sigma) || sigma <= 0 || !std::isfinite(sigma * sigma) || sigma * sigma == 0)
      throw std::invalid_argument("INLA Gaussian: finite positive representable variance required");
    INLALikelihoodPoint pt;
    pt.log_density = normal_lpdf(y, eta, sigma);
    pt.score       = (y - eta) / (sigma * sigma);
    pt.curvature   = 1.0 / (sigma * sigma);
    return pt;
  };
}

INLALikelihoodFn inla_binomial_logit_likelihood() {
  return [](double y, double eta, const std::vector<double>&) {
    if ((y != 0 && y != 1) || !std::isfinite(eta))
      throw std::invalid_argument("INLA Bernoulli: y must be zero or one and eta finite");
    const double p = eta >= 0 ? 1 / (1 + std::exp(-eta)) : std::exp(eta) / (1 + std::exp(eta));
    INLALikelihoodPoint pt;
    pt.log_density = y * eta - std::max(eta, 0.) - std::log1p(std::exp(-std::abs(eta)));
    pt.score       = y - p;
    const double e = std::exp(-std::abs(eta));
    pt.curvature   = e / ((1 + e) * (1 + e));
    return pt;
  };
}

INLALikelihoodFn inla_poisson_log_likelihood() {
  return [](double y, double eta, const std::vector<double>&) {
    if (!std::isfinite(y) || y < 0 || y != std::trunc(y) || !std::isfinite(eta))
      throw std::invalid_argument(
          "INLA Poisson: finite nonnegative integer count and finite eta required");
    const double        mu = std::exp(eta);
    INLALikelihoodPoint pt;
    pt.log_density = poisson_log_lpmf(y, eta);
    pt.score       = y - mu;
    pt.curvature   = mu;
    return pt;
  };
}

} // namespace datamunge::bayes
