#include <datamunge/bayes/inla.hpp>

#include <datamunge/bayes/distributions.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/optim/differential_evolution.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::bayes {

namespace {

using linalg::CholeskyDecomposition;
using linalg::DenseMatrix;

// Grid integration's cost is grid_points_per_dim^theta_dim; beyond this, EmpiricalBayes is
// the only practical strategy (matching R-INLA's own documented advice to switch away from
// "grid" once there are more than a handful of hyperparameters).
constexpr std::size_t kMaxGridThetaDim = 4;

double log_det_from_chol(const CholeskyDecomposition<double>& chol) {
    double s = 0.0;
    for (std::size_t i = 0; i < chol.L.rows(); ++i) s += 2.0 * std::log(chol.L(i, i));
    return s;
}

struct InnerFit {
    bool ok{true};
    std::vector<double> mode;                  // x*(theta)
    CholeskyDecomposition<double> hessian_chol; // Cholesky of H* = Q + D' diag(curvature) D
    double log_joint_at_mode{0.0};              // -0.5 x*'Qx* + sum_i log_density_i(eta_i*)
};

std::vector<double> mat_vec(const DenseMatrix<double>& D, const std::vector<double>& x) {
    std::vector<double> out(D.rows(), 0.0);
    for (std::size_t i = 0; i < D.rows(); ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < D.cols(); ++j) s += D(i, j) * x[j];
        out[i] = s;
    }
    return out;
}

// Newton-Raphson mode-finding for x in log p(x|y,theta) = -0.5 x'Qx + sum_i log p(y_i|eta_i),
// eta = D*x, with a simple step-halving safeguard against overshoot.
InnerFit newton_raphson_mode(const DenseMatrix<double>& D, const std::vector<double>& y,
                              const INLALikelihoodFn& likelihood, const DenseMatrix<double>& Q,
                              const std::vector<double>& theta, const INLAOptions& options) {
    const std::size_t n_obs = D.rows();
    const std::size_t n_latent = D.cols();

    std::vector<double> x(n_latent, 0.0);
    std::vector<double> eta = mat_vec(D, x);

    auto log_joint = [&](const std::vector<double>& xv, const std::vector<double>& etav) {
        double quad = 0.0;
        for (std::size_t j = 0; j < n_latent; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n_latent; ++k) s += Q(j, k) * xv[k];
            quad += xv[j] * s;
        }
        double ll = 0.0;
        for (std::size_t i = 0; i < n_obs; ++i) ll += likelihood(y[i], etav[i], theta).log_density;
        return -0.5 * quad + ll;
    };

    auto build_A = [&](const std::vector<double>& curvature) {
        DenseMatrix<double> A = Q;
        for (std::size_t i = 0; i < n_obs; ++i) {
            const double c = curvature[i];
            if (c == 0.0) continue;
            for (std::size_t j = 0; j < n_latent; ++j) {
                const double dij = D(i, j) * c;
                if (dij == 0.0) continue;
                for (std::size_t k = 0; k < n_latent; ++k) A(j, k) += dij * D(i, k);
            }
        }
        return A;
    };

    double cur_lj = log_joint(x, eta);
    InnerFit fit;

    for (std::size_t iter = 0; iter < options.newton_max_iterations; ++iter) {
        std::vector<double> score(n_obs), curvature(n_obs);
        for (std::size_t i = 0; i < n_obs; ++i) {
            const auto pt = likelihood(y[i], eta[i], theta);
            score[i] = pt.score;
            curvature[i] = pt.curvature;
        }

        const DenseMatrix<double> A = build_A(curvature);

        std::vector<double> Qx(n_latent, 0.0);
        for (std::size_t j = 0; j < n_latent; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n_latent; ++k) s += Q(j, k) * x[k];
            Qx[j] = s;
        }
        std::vector<double> rhs(n_latent, 0.0);
        for (std::size_t i = 0; i < n_obs; ++i) {
            const double si = score[i];
            if (si == 0.0) continue;
            for (std::size_t j = 0; j < n_latent; ++j) rhs[j] += D(i, j) * si;
        }
        for (std::size_t j = 0; j < n_latent; ++j) rhs[j] -= Qx[j];

        const auto chol_a = linalg::cholesky(A);
        if (!chol_a.ok) {
            fit.ok = false;
            return fit;
        }
        const std::vector<double> delta = chol_a.solve(rhs);

        double step_scale = 1.0;
        std::vector<double> x_new = x;
        std::vector<double> eta_new = eta;
        double new_lj = cur_lj;
        for (int half = 0; half < 10; ++half) {
            for (std::size_t j = 0; j < n_latent; ++j) x_new[j] = x[j] + step_scale * delta[j];
            eta_new = mat_vec(D, x_new);
            new_lj = log_joint(x_new, eta_new);
            if (new_lj >= cur_lj - 1e-10 || half == 9) break;
            step_scale *= 0.5;
        }

        double max_abs_delta = 0.0;
        for (std::size_t j = 0; j < n_latent; ++j) max_abs_delta = std::max(max_abs_delta, std::abs(x_new[j] - x[j]));

        x = x_new;
        eta = eta_new;
        cur_lj = new_lj;
        if (max_abs_delta < options.newton_tolerance) break;
    }

    std::vector<double> final_curvature(n_obs);
    for (std::size_t i = 0; i < n_obs; ++i) final_curvature[i] = likelihood(y[i], eta[i], theta).curvature;
    const auto chol_final = linalg::cholesky(build_A(final_curvature));
    if (!chol_final.ok) {
        fit.ok = false;
        return fit;
    }

    fit.ok = true;
    fit.mode = x;
    fit.hessian_chol = chol_final;
    fit.log_joint_at_mode = cur_lj;
    return fit;
}

struct LaplaceEval {
    bool ok{true};
    double log_posterior{0.0}; // log pi_tilde(theta|y), up to a theta-independent constant
    InnerFit inner;
};

LaplaceEval evaluate_laplace(const DenseMatrix<double>& D, const std::vector<double>& y,
                              const INLALikelihoodFn& likelihood, const INLAPrecisionFn& precision,
                              const INLALogPriorFn& log_prior, const std::vector<double>& theta,
                              const INLAOptions& options) {
    LaplaceEval result;
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

    result.ok = true;
    result.log_posterior =
        log_prior(theta) + 0.5 * log_det_from_chol(chol_q) + inner.log_joint_at_mode - 0.5 * log_det_from_chol(inner.hessian_chol);
    result.inner = std::move(inner);
    return result;
}

class ThetaObjective : public optim::ArbitraryFunction {
public:
    ThetaObjective(const DenseMatrix<double>& D, const std::vector<double>& y, const INLALikelihoodFn& likelihood,
                    const INLAPrecisionFn& precision, const INLALogPriorFn& log_prior, const INLAOptions& options)
        : D_(D), y_(y), likelihood_(likelihood), precision_(precision), log_prior_(log_prior), options_(options) {}

    double evaluate(const std::vector<double>& theta) override {
        const auto ev = evaluate_laplace(D_, y_, likelihood_, precision_, log_prior_, theta, options_);
        if (!ev.ok || !std::isfinite(ev.log_posterior)) return std::numeric_limits<double>::max() / 4.0;
        return -ev.log_posterior; // DifferentialEvolution minimizes
    }

private:
    const DenseMatrix<double>& D_;
    const std::vector<double>& y_;
    const INLALikelihoodFn& likelihood_;
    const INLAPrecisionFn& precision_;
    const INLALogPriorFn& log_prior_;
    const INLAOptions& options_;
};

// All grid_points_per_dim^theta_dim index tuples, odometer order.
std::vector<std::vector<long>> cartesian_indices(std::size_t theta_dim, std::size_t points_per_dim) {
    std::size_t total = 1;
    for (std::size_t d = 0; d < theta_dim; ++d) total *= points_per_dim;
    std::vector<std::vector<long>> result(total, std::vector<long>(theta_dim, 0));
    std::vector<long> idx(theta_dim, 0);
    for (std::size_t point = 0; point < total; ++point) {
        result[point] = idx;
        for (std::size_t d = 0; d < theta_dim; ++d) {
            if (++idx[d] < static_cast<long>(points_per_dim)) break;
            idx[d] = 0;
        }
    }
    return result;
}

} // namespace

INLA::INLA(INLAOptions options) : options_(options) {}

INLAResult INLA::fit(const DenseMatrix<double>& design, const std::vector<double>& y, const INLALikelihoodFn& likelihood,
                      const INLAPrecisionFn& precision, const INLALogPriorFn& log_prior,
                      const std::vector<double>& theta_lower, const std::vector<double>& theta_upper,
                      const std::vector<double>& theta_init) const {
    if (design.rows() != y.size()) throw std::invalid_argument("INLA::fit: design row count must match y size");
    const std::size_t theta_dim = theta_init.size();
    if (theta_dim == 0) throw std::invalid_argument("INLA::fit: theta_init must be non-empty");
    if (theta_lower.size() != theta_dim || theta_upper.size() != theta_dim)
        throw std::invalid_argument("INLA::fit: theta_lower/theta_upper must match theta_init's size");
    if (options_.strategy == INLAIntegrationStrategy::Grid && theta_dim > kMaxGridThetaDim)
        throw std::invalid_argument("INLA::fit: theta_dim (" + std::to_string(theta_dim) +
                                     ") exceeds the practical grid-integration limit (" + std::to_string(kMaxGridThetaDim) +
                                     "); use INLAIntegrationStrategy::EmpiricalBayes for higher-dimensional theta");
    if (options_.grid_points_per_dim % 2 == 0) throw std::invalid_argument("INLA::fit: grid_points_per_dim must be odd");

    ThetaObjective objective(design, y, likelihood, precision, log_prior, options_);
    std::vector<double> theta_mode = theta_init;
    optim::DEOptions de_options;
    de_options.population_size = options_.mode_population_size;
    de_options.max_generations = options_.mode_max_generations;
    de_options.seed = options_.seed;
    optim::DifferentialEvolution de(de_options);
    de.optimize(objective, theta_mode, theta_lower, theta_upper);

    const auto mode_eval = evaluate_laplace(design, y, likelihood, precision, log_prior, theta_mode, options_);
    if (!mode_eval.ok || !std::isfinite(mode_eval.log_posterior))
        throw std::runtime_error("INLA::fit: failed to evaluate the Laplace approximation at the mode of theta");

    const std::size_t n_latent = design.cols();

    std::vector<std::vector<double>> grid_theta;
    std::vector<LaplaceEval> grid_evals;
    double cell_volume = 1.0;

    if (options_.strategy == INLAIntegrationStrategy::EmpiricalBayes) {
        grid_theta.push_back(theta_mode);
        grid_evals.push_back(mode_eval);
    } else {
        DenseMatrix<double> neg_hess(theta_dim, theta_dim, 0.0);
        const double h = 1e-3;
        auto neg_log_post = [&](const std::vector<double>& th) {
            const auto ev = evaluate_laplace(design, y, likelihood, precision, log_prior, th, options_);
            return (ev.ok && std::isfinite(ev.log_posterior)) ? -ev.log_posterior : std::numeric_limits<double>::max() / 4.0;
        };
        const double f0 = neg_log_post(theta_mode);
        for (std::size_t i = 0; i < theta_dim; ++i) {
            for (std::size_t j = i; j < theta_dim; ++j) {
                double d2;
                if (i == j) {
                    std::vector<double> plus = theta_mode, minus = theta_mode;
                    plus[i] += h;
                    minus[i] -= h;
                    d2 = (neg_log_post(plus) - 2.0 * f0 + neg_log_post(minus)) / (h * h);
                } else {
                    std::vector<double> pp = theta_mode, pm = theta_mode, mp = theta_mode, mm = theta_mode;
                    pp[i] += h;
                    pp[j] += h;
                    pm[i] += h;
                    pm[j] -= h;
                    mp[i] -= h;
                    mp[j] += h;
                    mm[i] -= h;
                    mm[j] -= h;
                    d2 = (neg_log_post(pp) - neg_log_post(pm) - neg_log_post(mp) + neg_log_post(mm)) / (4.0 * h * h);
                }
                neg_hess(i, j) = d2;
                neg_hess(j, i) = d2;
            }
        }

        const auto chol_nh = linalg::cholesky(neg_hess);
        DenseMatrix<double> sigma_theta(theta_dim, theta_dim, 0.0);
        if (chol_nh.ok) {
            sigma_theta = chol_nh.solve(DenseMatrix<double>::identity(theta_dim));
        } else {
            // Numerical Hessian wasn't SPD (flat posterior / boundary mode) -- fall back to
            // a unit covariance so the grid still spans a sensible region around the mode.
            for (std::size_t i = 0; i < theta_dim; ++i) sigma_theta(i, i) = 1.0;
        }

        const auto eig = linalg::jacobi_eigen(sigma_theta);
        const std::size_t half = (options_.grid_points_per_dim - 1) / 2;
        const double delta_z = half > 0 ? options_.grid_span / static_cast<double>(half) : 0.0;

        // A near-zero (or, from finite-difference/Jacobi numerical noise, slightly negative)
        // eigenvalue is expected when the mode sits right at a variance-component boundary
        // (posterior curvature there is ill-behaved); floor it relative to the largest
        // eigenvalue rather than at literal zero, or det_l -- and every grid weight, via
        // cell_volume -- collapses to exactly zero and poisons the whole integral (a real bug
        // this floor fixes, not just defensive padding).
        double max_eigenvalue = 0.0;
        for (std::size_t d = 0; d < theta_dim; ++d) max_eigenvalue = std::max(max_eigenvalue, eig.eigenvalues[d]);
        const double min_eigenvalue = std::max(max_eigenvalue * 1e-6, 1e-12);
        auto floored_eigenvalue = [&](std::size_t d) { return std::max(eig.eigenvalues[d], min_eigenvalue); };

        double det_l = 1.0;
        for (std::size_t d = 0; d < theta_dim; ++d) det_l *= std::sqrt(floored_eigenvalue(d));
        cell_volume = std::pow(delta_z, static_cast<double>(theta_dim)) * det_l;

        for (const auto& idx : cartesian_indices(theta_dim, options_.grid_points_per_dim)) {
            bool is_center = true;
            std::vector<double> th = theta_mode;
            for (std::size_t d = 0; d < theta_dim; ++d) {
                if (idx[d] != static_cast<long>(half)) is_center = false;
                const double z = static_cast<double>(idx[d] - static_cast<long>(half)) * delta_z;
                const double scale = std::sqrt(floored_eigenvalue(d));
                for (std::size_t j = 0; j < theta_dim; ++j) th[j] += z * scale * eig.eigenvectors(j, d);
            }

            const LaplaceEval ev = is_center ? mode_eval : evaluate_laplace(design, y, likelihood, precision, log_prior, th, options_);
            // Skip unusable grid points: loses positive definiteness near a boundary (ev.ok
            // false), or a caller's precision()/likelihood() produces a technically-PD-looking
            // but numerically degenerate matrix (e.g. from a near-zero variance component)
            // whose Cholesky solve silently yields Inf/NaN rather than failing outright.
            if (!ev.ok || !std::isfinite(ev.log_posterior)) continue;
            grid_theta.push_back(th);
            grid_evals.push_back(ev);
        }
        if (grid_theta.empty()) throw std::runtime_error("INLA::fit: every grid point around the theta mode was unusable");
    }

    double max_lp = -std::numeric_limits<double>::infinity();
    for (const auto& ev : grid_evals) max_lp = std::max(max_lp, ev.log_posterior);

    std::vector<double> raw_weights(grid_evals.size());
    double sum_raw = 0.0;
    for (std::size_t k = 0; k < grid_evals.size(); ++k) {
        raw_weights[k] = std::exp(grid_evals[k].log_posterior - max_lp) * cell_volume;
        sum_raw += raw_weights[k];
    }
    std::vector<double> weights(grid_evals.size());
    for (std::size_t k = 0; k < grid_evals.size(); ++k) weights[k] = raw_weights[k] / sum_raw;

    INLAResult out;
    out.theta_dim = theta_dim;
    out.log_marginal_likelihood = max_lp + std::log(sum_raw);

    out.latent_mean.assign(n_latent, 0.0);
    for (std::size_t k = 0; k < grid_evals.size(); ++k)
        for (std::size_t j = 0; j < n_latent; ++j) out.latent_mean[j] += weights[k] * grid_evals[k].inner.mode[j];

    std::vector<double> latent_var(n_latent, 0.0);
    for (std::size_t k = 0; k < grid_evals.size(); ++k) {
        const DenseMatrix<double> hinv = grid_evals[k].inner.hessian_chol.solve(DenseMatrix<double>::identity(n_latent));
        for (std::size_t j = 0; j < n_latent; ++j) {
            const double diff = grid_evals[k].inner.mode[j] - out.latent_mean[j];
            latent_var[j] += weights[k] * (hinv(j, j) + diff * diff);
        }
    }
    out.latent_sd.resize(n_latent);
    for (std::size_t j = 0; j < n_latent; ++j) out.latent_sd[j] = std::sqrt(std::max(0.0, latent_var[j]));

    out.theta_mean.assign(theta_dim, 0.0);
    for (std::size_t k = 0; k < grid_theta.size(); ++k)
        for (std::size_t d = 0; d < theta_dim; ++d) out.theta_mean[d] += weights[k] * grid_theta[k][d];
    std::vector<double> theta_var(theta_dim, 0.0);
    for (std::size_t k = 0; k < grid_theta.size(); ++k)
        for (std::size_t d = 0; d < theta_dim; ++d) {
            const double diff = grid_theta[k][d] - out.theta_mean[d];
            theta_var[d] += weights[k] * diff * diff;
        }
    out.theta_sd.resize(theta_dim);
    for (std::size_t d = 0; d < theta_dim; ++d) out.theta_sd[d] = std::sqrt(std::max(0.0, theta_var[d]));

    out.theta_grid.reserve(grid_theta.size() * theta_dim);
    for (const auto& th : grid_theta) out.theta_grid.insert(out.theta_grid.end(), th.begin(), th.end());
    out.theta_grid_weights = std::move(weights);

    return out;
}

INLALikelihoodFn inla_gaussian_likelihood(std::function<double(const std::vector<double>&)> sigma_of_theta) {
    return [sigma_of_theta](double y, double eta, const std::vector<double>& theta) {
        const double sigma = sigma_of_theta(theta);
        INLALikelihoodPoint pt;
        pt.log_density = normal_lpdf(y, eta, sigma);
        pt.score = (y - eta) / (sigma * sigma);
        pt.curvature = 1.0 / (sigma * sigma);
        return pt;
    };
}

INLALikelihoodFn inla_binomial_logit_likelihood() {
    return [](double y, double eta, const std::vector<double>&) {
        const double p = 1.0 / (1.0 + std::exp(-eta));
        INLALikelihoodPoint pt;
        pt.log_density = bernoulli_logit_lpmf(y, eta);
        pt.score = y - p;
        pt.curvature = p * (1.0 - p);
        return pt;
    };
}

INLALikelihoodFn inla_poisson_log_likelihood() {
    return [](double y, double eta, const std::vector<double>&) {
        const double mu = std::exp(eta);
        INLALikelihoodPoint pt;
        pt.log_density = poisson_log_lpmf(y, eta);
        pt.score = y - mu;
        pt.curvature = mu;
        return pt;
    };
}

} // namespace datamunge::bayes
