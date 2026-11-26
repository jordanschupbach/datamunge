/// @file hawkes.cpp
/// @brief Recursive likelihood, gradient, Newton fitting, thinning simulation and residuals of the
/// exponential Hawkes process.
#include <datamunge/stats/hawkes.hpp>

#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for Hawkes processes.
namespace {
/// @brief Checks event times and resolves the end time.
/// @param t Event times.
/// @param end End time (NaN for the last event).
/// @return The end time.
double check_times(const std::vector<double>& t, double end) {
    for (std::size_t k = 0; k < t.size(); ++k)
        if (!std::isfinite(t[k]) || !(t[k] > 0) || (k > 0 && !(t[k] > t[k - 1])))
            throw std::invalid_argument("hawkes: times must be positive, finite and strictly increasing");
    const double tt = std::isnan(end) ? (t.empty() ? 0.0 : t.back()) : end;
    if (!(tt > 0) || !std::isfinite(tt) || (!t.empty() && tt < t.back()))
        throw std::invalid_argument("hawkes: the end time must be positive and not before the last event");
    return tt;
}

/// @brief Checks that the parameters are positive and finite.
/// @param p Parameters.
void check_params(const HawkesParameters& p) {
    if (!(p.mu > 0) || !(p.alpha > 0) || !(p.beta > 0) || !std::isfinite(p.mu) || !std::isfinite(p.alpha) ||
        !std::isfinite(p.beta))
        throw std::invalid_argument("hawkes: parameters must be positive and finite");
}

/// @brief Log-likelihood and its gradient in (mu, alpha, beta).
/// @param t Event times.
/// @param tt End time.
/// @param p Parameters.
/// @param grad Receives the gradient when non-null.
/// @return The log-likelihood.
double value_and_gradient(const std::vector<double>& t, double tt, const HawkesParameters& p,
                          std::array<double, 3>* grad) {
    double ll = -p.mu * tt, a = 0, b = 0, g_mu = -tt, g_alpha = 0, g_beta = 0;
    for (std::size_t k = 0; k < t.size(); ++k) {
        if (k > 0) {
            const double d = t[k] - t[k - 1], e = std::exp(-p.beta * d);
            b = e * (b - d * (1 + a)); // d A_k / d beta
            a = e * (1 + a);
        }
        const double lambda = p.mu + p.alpha * a;
        ll += std::log(lambda);
        g_mu += 1 / lambda;
        g_alpha += a / lambda;
        g_beta += p.alpha * b / lambda;
        const double tail = tt - t[k], e = std::exp(-p.beta * tail);
        ll -= p.alpha / p.beta * (1 - e);
        g_alpha -= (1 - e) / p.beta;
        g_beta += p.alpha / (p.beta * p.beta) * (1 - e) - p.alpha / p.beta * tail * e;
    }
    if (grad)
        *grad = {g_mu, g_alpha, g_beta};
    return ll;
}

/// @brief Hessian of the log-likelihood in the given coordinates by central differences of the gradient.
/// @param grad_at Gradient as a function of the coordinates.
/// @param x Coordinates.
/// @return The symmetric 3-by-3 Hessian.
template <typename GradientAt>
std::array<std::array<double, 3>, 3> hessian(const GradientAt& grad_at, const std::array<double, 3>& x) {
    std::array<std::array<double, 3>, 3> h{};
    for (std::size_t j = 0; j < 3; ++j) {
        const double step = 1e-5 * std::max(1.0, std::abs(x[j]));
        auto up = x, down = x;
        up[j] += step;
        down[j] -= step;
        const auto gu = grad_at(up), gd = grad_at(down);
        for (std::size_t i = 0; i < 3; ++i)
            h[i][j] = (gu[i] - gd[i]) / (2 * step);
    }
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < i; ++j)
            h[i][j] = h[j][i] = (h[i][j] + h[j][i]) / 2;
    return h;
}

/// @brief Solves a 3-by-3 linear system by Gaussian elimination with partial pivoting.
/// @param a Matrix.
/// @param b Right-hand side.
/// @param x Receives the solution.
/// @return False if the matrix is singular.
bool solve3(std::array<std::array<double, 3>, 3> a, std::array<double, 3> b, std::array<double, 3>& x) {
    for (std::size_t c = 0; c < 3; ++c) {
        std::size_t piv = c;
        for (std::size_t r = c + 1; r < 3; ++r)
            if (std::abs(a[r][c]) > std::abs(a[piv][c]))
                piv = r;
        if (std::abs(a[piv][c]) < 1e-300)
            return false;
        std::swap(a[c], a[piv]);
        std::swap(b[c], b[piv]);
        for (std::size_t r = c + 1; r < 3; ++r) {
            const double f = a[r][c] / a[c][c];
            for (std::size_t k = c; k < 3; ++k)
                a[r][k] -= f * a[c][k];
            b[r] -= f * b[c];
        }
    }
    for (std::size_t c = 3; c-- > 0;) {
        double s = b[c];
        for (std::size_t k = c + 1; k < 3; ++k)
            s -= a[c][k] * x[k];
        x[c] = s / a[c][c];
    }
    return true;
}

/// @brief Parameters from log coordinates.
/// @param th Log parameters.
/// @return The parameters.
HawkesParameters from_log(const std::array<double, 3>& th) {
    return {.mu = std::exp(th[0]), .alpha = std::exp(th[1]), .beta = std::exp(th[2])};
}
} // namespace

double hawkes_log_likelihood(const std::vector<double>& times, const HawkesParameters& params, double end_time) {
    check_params(params);
    const double tt = check_times(times, end_time);
    return value_and_gradient(times, tt, params, nullptr);
}

HawkesFit hawkes_fit(const std::vector<double>& times, const HawkesFitOptions& options) {
    if (times.size() < 3)
        throw std::invalid_argument("hawkes_fit: need at least three events");
    const double tt = check_times(times, options.end_time);
    const double n = static_cast<double>(times.size());
    const auto log_value = [&](const std::array<double, 3>& th) {
        return value_and_gradient(times, tt, from_log(th), nullptr);
    };
    const auto log_grad = [&](const std::array<double, 3>& th) {
        std::array<double, 3> g{};
        const HawkesParameters p = from_log(th);
        value_and_gradient(times, tt, p, &g);
        return std::array<double, 3>{g[0] * p.mu, g[1] * p.alpha, g[2] * p.beta};
    };
    HawkesFit best;
    best.log_likelihood = -std::numeric_limits<double>::infinity();
    std::array<double, 3> best_theta{};
    for (const double ratio : {0.2, 0.5, 0.8})
        for (const double scale : {0.3, 1.0, 3.0}) {
            const double beta = scale * n / tt;
            std::array<double, 3> th{std::log((1 - ratio) * n / tt), std::log(ratio * beta), std::log(beta)};
            double cur = log_value(th);
            std::size_t it = 0;
            bool converged = false;
            for (; it < options.max_iterations; ++it) {
                const auto g = log_grad(th);
                if (std::abs(g[0]) + std::abs(g[1]) + std::abs(g[2]) < 1e-8 * (1 + std::abs(cur))) {
                    converged = true;
                    break;
                }
                auto h = hessian(log_grad, th);
                // Damped Newton: solve (-H + tau I) d = g, raising tau until the likelihood increases.
                bool improved = false;
                for (double tau = 0; tau < 1e12; tau = tau == 0 ? 1e-6 : tau * 10) {
                    auto a = h;
                    for (std::size_t i = 0; i < 3; ++i) {
                        for (std::size_t j = 0; j < 3; ++j)
                            a[i][j] = -a[i][j];
                        a[i][i] += tau;
                    }
                    std::array<double, 3> d{};
                    if (!solve3(a, g, d))
                        continue;
                    double len = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    if (len > 2)
                        for (double& v : d)
                            v *= 2 / len;
                    auto next = th;
                    for (std::size_t i = 0; i < 3; ++i)
                        next[i] += d[i];
                    const double val = log_value(next);
                    if (std::isfinite(val) && val >= cur) {
                        th = next;
                        improved = val > cur;
                        cur = val;
                        break;
                    }
                }
                if (!improved) {
                    const auto g2 = log_grad(th);
                    converged = std::abs(g2[0]) + std::abs(g2[1]) + std::abs(g2[2]) < 1e-5 * (1 + std::abs(cur));
                    break;
                }
            }
            if (cur > best.log_likelihood) {
                best.log_likelihood = cur;
                best.iterations = it;
                best.converged = converged;
                best_theta = th;
            }
        }
    best.parameters = from_log(best_theta);
    best.branching_ratio = best.parameters.alpha / best.parameters.beta;
    best.aic = -2 * best.log_likelihood + 6;
    // Observed information on the natural scale.
    const auto nat_grad = [&](const std::array<double, 3>& x) {
        std::array<double, 3> g{};
        value_and_gradient(times, tt, {.mu = x[0], .alpha = x[1], .beta = x[2]}, &g);
        return g;
    };
    const std::array<double, 3> x{best.parameters.mu, best.parameters.alpha, best.parameters.beta};
    auto info = hessian(nat_grad, x);
    for (auto& row : info)
        for (double& v : row)
            v = -v;
    for (std::size_t j = 0; j < 3; ++j) {
        std::array<double, 3> e{}, col{};
        e[j] = 1;
        best.standard_errors[j] =
            solve3(info, e, col) && col[j] > 0 ? std::sqrt(col[j]) : std::numeric_limits<double>::quiet_NaN();
    }
    return best;
}

std::vector<double> hawkes_simulate(const HawkesParameters& params, double end_time, std::uint64_t seed) {
    check_params(params);
    if (!(end_time > 0) || !std::isfinite(end_time))
        throw std::invalid_argument("hawkes_simulate: the horizon must be positive and finite");
    random::SplitMix64 rng(seed);
    std::vector<double> out;
    double t = 0, excitation = 0; // excitation = sum alpha exp(-beta (t - t_i)) at time t
    while (true) {
        const double bound = params.mu + excitation; // intensity only decays until the next event
        const double wait = -std::log(1 - rng.uniform01()) / bound;
        excitation *= std::exp(-params.beta * wait);
        t += wait;
        if (t > end_time)
            break;
        if (rng.uniform01() * bound <= params.mu + excitation) {
            out.push_back(t);
            excitation += params.alpha;
        }
    }
    return out;
}

std::vector<double> hawkes_residuals(const std::vector<double>& times, const HawkesParameters& params) {
    check_params(params);
    check_times(times, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> out;
    double prev = 0, excitation = 0; // excitation just after the previous event
    for (const double t : times) {
        const double d = t - prev;
        out.push_back(params.mu * d + excitation / params.beta * (1 - std::exp(-params.beta * d)));
        excitation = excitation * std::exp(-params.beta * d) + params.alpha;
        prev = t;
    }
    return out;
}

} // namespace datamunge::stats
