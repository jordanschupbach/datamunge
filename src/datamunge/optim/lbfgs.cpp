#include <datamunge/optim/lbfgs.hpp>

#include <cmath>
#include <deque>
#include <limits>

namespace datamunge::optim {

LBFGS::LBFGS(LBFGSOptions options) : options_(options) {}

double LBFGS::optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const {
    const std::size_t n = coordinates.size();
    std::deque<std::vector<double>> s_history;
    std::deque<std::vector<double>> y_history;
    std::deque<double> rho_history;

    double value = function.evaluate(coordinates);
    std::vector<double> grad = function.gradient(coordinates);

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        double grad_norm_sq = 0.0;
        for (const double g : grad) grad_norm_sq += g * g;
        if (std::sqrt(grad_norm_sq) < options_.tolerance) break;

        // Two-loop recursion: computes direction = -H_k * grad using the last `m` (s, y) pairs.
        std::vector<double> q = grad;
        const std::size_t m = s_history.size();
        std::vector<double> alpha(m);
        for (std::size_t idx = m; idx-- > 0;) {
            double dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) dot += s_history[idx][j] * q[j];
            alpha[idx] = rho_history[idx] * dot;
            for (std::size_t j = 0; j < n; ++j) q[j] -= alpha[idx] * y_history[idx][j];
        }

        double gamma = 1.0;
        if (m > 0) {
            double sy = 0.0, yy = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                sy += s_history[m - 1][j] * y_history[m - 1][j];
                yy += y_history[m - 1][j] * y_history[m - 1][j];
            }
            if (yy > 0.0) gamma = sy / yy;
        }

        std::vector<double> r(n);
        for (std::size_t j = 0; j < n; ++j) r[j] = gamma * q[j];
        for (std::size_t idx = 0; idx < m; ++idx) {
            double dot = 0.0;
            for (std::size_t j = 0; j < n; ++j) dot += y_history[idx][j] * r[j];
            const double beta = rho_history[idx] * dot;
            for (std::size_t j = 0; j < n; ++j) r[j] += s_history[idx][j] * (alpha[idx] - beta);
        }

        std::vector<double> direction(n);
        for (std::size_t j = 0; j < n; ++j) direction[j] = -r[j];

        double dir_dot_grad = 0.0;
        for (std::size_t j = 0; j < n; ++j) dir_dot_grad += direction[j] * grad[j];
        if (dir_dot_grad > 0.0) { // numerical issue: fall back to steepest descent
            for (std::size_t j = 0; j < n; ++j) direction[j] = -grad[j];
            dir_dot_grad = -grad_norm_sq;
        }

        // Weak-Wolfe line search via bisection: shrinks the bracket [lo, hi] until a step
        // satisfies both the Armijo sufficient-decrease condition and the curvature
        // condition -- the standard pairing for L-BFGS (plain backtracking alone permits
        // steps too short to build a well-conditioned inverse-Hessian approximation).
        double lo = 0.0;
        double hi = std::numeric_limits<double>::infinity();
        double t = 1.0;
        std::vector<double> new_coords(n);
        std::vector<double> new_grad;
        double new_value = value;
        bool found = false;
        for (std::size_t trial = 0; trial < options_.max_line_search_trials; ++trial) {
            for (std::size_t j = 0; j < n; ++j) new_coords[j] = coordinates[j] + t * direction[j];
            new_value = function.evaluate(new_coords);
            if (new_value > value + options_.armijo_c1 * t * dir_dot_grad) {
                hi = t;
                t = 0.5 * (lo + hi);
                continue;
            }
            new_grad = function.gradient(new_coords);
            double new_dir_dot_grad = 0.0;
            for (std::size_t j = 0; j < n; ++j) new_dir_dot_grad += direction[j] * new_grad[j];
            if (new_dir_dot_grad < options_.wolfe_c2 * dir_dot_grad) {
                lo = t;
                t = std::isinf(hi) ? 2.0 * lo : 0.5 * (lo + hi);
                continue;
            }
            found = true;
            break;
        }
        if (!found) {
            if (lo == 0.0) break; // never found even an Armijo-improving step; give up
            t = lo;
            for (std::size_t j = 0; j < n; ++j) new_coords[j] = coordinates[j] + t * direction[j];
            new_value = function.evaluate(new_coords);
            new_grad = function.gradient(new_coords);
        }

        std::vector<double> s(n), y(n);
        double sy = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            s[j] = new_coords[j] - coordinates[j];
            y[j] = new_grad[j] - grad[j];
            sy += s[j] * y[j];
        }
        if (sy > 1e-12) { // curvature condition; skip the update otherwise
            if (s_history.size() >= options_.history_size) {
                s_history.pop_front();
                y_history.pop_front();
                rho_history.pop_front();
            }
            s_history.push_back(s);
            y_history.push_back(y);
            rho_history.push_back(1.0 / sy);
        }

        const bool converged = std::abs(new_value - value) < options_.tolerance;
        coordinates = new_coords;
        value = new_value;
        grad = new_grad;
        if (converged) break;
    }
    return value;
}

} // namespace datamunge::optim
