/// @file gmm_estimation.cpp
/// @brief Gauss-Newton minimisation of GMM quadratic forms, weighting updates and inference.
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/gmm_estimation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::stats {
/// @brief Implementation helpers local to this translation unit.
namespace {
/// @brief Dense row-major matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Evaluates and checks the moment matrix.
/// @param moments Moment function.
/// @param theta Parameters.
/// @param q Expected moment count (0 accepts any).
/// @param n Expected row count (0 accepts any).
/// @return The n-by-q matrix.
/// @throws std::invalid_argument For a shape change.
/// @throws std::runtime_error For nonfinite entries.
Matrix evaluate(const MomentFunction& moments, const std::vector<double>& theta, std::size_t q, std::size_t n) {
    Matrix g = moments(theta);
    if ((q != 0 && g.cols() != q) || (n != 0 && g.rows() != n))
        throw std::invalid_argument("generalized_method_of_moments: the moment matrix changed shape");
    if (!std::ranges::all_of(g.data(), [](double x) { return std::isfinite(x); }))
        throw std::runtime_error("generalized_method_of_moments: nonfinite moments");
    return g;
}

/// @brief Column means of a moment matrix.
/// @param g n-by-q moments.
/// @return gbar.
std::vector<double> column_means(const Matrix& g) {
    std::vector<double> m(g.cols(), 0.0);
    for (std::size_t i = 0; i < g.rows(); ++i)
        for (std::size_t j = 0; j < g.cols(); ++j)
            m[j] += g(i, j);
    for (double& v : m)
        v /= static_cast<double>(g.rows());
    return m;
}

/// @brief Moment covariance S, centred or not, divided by n.
/// @param g n-by-q moments.
/// @param centered Subtract column means first.
/// @return q-by-q S.
Matrix moment_covariance(const Matrix& g, bool centered) {
    const std::size_t n = g.rows(), q = g.cols();
    std::vector<double> mean(q, 0.0);
    if (centered)
        mean = column_means(g);
    Matrix s(q, q, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t a = 0; a < q; ++a)
            for (std::size_t b = 0; b <= a; ++b)
                s(a, b) += (g(i, a) - mean[a]) * (g(i, b) - mean[b]);
    for (std::size_t a = 0; a < q; ++a)
        for (std::size_t b = 0; b <= a; ++b)
            s(b, a) = s(a, b) = s(a, b) / static_cast<double>(n);
    return s;
}

/// @brief Inverse of a symmetric positive definite matrix through its Cholesky factor.
/// @param a SPD matrix.
/// @param what Name for the error message.
/// @return The inverse.
/// @throws std::runtime_error When the matrix is not numerically positive definite.
Matrix spd_inverse(const Matrix& a, const char* what) {
    const auto f = linalg::cholesky(a);
    if (!f.ok)
        throw std::runtime_error(std::string("generalized_method_of_moments: singular ") + what);
    const std::size_t m = a.rows();
    Matrix inv(m, m, 0.0);
    for (std::size_t c = 0; c < m; ++c) {
        std::vector<double> e(m, 0.0);
        e[c] = 1;
        const auto col = f.solve(e);
        for (std::size_t r = 0; r < m; ++r)
            inv(r, c) = col[r];
    }
    return inv;
}

/// @brief Quadratic form v' W v.
/// @param v Vector.
/// @param w Matrix.
/// @return The form.
double quadratic(const std::vector<double>& v, const Matrix& w) {
    double s = 0;
    for (std::size_t a = 0; a < v.size(); ++a)
        for (std::size_t b = 0; b < v.size(); ++b)
            s += v[a] * w(a, b) * v[b];
    return s;
}

/// @brief Jacobian of gbar, analytic when supplied and by central differences otherwise.
/// @param moments Moment function.
/// @param jacobian Optional analytic Jacobian.
/// @param theta Parameters.
/// @param q Moment count.
/// @param n Observations.
/// @return q-by-k Jacobian.
Matrix mean_jacobian(const MomentFunction& moments, const MomentJacobian& jacobian, const std::vector<double>& theta,
                     std::size_t q, std::size_t n) {
    const std::size_t k = theta.size();
    if (jacobian) {
        Matrix g = jacobian(theta);
        if (g.rows() != q || g.cols() != k)
            throw std::invalid_argument("generalized_method_of_moments: the Jacobian must be q-by-k");
        return g;
    }
    Matrix g(q, k, 0.0);
    for (std::size_t j = 0; j < k; ++j) {
        const double h = 1e-6 * std::max(1.0, std::abs(theta[j]));
        auto up = theta, down = theta;
        up[j] += h;
        down[j] -= h;
        const auto mu = column_means(evaluate(moments, up, q, n));
        const auto md = column_means(evaluate(moments, down, q, n));
        for (std::size_t a = 0; a < q; ++a)
            g(a, j) = (mu[a] - md[a]) / (up[j] - down[j]);
    }
    return g;
}

/// @brief Product G' W G (k-by-k).
/// @param g q-by-k Jacobian.
/// @param w q-by-q weights.
/// @return The product.
Matrix gwg(const Matrix& g, const Matrix& w) {
    const std::size_t q = g.rows(), k = g.cols();
    Matrix out(k, k, 0.0);
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b)
            for (std::size_t r = 0; r < q; ++r)
                for (std::size_t s = 0; s < q; ++s)
                    out(a, b) += g(r, a) * w(r, s) * g(s, b);
    return out;
}

/// @brief Minimises gbar' W gbar by damped Gauss-Newton from theta (updated in place).
/// @param moments Moment function.
/// @param options Options (Jacobian).
/// @param w Weights.
/// @param theta Start, overwritten by the minimiser.
/// @param q Moment count.
/// @param n Observations.
/// @return True when the iteration converged.
bool minimise(const MomentFunction& moments, const GmmOptions& options, const Matrix& w, std::vector<double>& theta,
              std::size_t q, std::size_t n) {
    const std::size_t k = theta.size();
    auto gbar = column_means(evaluate(moments, theta, q, n));
    double value = quadratic(gbar, w);
    for (int iter = 0; iter < 500; ++iter) {
        const Matrix g = mean_jacobian(moments, options.jacobian, theta, q, n);
        Matrix h = gwg(g, w);
        std::vector<double> grad(k, 0.0);
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t r = 0; r < q; ++r)
                for (std::size_t s = 0; s < q; ++s)
                    grad[a] += g(r, a) * w(r, s) * gbar[s];
        const auto f = linalg::cholesky(h);
        if (!f.ok)
            throw std::runtime_error("generalized_method_of_moments: the moments do not identify theta (singular G'WG)");
        auto step = f.solve(grad);
        double t = 1;
        bool improved = false;
        std::vector<double> trial(k), trial_gbar;
        double trial_value = value;
        for (int half = 0; half < 60; ++half, t /= 2) {
            for (std::size_t a = 0; a < k; ++a)
                trial[a] = theta[a] - t * step[a];
            try {
                trial_gbar = column_means(evaluate(moments, trial, q, n));
            } catch (const std::runtime_error&) {
                continue; // outside the moment function's domain: shorten the step
            }
            trial_value = quadratic(trial_gbar, w);
            if (trial_value <= value) {
                improved = true;
                break;
            }
        }
        if (!improved)
            return true; // no decrease possible along the Gauss-Newton direction: at the minimum
        double change = 0, scale = 0;
        for (std::size_t a = 0; a < k; ++a) {
            change = std::max(change, std::abs(trial[a] - theta[a]));
            scale = std::max(scale, std::abs(trial[a]));
        }
        const double drop = value - trial_value;
        theta = trial;
        gbar = trial_gbar;
        value = trial_value;
        if (change <= 1e-13 * (1 + scale) || drop <= 1e-16 * (1 + value))
            return true;
    }
    return false;
}
} // namespace

GmmResult generalized_method_of_moments(const MomentFunction& moments, const std::vector<double>& start,
                                        const GmmOptions& options) {
    const std::size_t k = start.size();
    if (k == 0 || !std::ranges::all_of(start, [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("generalized_method_of_moments: need a finite start");
    if (!(options.confidence > 0 && options.confidence < 1) || !(options.tolerance > 0))
        throw std::invalid_argument("generalized_method_of_moments: bad confidence or tolerance");
    if (!moments)
        throw std::invalid_argument("generalized_method_of_moments: no moment function");
    const Matrix g0 = evaluate(moments, start, 0, 0);
    const std::size_t n = g0.rows(), q = g0.cols();
    if (q < k)
        throw std::invalid_argument("generalized_method_of_moments: fewer moments than parameters");
    if (n <= q)
        throw std::invalid_argument("generalized_method_of_moments: need more observations than moments");
    Matrix w(q, q, 0.0);
    if (options.initial_weights.rows() == 0 && options.initial_weights.cols() == 0) {
        for (std::size_t a = 0; a < q; ++a)
            w(a, a) = 1;
    } else {
        if (options.initial_weights.rows() != q || options.initial_weights.cols() != q)
            throw std::invalid_argument("generalized_method_of_moments: initial weights must be q-by-q");
        w = options.initial_weights;
        if (!linalg::cholesky(w).ok)
            throw std::invalid_argument("generalized_method_of_moments: initial weights must be positive definite");
    }
    GmmResult out;
    out.n = n;
    out.df = q - k;
    std::vector<double> theta = start;
    bool converged = minimise(moments, options, w, theta, q, n);
    out.steps = 1;
    const bool reweight = q > k && options.estimator != GmmEstimator::one_step;
    if (reweight) {
        const std::size_t rounds = options.estimator == GmmEstimator::two_step ? 1 : options.max_iterations;
        bool settled = options.estimator == GmmEstimator::two_step;
        for (std::size_t r = 0; r < rounds; ++r) {
            w = spd_inverse(moment_covariance(evaluate(moments, theta, q, n), options.centered), "moment covariance");
            const auto previous = theta;
            converged = minimise(moments, options, w, theta, q, n) && converged;
            ++out.steps;
            double change = 0, scale = 0;
            for (std::size_t a = 0; a < k; ++a) {
                change = std::max(change, std::abs(theta[a] - previous[a]));
                scale = std::max(scale, std::abs(previous[a]));
            }
            if (options.estimator == GmmEstimator::iterated && change <= options.tolerance * (1 + scale)) {
                settled = true;
                break;
            }
        }
        converged = converged && settled;
    }
    out.converged = converged;
    out.coefficients = theta;
    out.weights = w;
    const Matrix g = evaluate(moments, theta, q, n);
    out.mean_moments = column_means(g);
    out.objective = quadratic(out.mean_moments, w);
    const Matrix s = moment_covariance(g, options.centered);
    const Matrix jac = mean_jacobian(moments, options.jacobian, theta, q, n);
    const auto dn = static_cast<double>(n);
    if (reweight || q == k) {
        // Efficient form; equals G^{-1} S G^{-T} when exactly identified.
        out.covariance = spd_inverse(gwg(jac, spd_inverse(s, "moment covariance")), "information matrix");
    } else {
        // Sandwich (G'WG)^{-1} G'W S W G (G'WG)^{-1}.
        const Matrix bread = spd_inverse(gwg(jac, w), "G'WG");
        Matrix wg(q, k, 0.0); // W G
        for (std::size_t r = 0; r < q; ++r)
            for (std::size_t a = 0; a < k; ++a)
                for (std::size_t c = 0; c < q; ++c)
                    wg(r, a) += w(r, c) * jac(c, a);
        const Matrix meat = gwg(wg, s);
        Matrix tmp(k, k, 0.0);
        out.covariance = Matrix(k, k, 0.0);
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t b = 0; b < k; ++b)
                for (std::size_t c = 0; c < k; ++c)
                    tmp(a, b) += bread(a, c) * meat(c, b);
        for (std::size_t a = 0; a < k; ++a)
            for (std::size_t b = 0; b < k; ++b)
                for (std::size_t c = 0; c < k; ++c)
                    out.covariance(a, b) += tmp(a, c) * bread(c, b);
    }
    for (double& v : out.covariance.data())
        v /= dn;
    if (q > k) {
        out.j_statistic = reweight ? dn * out.objective
                                   : dn * quadratic(out.mean_moments, spd_inverse(s, "moment covariance"));
        out.j_p_value = 1 - random::chi_squared_cdf(out.j_statistic, static_cast<double>(q - k));
    } else {
        out.j_statistic = 0;
        out.j_p_value = 1;
    }
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    for (std::size_t a = 0; a < k; ++a) {
        const double se = std::sqrt(out.covariance(a, a));
        out.standard_errors.push_back(se);
        out.z_values.push_back(theta[a] / se);
        out.p_values.push_back(2 * random::normal_cdf(-std::abs(theta[a] / se)));
        out.lower.push_back(theta[a] - z * se);
        out.upper.push_back(theta[a] + z * se);
    }
    return out;
}

} // namespace datamunge::stats
