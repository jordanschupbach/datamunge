/// @file rbf_gaussian_process_surrogate.cpp
/// @brief Cholesky-based Gaussian-process regression, profile marginal-likelihood fitting of
///        the length scale (and optionally the nugget), and the acquisition functions.

#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace datamunge::optim {

/// @brief Normal density and CDF, dense Cholesky helpers and the profile likelihood.
namespace {

/// @brief Standard normal CDF.
/// @param u  Argument.
/// @return \f$\Phi(u)\f$.
double phi_cdf(double u) { return 0.5 * std::erfc(-u / std::numbers::sqrt2); }

/// @brief Standard normal density.
/// @param u  Argument.
/// @return \f$\varphi(u)\f$.
double phi_pdf(double u) { return std::exp(-0.5 * u * u) / std::sqrt(2.0 * std::numbers::pi); }

/// @brief Squared Euclidean distance.
/// @param a  First point.
/// @param b  Second point (same length).
/// @return \f$\lVert a-b\rVert^2\f$.
double squared_distance(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (std::size_t k = 0; k < a.size(); ++k) s += (a[k] - b[k]) * (a[k] - b[k]);
    return s;
}

/// @brief In-place Cholesky factor of a row-major symmetric matrix (lower triangle used).
/// @param a  Matrix; on return its lower triangle holds \f$L\f$.
/// @param n  Order.
/// @return False if the matrix is not numerically positive definite.
bool cholesky_in_place(std::vector<double>& a, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double s = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) s -= a[i * n + k] * a[j * n + k];
            if (i == j) {
                if (!(s > 0.0)) return false;
                a[i * n + i] = std::sqrt(s);
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
    return true;
}

/// @brief Solves \f$LL^\top x=b\f$ given the lower factor.
/// @param l  Row-major lower factor.
/// @param n  Order.
/// @param b  Right-hand side, overwritten by the solution.
void cholesky_solve(const std::vector<double>& l, std::size_t n, std::vector<double>& b) {
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k) b[i] -= l[i * n + k] * b[k];
        b[i] /= l[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
        for (std::size_t k = i + 1; k < n; ++k) b[i] -= l[k * n + i] * b[k];
        b[i] /= l[i * n + i];
    }
}

/// @brief Profile fit for one (length scale, nugget ratio) pair.
struct ProfileFit {
    double log_likelihood{-std::numeric_limits<double>::infinity()};  ///< Profile log marginal likelihood.
    double signal_variance{1.0};                                      ///< \f$\hat s=y^\top(R+gI)^{-1}y/n\f$.
};

/// @brief Profile log likelihood of normalised values @p y for correlation length @p ell and
///        nugget ratio @p g, with the signal variance maximised in closed form.
/// @param sq    Pairwise squared distances, row-major.
/// @param y     Normalised values.
/// @param ell   Length scale.
/// @param g     Nugget ratio noise / signal variance.
/// @return The profile fit (\f$-\infty\f$ if the matrix is not positive definite).
ProfileFit profile_fit(const std::vector<double>& sq, const std::vector<double>& y, double ell, double g) {
    const std::size_t   n = y.size();
    std::vector<double> a(n * n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            a[i * n + j] = std::exp(-sq[i * n + j] / (2.0 * ell * ell)) + (i == j ? g : 0.0);
    if (!cholesky_in_place(a, n)) return {};
    std::vector<double> z = y;
    double              logdet = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = 0; k < i; ++k) z[i] -= a[i * n + k] * z[k];
        z[i] /= a[i * n + i];
        logdet += 2.0 * std::log(a[i * n + i]);
    }
    double q = 0.0;
    for (const double v : z) q += v * v;
    const double s  = std::max(q / static_cast<double>(n), 1e-300);
    const double nn = static_cast<double>(n);
    return {-0.5 * nn * std::log(2.0 * std::numbers::pi * s) - 0.5 * nn - 0.5 * logdet, s};
}

} // namespace

double expected_improvement(double mean, double sd, double incumbent, double exploration) {
    const double gap = incumbent - exploration - mean;
    if (!(sd > 0.0)) return std::max(gap, 0.0);
    const double u = gap / sd;
    return std::max(0.0, gap * phi_cdf(u) + sd * phi_pdf(u));
}

double probability_of_improvement(double mean, double sd, double incumbent, double exploration) {
    const double gap = incumbent - exploration - mean;
    if (!(sd > 0.0)) return gap > 0.0 ? 1.0 : 0.0;
    return phi_cdf(gap / sd);
}

RBFGaussianProcessSurrogate::RBFGaussianProcessSurrogate(double length_scale, double noise)
    : RBFGaussianProcessSurrogate(GaussianProcessSurrogateOptions{.length_scale        = length_scale,
                                                                  .signal_variance     = 1.0,
                                                                  .noise               = noise,
                                                                  .normalize_values    = false,
                                                                  .fit_hyperparameters = false}) {}

RBFGaussianProcessSurrogate::RBFGaussianProcessSurrogate(const GaussianProcessSurrogateOptions& options)
    : options_(options), length_scale_(options.length_scale), noise_(options.noise),
      signal_variance_(options.signal_variance) {
    if (!(options.length_scale > 0.0) || !(options.noise >= 0.0) || !(options.signal_variance > 0.0))
        throw std::invalid_argument("invalid GP hyperparameters");
    if (options.min_length_scale < 0.0 || options.max_length_scale < 0.0 ||
        (options.max_length_scale > 0.0 && options.min_length_scale > options.max_length_scale))
        throw std::invalid_argument("invalid GP length-scale bounds");
    if (options.exploration < 0.0 || options.kappa < 0.0) throw std::invalid_argument("invalid GP acquisition settings");
}

void RBFGaussianProcessSurrogate::fit(const std::vector<std::vector<double>>& p, const std::vector<double>& v) {
    if (p.empty() || p.size() != v.size()) throw std::invalid_argument("invalid GP training data");
    for (const auto& point : p)
        if (point.size() != p.front().size()) throw std::invalid_argument("inconsistent GP dimensions");
    points_             = p;
    values_             = v;
    const std::size_t n = v.size();

    offset_ = 0.0, scale_ = 1.0;
    if (options_.normalize_values) {
        for (const double y : v) offset_ += y / static_cast<double>(n);
        double ss = 0.0;
        for (const double y : v) ss += (y - offset_) * (y - offset_);
        const double sd = std::sqrt(ss / static_cast<double>(n));
        scale_          = sd > 0.0 ? sd : 1.0;
    }
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) y[i] = (v[i] - offset_) / scale_;

    std::vector<double> sq(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < i; ++j) sq[i * n + j] = sq[j * n + i] = squared_distance(points_[i], points_[j]);

    length_scale_ = options_.length_scale, signal_variance_ = options_.signal_variance, noise_ = options_.noise;
    if (options_.fit_hyperparameters && n >= 2) {
        double range = 0.0;
        for (std::size_t k = 0; k < p.front().size(); ++k) {
            double lo = p.front()[k], hi = lo;
            for (const auto& q : p) lo = std::min(lo, q[k]), hi = std::max(hi, q[k]);
            range = std::max(range, hi - lo);
        }
        if (!(range > 0.0)) range = 1.0;
        const double lmin = options_.min_length_scale > 0.0 ? options_.min_length_scale : 0.01 * range;
        const double lmax = options_.max_length_scale > 0.0 ? options_.max_length_scale : 10.0 * range;
        std::vector<double> nuggets{options_.noise / options_.signal_variance};
        if (options_.fit_noise) {
            nuggets.clear();
            for (int e = -8; e <= 0; ++e) nuggets.push_back(std::pow(10.0, e));
        }
        // Coarse log grid over (length scale, nugget), then golden-section refinement in log length scale.
        const int  grid = 40;
        double     best_ll = -std::numeric_limits<double>::infinity(), best_log_ell = std::log(lmin), best_g = nuggets[0];
        const double a = std::log(lmin), b = std::log(lmax), h = (b - a) / (grid - 1);
        for (const double g : nuggets)
            for (int i = 0; i < grid; ++i) {
                const double le = a + h * i;
                const auto   f  = profile_fit(sq, y, std::exp(le), g);
                if (f.log_likelihood > best_ll) best_ll = f.log_likelihood, best_log_ell = le, best_g = g;
            }
        double lo = std::max(a, best_log_ell - h), hi = std::min(b, best_log_ell + h);
        const double r = (std::sqrt(5.0) - 1.0) / 2.0;
        double x1 = hi - r * (hi - lo), x2 = lo + r * (hi - lo);
        double f1 = profile_fit(sq, y, std::exp(x1), best_g).log_likelihood;
        double f2 = profile_fit(sq, y, std::exp(x2), best_g).log_likelihood;
        for (int it = 0; it < 40; ++it) {
            if (f1 > f2) hi = x2, x2 = x1, f2 = f1, x1 = hi - r * (hi - lo), f1 = profile_fit(sq, y, std::exp(x1), best_g).log_likelihood;
            else lo = x1, x1 = x2, f1 = f2, x2 = lo + r * (hi - lo), f2 = profile_fit(sq, y, std::exp(x2), best_g).log_likelihood;
        }
        const double le = f1 > f2 ? x1 : x2;
        if (std::max(f1, f2) > best_ll) best_log_ell = le;
        const auto fit    = profile_fit(sq, y, std::exp(best_log_ell), best_g);
        length_scale_     = std::exp(best_log_ell);
        signal_variance_  = fit.signal_variance;
        noise_            = best_g * fit.signal_variance;
    }

    // Final factorisation at the hyperparameters in use.
    cholesky_.assign(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            cholesky_[i * n + j] =
                signal_variance_ * std::exp(-sq[i * n + j] / (2.0 * length_scale_ * length_scale_)) + (i == j ? noise_ : 0.0);
    if (!cholesky_in_place(cholesky_, n))
        throw std::runtime_error("RBFGaussianProcessSurrogate: kernel matrix not positive definite (add noise)");
    alpha_ = y;
    cholesky_solve(cholesky_, n, alpha_);
    double quad = 0.0, logdet = 0.0;
    for (std::size_t i = 0; i < n; ++i) quad += y[i] * alpha_[i], logdet += 2.0 * std::log(cholesky_[i * n + i]);
    const double nn          = static_cast<double>(n);
    log_marginal_likelihood_ = -0.5 * quad - 0.5 * logdet - 0.5 * nn * std::log(2.0 * std::numbers::pi) - nn * std::log(scale_);
}

void RBFGaussianProcessSurrogate::latent(const std::vector<double>& x, double& mean, double& var) const {
    if (x.size() != points_.front().size()) throw std::invalid_argument("GP query dimension mismatch");
    const std::size_t   n = values_.size();
    std::vector<double> z(n);
    mean = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double k = signal_variance_ * std::exp(-squared_distance(x, points_[i]) / (2.0 * length_scale_ * length_scale_));
        mean += k * alpha_[i];
        z[i] = k;
        for (std::size_t q = 0; q < i; ++q) z[i] -= cholesky_[i * n + q] * z[q];
        z[i] /= cholesky_[i * n + i];
    }
    var = signal_variance_;
    for (const double q : z) var -= q * q;
}

SurrogatePrediction RBFGaussianProcessSurrogate::predict(const std::vector<double>& x) const {
    if (values_.empty()) throw std::logic_error("RBFGaussianProcessSurrogate::predict: call fit() first");
    double m = 0.0, var = 0.0;
    latent(x, m, var);
    return {offset_ + scale_ * m, scale_ * std::sqrt(std::max(0.0, var))};
}

double RBFGaussianProcessSurrogate::acquisition(const std::vector<double>& x, double incumbent) {
    if (values_.empty()) return 0.0;
    double m = 0.0, var = 0.0;
    latent(x, m, var);
    // The variance floor keeps the score finite at observed points.
    const double mean = offset_ + scale_ * m, sd = scale_ * std::sqrt(std::max(1e-12 * signal_variance_, var));
    switch (options_.acquisition) {
    case Acquisition::ProbabilityOfImprovement: return probability_of_improvement(mean, sd, incumbent, options_.exploration);
    case Acquisition::LowerConfidenceBound: return -(mean - options_.kappa * sd);
    case Acquisition::ExpectedImprovement: break;
    }
    return expected_improvement(mean, sd, incumbent, options_.exploration);
}

} // namespace datamunge::optim
