/// @file meta_analysis.cpp
/// @brief Inverse-variance pooling and heterogeneity statistics.
#include <datamunge/stats/meta_analysis.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::stats {

/// @brief Implementation helpers for meta-analysis.
namespace {
/// @brief Validates study effects and variances.
/// @param y Effects.
/// @param v Variances.
/// @param confidence Interval coverage.
/// @param what Function name for messages.
void check_studies(const std::vector<double>& y, const std::vector<double>& v, double confidence, const char* what) {
    if (y.empty() || y.size() != v.size())
        throw std::invalid_argument(std::string(what) + ": need equal-length, nonempty effects and variances");
    for (std::size_t i = 0; i < y.size(); ++i)
        if (!std::isfinite(y[i]) || !(v[i] > 0) || !std::isfinite(v[i]))
            throw std::invalid_argument(std::string(what) + ": effects must be finite and variances positive");
    if (!(confidence > 0 && confidence < 1))
        throw std::invalid_argument(std::string(what) + ": confidence must be in (0, 1)");
}

/// @brief Cochran's Q and the derived heterogeneity statistics, stored into a result.
/// @param y Effects.
/// @param v Variances.
/// @param out Result receiving q, q_df, q_p_value, i2 and h2.
void heterogeneity(const std::vector<double>& y, const std::vector<double>& v, MetaAnalysis& out) {
    double sw = 0, swy = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        sw += 1 / v[i], swy += y[i] / v[i];
    const double ce = swy / sw;
    double q = 0;
    for (std::size_t i = 0; i < y.size(); ++i)
        q += (y[i] - ce) * (y[i] - ce) / v[i];
    const double df = static_cast<double>(y.size()) - 1;
    out.q = q;
    out.q_df = df;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.q_p_value = df > 0 ? random::chi_squared_survival(q, df) : nan;
    out.i2 = df > 0 && q > 0 ? 100 * std::max(0.0, (q - df) / q) : (df > 0 ? 0.0 : nan);
    out.h2 = df > 0 ? q / df : nan;
}
} // namespace

MetaAnalysis common_effect_meta(const std::vector<double>& effects, const std::vector<double>& variances,
                                double confidence) {
    check_studies(effects, variances, confidence, "common_effect_meta");
    MetaAnalysis m;
    m.k = effects.size();
    double sw = 0, swy = 0;
    for (std::size_t i = 0; i < m.k; ++i)
        sw += 1 / variances[i], swy += effects[i] / variances[i];
    m.estimate = swy / sw;
    m.standard_error = 1 / std::sqrt(sw);
    m.statistic = m.estimate / m.standard_error;
    m.p_value = 2 * (1 - random::normal_cdf(std::abs(m.statistic)));
    const double z = random::normal_quantile(0.5 + confidence / 2);
    m.lower = m.estimate - z * m.standard_error;
    m.upper = m.estimate + z * m.standard_error;
    m.weights.resize(m.k);
    for (std::size_t i = 0; i < m.k; ++i)
        m.weights[i] = 1 / variances[i] / sw;
    heterogeneity(effects, variances, m);
    m.tau2 = m.tau = 0;
    m.tau2_lower = m.tau2_upper = std::numeric_limits<double>::quiet_NaN();
    m.prediction_lower = m.lower;
    m.prediction_upper = m.upper;
    return m;
}

/// @brief Implementation helpers for random-effects models with a design matrix.
namespace {
/// @brief Dense matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Weighted least-squares quantities of a mixed-effects meta-analytic model at a given tau2.
struct WeightedFit {
    std::vector<double> beta; ///< Fixed-effect coefficients.
    Matrix covariance;        ///< (X'WX)^{-1}.
    Matrix p;                 ///< P = W - W X (X'WX)^{-1} X' W.
    std::vector<double> py;   ///< P y.
    double ypy{};             ///< y' P y (generalised Q).
    double ypppy{};           ///< y' P P y.
    double trace_p{};         ///< tr(P).
    double trace_pp{};        ///< tr(P P).
    double trace_w{};         ///< tr(W).
    double trace_ww{};        ///< tr(W W).
    double log_det_xwx{};     ///< log det(X'WX) (for the restricted likelihood).
    double sum_log_var{};     ///< sum log(v_i + tau2).
};

/// @brief Computes the weighted fit at tau2.
/// @param y Effects.
/// @param v Sampling variances.
/// @param x Design matrix (k x p).
/// @param tau2 Between-study variance.
/// @return The fit quantities.
/// @throws std::invalid_argument If X'WX is singular.
WeightedFit weighted_fit(const std::vector<double>& y, const std::vector<double>& v, const Matrix& x, double tau2) {
    const std::size_t k = y.size(), q = x.cols();
    std::vector<double> w(k);
    WeightedFit f;
    for (std::size_t i = 0; i < k; ++i) {
        w[i] = 1 / (v[i] + tau2);
        f.trace_w += w[i], f.trace_ww += w[i] * w[i], f.sum_log_var += std::log(v[i] + tau2);
    }
    Matrix a(q, q, 0.0);
    std::vector<double> b(q, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t r = 0; r < q; ++r) {
            b[r] += x(i, r) * w[i] * y[i];
            for (std::size_t c = 0; c < q; ++c)
                a(r, c) += x(i, r) * w[i] * x(i, c);
        }
    const auto ch = linalg::cholesky(a);
    if (!ch.ok)
        throw std::invalid_argument("meta-analysis: the weighted design matrix is singular");
    f.beta = ch.solve(b);
    f.covariance = Matrix(q, q, 0.0);
    for (std::size_t c = 0; c < q; ++c) {
        std::vector<double> e(q, 0.0);
        e[c] = 1;
        const auto col = ch.solve(e);
        for (std::size_t r = 0; r < q; ++r)
            f.covariance(r, c) = col[r];
    }
    for (std::size_t r = 0; r < q; ++r)
        f.log_det_xwx += 2 * std::log(ch.L(r, r));
    // P = W - W X C X' W.
    Matrix wx(k, q, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t r = 0; r < q; ++r)
            wx(i, r) = w[i] * x(i, r);
    Matrix wxc(k, q, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t c = 0; c < q; ++c)
            for (std::size_t r = 0; r < q; ++r)
                wxc(i, c) += wx(i, r) * f.covariance(r, c);
    f.p = Matrix(k, k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0;
            for (std::size_t c = 0; c < q; ++c)
                s += wxc(i, c) * wx(j, c);
            f.p(i, j) = (i == j ? w[i] : 0.0) - s;
        }
    f.py.assign(k, 0.0);
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j)
            f.py[i] += f.p(i, j) * y[j];
    for (std::size_t i = 0; i < k; ++i) {
        f.ypy += y[i] * f.py[i];
        f.ypppy += f.py[i] * f.py[i];
        f.trace_p += f.p(i, i);
        for (std::size_t j = 0; j < k; ++j)
            f.trace_pp += f.p(i, j) * f.p(j, i);
    }
    return f;
}

/// @brief Solves Q(tau2) = target for tau2 >= 0, where the generalised Q decreases in tau2.
/// @param y Effects.
/// @param v Variances.
/// @param x Design matrix.
/// @param target Target value of y'P(tau2)y.
/// @param tolerance Relative tolerance on tau2.
/// @param max_iterations Iteration cap.
/// @param iterations Receives the number of bisection steps.
/// @return The root (0 if Q(0) <= target).
double solve_generalised_q(const std::vector<double>& y, const std::vector<double>& v, const Matrix& x, double target,
                           double tolerance, std::size_t max_iterations, std::size_t& iterations) {
    iterations = 0;
    if (weighted_fit(y, v, x, 0.0).ypy <= target)
        return 0.0;
    double lo = 0, hi = 1;
    for (const double vi : v)
        hi = std::max(hi, 10 * vi);
    while (weighted_fit(y, v, x, hi).ypy > target)
        hi *= 4;
    for (; iterations < max_iterations && hi - lo > tolerance * std::max(1e-8, hi); ++iterations) {
        const double mid = (lo + hi) / 2;
        (weighted_fit(y, v, x, mid).ypy > target ? lo : hi) = mid;
    }
    return (lo + hi) / 2;
}

/// @brief Restricted (or ordinary) log-likelihood of the model at tau2, up to a constant.
/// @param f Fit at tau2.
/// @param restricted Whether to include the REML term.
/// @return The log-likelihood.
double model_log_likelihood(const WeightedFit& f, bool restricted) {
    return -0.5 * (f.sum_log_var + f.ypy + (restricted ? f.log_det_xwx : 0.0));
}

/// @brief Estimated tau2 and iteration diagnostics.
struct TauFit {
    double tau2{};            ///< The estimate.
    std::size_t iterations{}; ///< Iterations used.
    bool converged{true};     ///< Whether the iteration converged.
};

/// @brief Estimates tau2 for the model y = X beta + u + e.
/// @param y Effects.
/// @param v Variances.
/// @param x Design matrix.
/// @param options Estimator and iteration control.
/// @return tau2 with diagnostics.
TauFit estimate_tau2(const std::vector<double>& y, const std::vector<double>& v, const Matrix& x,
                     const RandomEffectsOptions& options) {
    const double df = static_cast<double>(y.size()) - static_cast<double>(x.cols());
    const WeightedFit f0 = weighted_fit(y, v, x, 0.0);
    TauFit out;
    const double dl = std::max(0.0, (f0.ypy - df) / f0.trace_p);
    switch (options.estimator) {
    case TauEstimator::dersimonian_laird:
        out.tau2 = dl;
        return out;
    case TauEstimator::paule_mandel:
        out.tau2 = solve_generalised_q(y, v, x, df, options.tolerance, options.max_iterations, out.iterations);
        out.converged = out.iterations < options.max_iterations;
        return out;
    case TauEstimator::reml:
    case TauEstimator::ml: {
        const bool reml = options.estimator == TauEstimator::reml;
        double tau2 = dl;
        out.converged = false;
        for (out.iterations = 1; out.iterations <= options.max_iterations; ++out.iterations) {
            const WeightedFit f = weighted_fit(y, v, x, tau2);
            const double step = reml ? (f.ypppy - f.trace_p) / f.trace_pp : (f.ypppy - f.trace_w) / f.trace_ww;
            double next = std::max(0.0, tau2 + step);
            // Step halving keeps the (restricted) likelihood from decreasing.
            const double base = model_log_likelihood(f, reml);
            for (int h = 0; h < 50 && model_log_likelihood(weighted_fit(y, v, x, next), reml) < base - 1e-12; ++h)
                next = std::max(0.0, tau2 + (next - tau2) / 2);
            const double change = std::abs(next - tau2);
            tau2 = next;
            if (change <= options.tolerance * std::max(1.0, tau2)) {
                out.converged = true;
                break;
            }
        }
        out.iterations = std::min(out.iterations, options.max_iterations);
        out.tau2 = tau2;
        return out;
    }
    }
    return out;
}
} // namespace

MetaAnalysis random_effects_meta(const std::vector<double>& effects, const std::vector<double>& variances,
                                 const RandomEffectsOptions& options) {
    check_studies(effects, variances, options.confidence, "random_effects_meta");
    const std::size_t k = effects.size();
    if (k < 2 || (options.t_prediction && k < 3))
        throw std::invalid_argument("random_effects_meta: need at least two studies (three for a t prediction interval)");
    const Matrix x(k, 1, 1.0);
    const TauFit tf = estimate_tau2(effects, variances, x, options);
    MetaAnalysis m;
    m.k = k;
    heterogeneity(effects, variances, m);
    m.tau2 = tf.tau2;
    m.tau = std::sqrt(tf.tau2);
    m.iterations = tf.iterations;
    m.converged = tf.converged;
    const WeightedFit f = weighted_fit(effects, variances, x, tf.tau2);
    m.estimate = f.beta[0];
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    double crit_mean = z;
    if (options.inference != MetaInference::wald) {
        m.df = static_cast<double>(k) - 1;
        m.variance_scale = f.ypy / m.df;
        if (options.inference == MetaInference::hartung_knapp_truncated)
            m.variance_scale = std::max(1.0, m.variance_scale);
        crit_mean = random::student_t_quantile(0.5 + options.confidence / 2, m.df);
    }
    m.standard_error = std::sqrt(m.variance_scale * f.covariance(0, 0));
    m.statistic = m.estimate / m.standard_error;
    m.p_value = std::isnan(m.df) ? 2 * (1 - random::normal_cdf(std::abs(m.statistic)))
                                 : 2 * (1 - random::student_t_cdf(std::abs(m.statistic), m.df));
    m.lower = m.estimate - crit_mean * m.standard_error;
    m.upper = m.estimate + crit_mean * m.standard_error;
    double sw = 0;
    for (std::size_t i = 0; i < k; ++i)
        sw += 1 / (variances[i] + tf.tau2);
    m.weights.resize(k);
    for (std::size_t i = 0; i < k; ++i)
        m.weights[i] = 1 / (variances[i] + tf.tau2) / sw;
    // I^2 and H^2 from tau2 and the typical within-study variance (k - 1) / tr(P_0).
    const WeightedFit f0 = weighted_fit(effects, variances, x, 0.0);
    const double s2 = (static_cast<double>(k) - 1) / f0.trace_p;
    m.i2 = 100 * tf.tau2 / (tf.tau2 + s2);
    m.h2 = (tf.tau2 + s2) / s2;
    // Prediction interval.
    const double crit = options.t_prediction
                            ? random::student_t_quantile(0.5 + options.confidence / 2, static_cast<double>(k) - 2)
                            : crit_mean;
    const double half = crit * std::sqrt(m.standard_error * m.standard_error + tf.tau2);
    m.prediction_lower = m.estimate - half;
    m.prediction_upper = m.estimate + half;
    // Q-profile interval for tau2.
    const double df = static_cast<double>(k) - 1, alpha = 1 - options.confidence;
    std::size_t it = 0;
    m.tau2_lower = solve_generalised_q(effects, variances, x, random::chi_squared_quantile(1 - alpha / 2, df), 1e-12, 2000, it);
    m.tau2_upper = solve_generalised_q(effects, variances, x, random::chi_squared_quantile(alpha / 2, df), 1e-12, 2000, it);
    return m;
}

/// @brief Implementation helpers for meta-regression.
namespace {
/// @brief Core fit of a meta-regression on a full design matrix (no influence diagnostics).
/// @param y Effects.
/// @param v Variances.
/// @param x Design matrix including the intercept.
/// @param options Model options.
/// @param r2_reference Intercept-only tau2 for pseudo-R^2 (NaN to skip).
/// @return The fit.
MetaRegression fit_meta_regression(const std::vector<double>& y, const std::vector<double>& v, const Matrix& x,
                                   const RandomEffectsOptions& options, double r2_reference) {
    const std::size_t k = y.size(), p = x.cols();
    MetaRegression r;
    r.k = k;
    r.p = p;
    const TauFit tf = estimate_tau2(y, v, x, options);
    r.tau2 = tf.tau2;
    r.iterations = tf.iterations;
    r.converged = tf.converged;
    const WeightedFit f = weighted_fit(y, v, x, tf.tau2);
    const WeightedFit f0 = weighted_fit(y, v, x, 0.0);
    const double resid_df = static_cast<double>(k - p);
    r.qe = f0.ypy;
    r.qe_df = resid_df;
    r.qe_p_value = random::chi_squared_survival(r.qe, resid_df);
    const double s2 = resid_df / f0.trace_p;
    r.i2 = 100 * r.tau2 / (r.tau2 + s2);
    r.h2 = (r.tau2 + s2) / s2;
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    double crit = z;
    if (options.inference != MetaInference::wald) {
        r.df = resid_df;
        r.variance_scale = f.ypy / resid_df;
        if (options.inference == MetaInference::hartung_knapp_truncated)
            r.variance_scale = std::max(1.0, r.variance_scale);
        crit = random::student_t_quantile(0.5 + options.confidence / 2, r.df);
    }
    r.covariance = Matrix(p, p, 0.0);
    for (std::size_t a = 0; a < p; ++a)
        for (std::size_t b = 0; b < p; ++b)
            r.covariance(a, b) = r.variance_scale * f.covariance(a, b);
    r.coefficients = f.beta;
    for (std::size_t j = 0; j < p; ++j) {
        const double se = std::sqrt(r.covariance(j, j)), st = f.beta[j] / se;
        r.standard_errors.push_back(se);
        r.statistics.push_back(st);
        r.p_values.push_back(std::isnan(r.df) ? 2 * (1 - random::normal_cdf(std::abs(st)))
                                              : 2 * (1 - random::student_t_cdf(std::abs(st), r.df)));
        r.lower.push_back(f.beta[j] - crit * se);
        r.upper.push_back(f.beta[j] + crit * se);
    }
    // Omnibus test of the moderator coefficients (all but the intercept).
    const std::size_t m = p - 1;
    r.qm_df = static_cast<double>(m);
    if (m > 0) {
        Matrix sub(m, m, 0.0);
        std::vector<double> b(m);
        for (std::size_t a = 0; a < m; ++a) {
            b[a] = f.beta[a + 1];
            for (std::size_t c = 0; c < m; ++c)
                sub(a, c) = r.covariance(a + 1, c + 1);
        }
        const auto ch = linalg::cholesky(sub);
        const auto sol = ch.solve(b);
        double qm = 0;
        for (std::size_t a = 0; a < m; ++a)
            qm += b[a] * sol[a];
        if (std::isnan(r.df)) {
            r.qm = qm;
            r.qm_p_value = random::chi_squared_survival(qm, r.qm_df);
        } else {
            r.qm = qm / r.qm_df;
            r.qm_p_value = 1 - random::f_cdf(r.qm, r.qm_df, r.df);
        }
    }
    double sw = 0;
    for (std::size_t i = 0; i < k; ++i)
        sw += 1 / (v[i] + tf.tau2);
    for (std::size_t i = 0; i < k; ++i) {
        double fit = 0;
        for (std::size_t j = 0; j < p; ++j)
            fit += x(i, j) * f.beta[j];
        r.fitted.push_back(fit);
        r.residuals.push_back(y[i] - fit);
        r.weights.push_back(1 / (v[i] + tf.tau2) / sw);
    }
    const double alpha = 1 - options.confidence;
    std::size_t it = 0;
    r.tau2_lower = solve_generalised_q(y, v, x, random::chi_squared_quantile(1 - alpha / 2, resid_df), 1e-12, 2000, it);
    r.tau2_upper = solve_generalised_q(y, v, x, random::chi_squared_quantile(alpha / 2, resid_df), 1e-12, 2000, it);
    r.r2 = std::isnan(r2_reference) ? std::numeric_limits<double>::quiet_NaN()
           : r2_reference > 0      ? std::clamp(100 * (r2_reference - r.tau2) / r2_reference, 0.0, 100.0)
                                   : 0.0;
    return r;
}

/// @brief Log-determinant of a symmetric positive definite matrix.
/// @param a Matrix.
/// @return log det(a).
double log_det_spd(const Matrix& a) {
    const auto ch = linalg::cholesky(a);
    double s = 0;
    for (std::size_t i = 0; i < a.rows(); ++i)
        s += 2 * std::log(ch.L(i, i));
    return s;
}
/// @brief Applies a p x p matrix to a vector.
/// @param a Matrix.
/// @param v Vector.
/// @return a v.
std::vector<double> apply(const Matrix& a, const std::vector<double>& v) {
    std::vector<double> out(a.rows(), 0.0);
    for (std::size_t i = 0; i < a.rows(); ++i)
        for (std::size_t j = 0; j < a.cols(); ++j)
            out[i] += a(i, j) * v[j];
    return out;
}

/// @brief Congruence transform T C T'.
/// @param c Square matrix.
/// @param tr Transformation.
/// @return T C T'.
Matrix transform(const Matrix& c, const Matrix& tr) {
    const std::size_t p = c.rows();
    Matrix out(p, p, 0.0);
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    out(i, j) += tr(i, a) * c(a, b) * tr(j, b);
    return out;
}

/// @brief Maps coefficients from the standardised to the original moderator scale and recomputes the
/// coefficient table (QM, tau2 and fitted values are invariant).
/// @param r Fit on the standardised scale, updated in place.
/// @param tr Map beta = T beta_std.
/// @param confidence Interval coverage.
void map_coefficients(MetaRegression& r, const Matrix& tr, double confidence) {
    r.coefficients = apply(tr, r.coefficients);
    r.covariance = transform(r.covariance, tr);
    const double crit = std::isnan(r.df) ? random::normal_quantile(0.5 + confidence / 2)
                                         : random::student_t_quantile(0.5 + confidence / 2, r.df);
    for (std::size_t j = 0; j < r.p; ++j) {
        const double se = std::sqrt(r.covariance(j, j)), st = r.coefficients[j] / se;
        r.standard_errors[j] = se;
        r.statistics[j] = st;
        r.p_values[j] = std::isnan(r.df) ? 2 * (1 - random::normal_cdf(std::abs(st)))
                                         : 2 * (1 - random::student_t_cdf(std::abs(st), r.df));
        r.lower[j] = r.coefficients[j] - crit * se;
        r.upper[j] = r.coefficients[j] + crit * se;
    }
}
} // namespace

MetaRegression meta_regression(const std::vector<double>& effects, const std::vector<double>& variances,
                               const linalg::DenseMatrix<double>& moderators, const MetaRegressionOptions& options) {
    check_studies(effects, variances, options.model.confidence, "meta_regression");
    const std::size_t k = effects.size(), m = moderators.cols(), p = m + 1;
    if (moderators.rows() != k || m == 0 || k <= p)
        throw std::invalid_argument("meta_regression: need a k x m moderator matrix with m >= 1 and k > m + 1");
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < m; ++j)
            if (!std::isfinite(moderators(i, j)))
                throw std::invalid_argument("meta_regression: moderators must be finite");
    // Fit on centred and scaled moderators for numerical stability; beta = T beta_std.
    std::vector<double> centre(m, 0.0), scale(m, 0.0);
    for (std::size_t j = 0; j < m; ++j) {
        for (std::size_t i = 0; i < k; ++i)
            centre[j] += moderators(i, j) / static_cast<double>(k);
        for (std::size_t i = 0; i < k; ++i)
            scale[j] += (moderators(i, j) - centre[j]) * (moderators(i, j) - centre[j]);
        scale[j] = std::sqrt(scale[j] / static_cast<double>(k));
        if (!(scale[j] > 0))
            throw std::invalid_argument("meta_regression: a moderator is constant");
    }
    Matrix tr(p, p, 0.0);
    tr(0, 0) = 1;
    for (std::size_t j = 0; j < m; ++j) {
        tr(j + 1, j + 1) = 1 / scale[j];
        tr(0, j + 1) = -centre[j] / scale[j];
    }
    const auto design = [&](const std::vector<std::size_t>& rows) {
        Matrix x(rows.size(), p, 1.0);
        for (std::size_t a = 0; a < rows.size(); ++a)
            for (std::size_t j = 0; j < m; ++j)
                x(a, j + 1) = (moderators(rows[a], j) - centre[j]) / scale[j];
        return x;
    };
    std::vector<std::size_t> all(k);
    for (std::size_t i = 0; i < k; ++i)
        all[i] = i;
    const Matrix x = design(all);
    const double tau2_null = estimate_tau2(effects, variances, Matrix(k, 1, 1.0), options.model).tau2;
    MetaRegression r = fit_meta_regression(effects, variances, x, options.model, tau2_null);
    const Matrix cov_std = r.covariance;
    const std::vector<double> beta_std = r.coefficients;
    map_coefficients(r, tr, options.model.confidence);
    if (!options.influence)
        return r;
    if (k <= p + 1)
        throw std::invalid_argument("meta_regression: influence diagnostics need k > p + 1");
    // Hat values and internally standardised residuals.
    MetaInfluence& inf = r.influence;
    const WeightedFit f = weighted_fit(effects, variances, x, r.tau2);
    Matrix h(k, k, 0.0); // H = X C X' W (invariant to the reparameterisation)
    for (std::size_t i = 0; i < k; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            double s = 0;
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    s += x(i, a) * f.covariance(a, b) * x(j, b);
            h(i, j) = s / (variances[j] + r.tau2);
        }
    for (std::size_t i = 0; i < k; ++i) {
        inf.hat.push_back(h(i, i));
        inf.weight.push_back(100 * r.weights[i]);
        // Var(e) = (I - H) M (I - H)' with M = diag(v + tau2).
        double var = 0;
        for (std::size_t j = 0; j < k; ++j) {
            const double a = (i == j ? 1.0 : 0.0) - h(i, j);
            var += a * a * (variances[j] + r.tau2);
        }
        inf.standardized_residuals.push_back(r.residuals[i] / std::sqrt(var));
    }
    const double logdet_full = log_det_spd(cov_std);
    const auto pd = static_cast<double>(p), kd = static_cast<double>(k);
    for (std::size_t i = 0; i < k; ++i) {
        std::vector<std::size_t> rows;
        std::vector<double> yy, vv;
        for (std::size_t j = 0; j < k; ++j)
            if (j != i)
                rows.push_back(j), yy.push_back(effects[j]), vv.push_back(variances[j]);
        const Matrix xd = design(rows);
        const MetaRegression d = fit_meta_regression(yy, vv, xd, options.model, std::numeric_limits<double>::quiet_NaN());
        inf.tau2_deleted.push_back(d.tau2);
        inf.qe_deleted.push_back(d.qe);
        // Deleted prediction for study i.
        double pred = 0, pvar = 0;
        for (std::size_t a = 0; a < p; ++a) {
            pred += x(i, a) * d.coefficients[a];
            for (std::size_t b = 0; b < p; ++b)
                pvar += x(i, a) * d.covariance(a, b) * x(i, b);
        }
        inf.studentized_residuals.push_back((effects[i] - pred) / std::sqrt(variances[i] + d.tau2 + pvar));
        inf.dffits.push_back((r.fitted[i] - pred) / std::sqrt(inf.hat[i] * (d.tau2 + variances[i])));
        std::vector<double> diff_std(p);
        for (std::size_t a = 0; a < p; ++a)
            diff_std[a] = beta_std[a] - d.coefficients[a];
        const auto sol = linalg::cholesky(cov_std).solve(diff_std);
        double cook = 0;
        for (std::size_t a = 0; a < p; ++a)
            cook += diff_std[a] * sol[a];
        inf.cooks_distance.push_back(cook);
        inf.covariance_ratio.push_back(std::exp(log_det_spd(d.covariance) - logdet_full));
        // DFBETAS as metafor: the change divided by SE from all k studies weighted with tau2_(-i).
        const WeightedFit fd = weighted_fit(effects, variances, x, d.tau2);
        const Matrix vdel = transform(fd.covariance, tr);
        const std::vector<double> diff = apply(tr, diff_std);
        std::vector<double> dfb(p);
        bool big = false;
        for (std::size_t a = 0; a < p; ++a) {
            dfb[a] = diff[a] / std::sqrt(d.variance_scale * vdel(a, a));
            big = big || std::abs(dfb[a]) > 1;
        }
        inf.dfbetas.push_back(dfb);
        inf.influential.push_back(std::abs(inf.dffits[i]) > 3 * std::sqrt(pd / (kd - pd)) ||
                                  random::chi_squared_cdf(cook, pd) > 0.5 || inf.hat[i] > 3 * pd / kd || big);
    }
    return r;
}

EggerTest egger_test(const std::vector<double>& effects, const std::vector<double>& variances,
                     const EggerOptions& options) {
    check_studies(effects, variances, options.random_effects.confidence, "egger_test");
    const std::size_t k = effects.size();
    if (k < 3)
        throw std::invalid_argument("egger_test: need at least three studies");
    const bool needs_n = options.predictor == SmallStudyPredictor::sample_size ||
                         options.predictor == SmallStudyPredictor::inverse_sample_size ||
                         options.predictor == SmallStudyPredictor::inverse_sqrt_sample_size;
    if (needs_n && (options.sample_sizes.size() != k ||
                    !std::ranges::all_of(options.sample_sizes, [](double n) { return n > 0 && std::isfinite(n); })))
        throw std::invalid_argument("egger_test: sample-size predictors need k positive sample sizes");
    Matrix pred(k, 1, 0.0);
    for (std::size_t i = 0; i < k; ++i) {
        switch (options.predictor) {
        case SmallStudyPredictor::standard_error: pred(i, 0) = std::sqrt(variances[i]); break;
        case SmallStudyPredictor::variance: pred(i, 0) = variances[i]; break;
        case SmallStudyPredictor::sample_size: pred(i, 0) = options.sample_sizes[i]; break;
        case SmallStudyPredictor::inverse_sample_size: pred(i, 0) = 1 / options.sample_sizes[i]; break;
        case SmallStudyPredictor::inverse_sqrt_sample_size: pred(i, 0) = 1 / std::sqrt(options.sample_sizes[i]); break;
        }
    }
    const double conf = options.random_effects.confidence;
    EggerTest e;
    if (options.model == SmallStudyModel::random_effects) {
        const auto r = meta_regression(effects, variances, pred, {.model = options.random_effects});
        e.slope = r.coefficients[1];
        e.slope_se = r.standard_errors[1];
        e.statistic = r.statistics[1];
        e.df = r.df;
        e.p_value = r.p_values[1];
        e.limit_estimate = r.coefficients[0];
        e.limit_se = r.standard_errors[0];
        e.limit_lower = r.lower[0];
        e.limit_upper = r.upper[0];
        e.dispersion = r.variance_scale;
        e.tau2 = r.tau2;
        return e;
    }
    // Weighted least squares with a multiplicative dispersion (lm with weights).
    double m = 0;
    for (std::size_t i = 0; i < k; ++i)
        m += pred(i, 0) / static_cast<double>(k);
    double s0 = 0, s1 = 0, s2 = 0, t0 = 0, t1 = 0;
    std::vector<double> w(k);
    for (std::size_t i = 0; i < k; ++i) {
        w[i] = options.inverse_variance_weights ? 1 / variances[i] : 1.0;
        const double c = pred(i, 0) - m; // centred for stability
        s0 += w[i], s1 += w[i] * c, s2 += w[i] * c * c, t0 += w[i] * effects[i], t1 += w[i] * c * effects[i];
    }
    const double det = s0 * s2 - s1 * s1;
    if (!(det > 1e-14 * s0 * s2))
        throw std::invalid_argument("egger_test: the predictor is constant");
    const double b1 = (s0 * t1 - s1 * t0) / det, a0 = (s2 * t0 - s1 * t1) / det; // intercept at the mean
    double rss = 0;
    for (std::size_t i = 0; i < k; ++i) {
        const double res = effects[i] - a0 - b1 * (pred(i, 0) - m);
        rss += w[i] * res * res;
    }
    e.df = static_cast<double>(k) - 2;
    e.dispersion = rss / e.df;
    // Covariance of (a0, b1) is s^2 * [s2, -s1; -s1, s0] / det; the intercept at 0 is a0 - b1 m.
    const double va = e.dispersion * s2 / det, vb = e.dispersion * s0 / det, cab = -e.dispersion * s1 / det;
    e.slope = b1;
    e.slope_se = std::sqrt(vb);
    e.statistic = b1 / e.slope_se;
    e.p_value = 2 * (1 - random::student_t_cdf(std::abs(e.statistic), e.df));
    e.limit_estimate = a0 - b1 * m;
    e.limit_se = std::sqrt(va + m * m * vb - 2 * m * cab);
    const double crit = random::student_t_quantile(0.5 + conf / 2, e.df);
    e.limit_lower = e.limit_estimate - crit * e.limit_se;
    e.limit_upper = e.limit_estimate + crit * e.limit_se;
    e.tau2 = 0;
    return e;
}

} // namespace datamunge::stats
