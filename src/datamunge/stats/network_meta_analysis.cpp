/// @file network_meta_analysis.cpp
/// @brief Generalised least squares for contrast-based network meta-analysis.
#include <datamunge/stats/network_meta_analysis.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Implementation helpers for network meta-analysis.
namespace {
/// @brief Dense matrix of doubles.
using Matrix = linalg::DenseMatrix<double>;

/// @brief Inverse of a symmetric positive definite matrix.
/// @param a Matrix.
/// @return a^{-1}.
/// @throws std::invalid_argument If a is not positive definite.
Matrix spd_inverse(const Matrix& a) {
    const auto ch = linalg::cholesky(a);
    if (!ch.ok)
        throw std::invalid_argument("network_meta_analysis: a covariance or information matrix is not positive definite");
    const std::size_t n = a.rows();
    Matrix out(n, n, 0.0);
    for (std::size_t c = 0; c < n; ++c) {
        std::vector<double> e(n, 0.0);
        e[c] = 1;
        const auto col = ch.solve(e);
        for (std::size_t r = 0; r < n; ++r)
            out(r, c) = col[r];
    }
    return out;
}

/// @brief Stacked contrast data of a set of studies.
struct Stacked {
    std::vector<double> y;           ///< All contrasts.
    std::vector<std::size_t> start;  ///< First row of each study.
    std::vector<std::size_t> size;   ///< Contrasts per study.
    std::vector<std::vector<int>> x; ///< Design rows over all treatments: +1 arm, -1 baseline.
};

/// @brief Stacks studies into contrast rows.
/// @param studies Studies.
/// @param t Number of treatments.
/// @return Stacked contrasts and design rows.
Stacked stack(const std::vector<const NetworkStudy*>& studies, std::size_t t) {
    Stacked s;
    for (const NetworkStudy* st : studies) {
        s.start.push_back(s.y.size());
        s.size.push_back(st->effects.size());
        for (std::size_t j = 0; j < st->effects.size(); ++j) {
            s.y.push_back(st->effects[j]);
            std::vector<int> row(t, 0);
            row[st->treatments[j + 1]] = 1;
            row[st->treatments[0]] = -1;
            s.x.push_back(row);
        }
    }
    return s;
}

/// @brief GLS fit at a given tau2 with parameters d_j for the treatments in `params` (others fixed at 0).
struct GlsFit {
    std::vector<double> d;     ///< Parameter estimates (indexed like params).
    Matrix covariance;         ///< Their covariance.
    double q{};                ///< Residual quadratic form (y - X d)' V^{-1} (y - X d).
    double trace_p_sigma{};    ///< tr(P Sigma).
    double score{};            ///< y' P Sigma P y.
    double trace_ps_ps{};      ///< tr(P Sigma P Sigma).
};

/// @brief Within-study covariance plus tau2 times the homogeneous heterogeneity structure, per study.
/// @param studies Studies.
/// @param tau2 Between-study variance.
/// @return Block inverses W_s = V_s^{-1} of S_s + tau2 Sigma_s (Sigma: 1 on the diagonal, 1/2 off), in study order.
std::vector<Matrix> study_weights(const std::vector<const NetworkStudy*>& studies, double tau2) {
    std::vector<Matrix> w;
    for (const NetworkStudy* st : studies) {
        Matrix v = st->covariance;
        for (std::size_t a = 0; a < v.rows(); ++a)
            for (std::size_t b = 0; b < v.cols(); ++b)
                v(a, b) += tau2 * (a == b ? 1.0 : 0.5);
        w.push_back(spd_inverse(v));
    }
    return w;
}

/// @brief Generalised least-squares fit of the consistency model.
/// @param studies Studies.
/// @param s Stacked data.
/// @param params Treatments whose effects are estimated (the others are fixed at 0).
/// @param tau2 Between-study variance.
/// @param reml_terms Whether to compute the REML score terms.
/// @return The fit.
GlsFit gls(const std::vector<const NetworkStudy*>& studies, const Stacked& s, const std::vector<std::size_t>& params,
           double tau2, bool reml_terms) {
    const std::size_t n = s.y.size(), p = params.size();
    const auto w = study_weights(studies, tau2);
    // Dense block-diagonal weight matrix and design.
    Matrix wm(n, n, 0.0), x(n, p, 0.0), sigma(n, n, 0.0);
    for (std::size_t k = 0; k < studies.size(); ++k)
        for (std::size_t a = 0; a < s.size[k]; ++a)
            for (std::size_t b = 0; b < s.size[k]; ++b) {
                wm(s.start[k] + a, s.start[k] + b) = w[k](a, b);
                sigma(s.start[k] + a, s.start[k] + b) = a == b ? 1.0 : 0.5;
            }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            x(i, j) = s.x[i][params[j]];
    Matrix wx(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t l = 0; l < n; ++l)
                wx(i, j) += wm(i, l) * x(l, j);
    Matrix info(p, p, 0.0);
    std::vector<double> rhs(p, 0.0);
    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t i = 0; i < n; ++i)
            rhs[a] += wx(i, a) * s.y[i];
        for (std::size_t b = 0; b < p; ++b)
            for (std::size_t i = 0; i < n; ++i)
                info(a, b) += x(i, a) * wx(i, b);
    }
    const auto ch = linalg::cholesky(info);
    if (!ch.ok)
        throw std::invalid_argument("network_meta_analysis: the network is disconnected");
    GlsFit f;
    f.d = ch.solve(rhs);
    f.covariance = spd_inverse(info);
    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        r[i] = s.y[i];
        for (std::size_t j = 0; j < p; ++j)
            r[i] -= x(i, j) * f.d[j];
    }
    std::vector<double> wr(n, 0.0); // P y = W r
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t l = 0; l < n; ++l)
            wr[i] += wm(i, l) * r[l];
    for (std::size_t i = 0; i < n; ++i)
        f.q += r[i] * wr[i];
    if (!reml_terms)
        return f;
    // P = W - W X C X' W.
    Matrix pm(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double v = 0;
            for (std::size_t a = 0; a < p; ++a)
                for (std::size_t b = 0; b < p; ++b)
                    v += wx(i, a) * f.covariance(a, b) * wx(j, b);
            pm(i, j) = wm(i, j) - v;
        }
    Matrix ps(n, n, 0.0); // P Sigma
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t l = 0; l < n; ++l)
                ps(i, j) += pm(i, l) * sigma(l, j);
    std::vector<double> sw(n, 0.0); // Sigma P y
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t l = 0; l < n; ++l)
            sw[i] += sigma(i, l) * wr[l];
    for (std::size_t i = 0; i < n; ++i) {
        f.trace_p_sigma += ps(i, i);
        f.score += wr[i] * sw[i];
        for (std::size_t j = 0; j < n; ++j)
            f.trace_ps_ps += ps(i, j) * ps(j, i);
    }
    return f;
}

/// @brief Pairwise estimates, intervals, p-values and P-scores from parameter estimates.
/// @param f GLS fit with parameters for all treatments except the reference.
/// @param params Treatment of each parameter.
/// @param t Number of treatments.
/// @param options Confidence and ranking direction.
/// @return The estimates.
NetworkEstimates pairwise(const GlsFit& f, const std::vector<std::size_t>& params, std::size_t t,
                          const NetworkMetaOptions& options) {
    std::vector<double> d(t, 0.0);
    std::vector<long> idx(t, -1);
    for (std::size_t j = 0; j < params.size(); ++j)
        d[params[j]] = f.d[j], idx[params[j]] = static_cast<long>(j);
    const auto cov = [&](std::size_t a, std::size_t b) {
        return idx[a] < 0 || idx[b] < 0 ? 0.0
                                        : f.covariance(static_cast<std::size_t>(idx[a]), static_cast<std::size_t>(idx[b]));
    };
    NetworkEstimates e;
    e.estimate = Matrix(t, t, 0.0);
    e.standard_error = Matrix(t, t, 0.0);
    e.lower = Matrix(t, t, 0.0);
    e.upper = Matrix(t, t, 0.0);
    e.p_value = Matrix(t, t, 1.0);
    const double z = random::normal_quantile(0.5 + options.confidence / 2);
    for (std::size_t i = 0; i < t; ++i)
        for (std::size_t j = 0; j < t; ++j) {
            const double est = d[i] - d[j];
            const double var = std::max(0.0, cov(i, i) + cov(j, j) - 2 * cov(i, j));
            const double se = std::sqrt(var);
            e.estimate(i, j) = est;
            e.standard_error(i, j) = se;
            e.lower(i, j) = est - z * se;
            e.upper(i, j) = est + z * se;
            if (i != j)
                e.p_value(i, j) = 2 * (1 - random::normal_cdf(std::abs(est / se)));
        }
    e.p_scores.assign(t, 0.0);
    for (std::size_t i = 0; i < t; ++i) {
        for (std::size_t j = 0; j < t; ++j)
            if (j != i) {
                const double zz = e.estimate(i, j) / e.standard_error(i, j);
                e.p_scores[i] += random::normal_cdf(options.larger_is_better ? zz : -zz);
            }
        e.p_scores[i] /= static_cast<double>(t - 1);
    }
    return e;
}
} // namespace

NetworkStudy network_study_from_arms(const std::vector<std::size_t>& treatments, const std::vector<double>& estimates,
                                     const std::vector<double>& variances) {
    const std::size_t m = treatments.size();
    if (m < 2 || estimates.size() != m || variances.size() != m)
        throw std::invalid_argument("network_study_from_arms: need at least two arms with matching estimates and variances");
    for (std::size_t a = 0; a < m; ++a) {
        if (!std::isfinite(estimates[a]) || !(variances[a] > 0) || !std::isfinite(variances[a]))
            throw std::invalid_argument("network_study_from_arms: estimates must be finite and variances positive");
        for (std::size_t b = 0; b < a; ++b)
            if (treatments[a] == treatments[b])
                throw std::invalid_argument("network_study_from_arms: treatments must be distinct");
    }
    NetworkStudy s;
    s.treatments = treatments;
    s.covariance = Matrix(m - 1, m - 1, variances[0]);
    for (std::size_t j = 1; j < m; ++j) {
        s.effects.push_back(estimates[j] - estimates[0]);
        s.covariance(j - 1, j - 1) += variances[j];
    }
    return s;
}

NetworkMetaAnalysis network_meta_analysis(const std::vector<NetworkStudy>& studies, const NetworkMetaOptions& options) {
    if (studies.empty() || !(options.confidence > 0 && options.confidence < 1))
        throw std::invalid_argument("network_meta_analysis: need studies and confidence in (0, 1)");
    std::size_t t = 0;
    for (const auto& s : studies) {
        const std::size_t m = s.treatments.size();
        if (m < 2 || s.effects.size() != m - 1 || s.covariance.rows() != m - 1 || s.covariance.cols() != m - 1)
            throw std::invalid_argument("network_meta_analysis: each study needs arms - 1 effects and a matching covariance");
        for (std::size_t a = 0; a < m; ++a) {
            for (std::size_t b = 0; b < a; ++b)
                if (s.treatments[a] == s.treatments[b])
                    throw std::invalid_argument("network_meta_analysis: treatments within a study must be distinct");
            t = std::max(t, s.treatments[a] + 1);
        }
        for (const double e : s.effects)
            if (!std::isfinite(e))
                throw std::invalid_argument("network_meta_analysis: effects must be finite");
    }
    if (t < 2 || options.reference >= t)
        throw std::invalid_argument("network_meta_analysis: need two or more treatments and a valid reference");
    std::vector<const NetworkStudy*> ptr;
    for (const auto& s : studies)
        ptr.push_back(&s);
    const Stacked st = stack(ptr, t);
    std::vector<std::size_t> params;
    for (std::size_t j = 0; j < t; ++j)
        if (j != options.reference)
            params.push_back(j);
    NetworkMetaAnalysis out;
    out.treatments = t;
    out.studies = studies.size();
    out.contrasts = st.y.size();
    // Every treatment must appear (otherwise the information matrix is singular: reported as disconnected).
    const GlsFit ce = gls(ptr, st, params, 0.0, true);
    out.common = pairwise(ce, params, t, options);
    out.q = ce.q;
    out.q_df = static_cast<double>(out.contrasts) - static_cast<double>(t - 1);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    out.q_p_value = out.q_df > 0 ? random::chi_squared_survival(out.q, out.q_df) : nan;
    out.i2 = out.q_df > 0 && out.q > 0 ? 100 * std::max(0.0, (out.q - out.q_df) / out.q) : 0.0;
    // Design-by-treatment decomposition: separate effects within each design.
    std::map<std::vector<std::size_t>, std::vector<const NetworkStudy*>> designs;
    for (const auto& s : studies) {
        auto key = s.treatments;
        std::ranges::sort(key);
        designs[key].push_back(&s);
    }
    out.designs = designs.size();
    for (const auto& [key, members] : designs) {
        const Stacked ds = stack(members, t);
        const std::vector<std::size_t> dp(key.begin() + 1, key.end()); // d of the first treatment fixed at 0
        const double dfd = static_cast<double>(ds.y.size()) - static_cast<double>(dp.size());
        if (dfd > 0) {
            out.q_within += gls(members, ds, dp, 0.0, false).q;
            out.q_within_df += dfd;
        }
    }
    out.q_within_p_value = out.q_within_df > 0 ? random::chi_squared_survival(out.q_within, out.q_within_df) : nan;
    out.q_between = std::max(0.0, out.q - out.q_within);
    out.q_between_df = out.q_df - out.q_within_df;
    out.q_between_p_value = out.q_between_df > 0 ? random::chi_squared_survival(out.q_between, out.q_between_df) : nan;
    // tau2.
    double tau2 = out.q_df > 0 ? std::max(0.0, (out.q - out.q_df) / ce.trace_p_sigma) : 0.0;
    if (options.estimator == NetworkTauEstimator::reml && out.q_df > 0) {
        out.converged = false;
        for (out.iterations = 1; out.iterations <= options.max_iterations; ++out.iterations) {
            const GlsFit f = gls(ptr, st, params, tau2, true);
            const double next = std::max(0.0, tau2 + (f.score - f.trace_p_sigma) / f.trace_ps_ps);
            const double change = std::abs(next - tau2);
            tau2 = next;
            if (change <= options.tolerance * std::max(1.0, tau2)) {
                out.converged = true;
                break;
            }
        }
        out.iterations = std::min(out.iterations, options.max_iterations);
    }
    out.tau2 = tau2;
    out.tau = std::sqrt(tau2);
    out.random = pairwise(gls(ptr, st, params, tau2, false), params, t, options);
    return out;
}

} // namespace datamunge::stats
