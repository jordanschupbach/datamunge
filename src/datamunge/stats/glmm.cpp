#include <datamunge/stats/glmm.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/optim/differential_evolution.hpp>
#include <datamunge/random/distributions.hpp>
#include <datamunge/stats/detail/mixed_model_formula.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace datamunge::stats {

namespace {

using linalg::CholeskyDecomposition;
using linalg::cholesky;
using linalg::DenseMatrix;

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kMuEps = 1e-8;

using detail::build_lambda;
using detail::find_random_effect_span;
using detail::mixed_model_trim;
using detail::parse_random_effect_rhs;
using detail::RandomEffectSpec;

double inverse_link(const GLMMFamily family, const double eta) {
    switch (family) {
        case GLMMFamily::Binomial: return 1.0 / (1.0 + std::exp(-eta));
        case GLMMFamily::Poisson: return std::exp(eta);
    }
    return 0.0;
}

double link_derivative(const GLMMFamily family, const double mu) {
    switch (family) {
        case GLMMFamily::Binomial: return 1.0 / (mu * (1.0 - mu));
        case GLMMFamily::Poisson: return 1.0 / mu;
    }
    return 0.0;
}

double variance_function(const GLMMFamily family, const double mu) {
    switch (family) {
        case GLMMFamily::Binomial: return mu * (1.0 - mu);
        case GLMMFamily::Poisson: return mu;
    }
    return 0.0;
}

double clamp_mu(const GLMMFamily family, const double mu) {
    if (family == GLMMFamily::Binomial) return std::min(1.0 - kMuEps, std::max(kMuEps, mu));
    return std::max(kMuEps, mu);
}

void validate_response(const GLMMFamily family, const std::vector<double>& y) {
    if (family == GLMMFamily::Binomial) {
        for (const double v : y)
            if (v < 0.0 || v > 1.0) throw std::invalid_argument("GLMM: binomial response must be in [0, 1]");
    } else {
        for (const double v : y)
            if (v < 0.0) throw std::invalid_argument("GLMM: poisson response must be non-negative");
    }
}

struct WeightedGroupBlock {
    CholeskyDecomposition<double> chol_m; // Cholesky of M_g = Z_g*Lambda*Lambda'*Z_g' + diag(1/w)
};

struct WeightedProfileFit {
    std::vector<WeightedGroupBlock> blocks;
    std::vector<double> beta;
    DenseMatrix<double> xtwx; // sum_g X_g' M_g^-1 X_g  (p x p)
    double rss{0.0};
    double criterion{0.0}; // -2 log (working) pseudo-likelihood, dispersion fixed at 1
};

// Weighted analogue of LMM's per-group profiled fit: M_g = Z_g*Lambda*Lambda'*Z_g' +
// diag(1/w) instead of + I, and there is no free dispersion to profile out (fixed at 1 for
// Binomial/Poisson), so the criterion is a direct quadratic + log-determinant sum rather
// than LMM's log(RSS/n) form.
WeightedProfileFit compute_weighted_profile(const std::vector<double>& theta, const DenseMatrix<double>& X,
                                             const std::vector<double>& z, const std::vector<double>& w,
                                             const DenseMatrix<double>& Z,
                                             const std::vector<std::vector<std::size_t>>& group_rows, const std::size_t q) {
    const std::size_t p = X.cols();
    const DenseMatrix<double> lambda = build_lambda(theta, q);

    WeightedProfileFit result;
    result.blocks.resize(group_rows.size());
    result.xtwx = DenseMatrix<double>(p, p, 0.0);
    std::vector<double> xtwz(p, 0.0);
    double log_det_sum = 0.0;

    for (std::size_t g = 0; g < group_rows.size(); ++g) {
        const auto& rows = group_rows[g];
        const std::size_t ng = rows.size();

        DenseMatrix<double> zl(ng, q, 0.0); // Z_g * Lambda
        for (std::size_t a = 0; a < ng; ++a)
            for (std::size_t k = 0; k < q; ++k) {
                double s = 0.0;
                for (std::size_t c = 0; c <= k; ++c) s += Z(rows[a], c) * lambda(k, c);
                zl(a, k) = s;
            }

        DenseMatrix<double> M(ng, ng, 0.0);
        for (std::size_t a = 0; a < ng; ++a) {
            for (std::size_t b = 0; b < ng; ++b) {
                double s = (a == b) ? (1.0 / w[rows[a]]) : 0.0;
                for (std::size_t k = 0; k < q; ++k) s += zl(a, k) * zl(b, k);
                M(a, b) = s;
            }
        }

        auto chol_m = cholesky(M);
        if (!chol_m.ok) throw std::runtime_error("GLMM: variance-component matrix is not positive definite");
        double log_det = 0.0;
        for (std::size_t a = 0; a < ng; ++a) log_det += 2.0 * std::log(chol_m.L(a, a));
        result.blocks[g] = WeightedGroupBlock{chol_m};
        log_det_sum += log_det;

        DenseMatrix<double> Xg(ng, p, 0.0);
        std::vector<double> zg(ng);
        for (std::size_t a = 0; a < ng; ++a) {
            for (std::size_t j = 0; j < p; ++j) Xg(a, j) = X(rows[a], j);
            zg[a] = z[rows[a]];
        }
        const DenseMatrix<double> minv_xg = chol_m.solve(Xg);
        const std::vector<double> minv_zg = chol_m.solve(zg);

        for (std::size_t j1 = 0; j1 < p; ++j1) {
            for (std::size_t j2 = 0; j2 < p; ++j2) {
                double s = 0.0;
                for (std::size_t a = 0; a < ng; ++a) s += Xg(a, j1) * minv_xg(a, j2);
                result.xtwx(j1, j2) += s;
            }
            double sy = 0.0;
            for (std::size_t a = 0; a < ng; ++a) sy += Xg(a, j1) * minv_zg[a];
            xtwz[j1] += sy;
        }
    }

    auto chol_xtwx = cholesky(result.xtwx);
    if (!chol_xtwx.ok) throw std::runtime_error("GLMM: fixed-effects design is rank-deficient");
    result.beta = chol_xtwx.solve(xtwz);

    double rss = 0.0;
    for (std::size_t g = 0; g < group_rows.size(); ++g) {
        const auto& rows = group_rows[g];
        const std::size_t ng = rows.size();
        std::vector<double> rg(ng);
        for (std::size_t a = 0; a < ng; ++a) {
            double fitted = 0.0;
            for (std::size_t j = 0; j < p; ++j) fitted += X(rows[a], j) * result.beta[j];
            rg[a] = z[rows[a]] - fitted;
        }
        const auto minv_rg = result.blocks[g].chol_m.solve(rg);
        for (std::size_t a = 0; a < ng; ++a) rss += rg[a] * minv_rg[a];
    }
    result.rss = rss;

    const double n = static_cast<double>(z.size());
    result.criterion = n * std::log(kTwoPi) + log_det_sum + rss;
    return result;
}

class GLMMObjective : public optim::ArbitraryFunction {
public:
    GLMMObjective(const DenseMatrix<double>& X, const std::vector<double>& z, const std::vector<double>& w,
                  const DenseMatrix<double>& Z, const std::vector<std::vector<std::size_t>>& group_rows, const std::size_t q)
        : X_(X), z_(z), w_(w), Z_(Z), group_rows_(group_rows), q_(q) {}

    double evaluate(const std::vector<double>& theta) override {
        try {
            return compute_weighted_profile(theta, X_, z_, w_, Z_, group_rows_, q_).criterion;
        } catch (const std::runtime_error&) {
            return std::numeric_limits<double>::max() / 4.0;
        }
    }

private:
    const DenseMatrix<double>& X_;
    const std::vector<double>& z_;
    const std::vector<double>& w_;
    const DenseMatrix<double>& Z_;
    const std::vector<std::vector<std::size_t>>& group_rows_;
    std::size_t q_;
};

} // namespace

GLMM::GLMM(const dstruct::DataFrame& data, const std::string& formula, GLMMOptions options)
    : formula_text_(formula), fixed_formula_("y ~ 1"), options_(options) {
    fit(data);
}

std::string GLMM::family() const {
    switch (options_.family) {
        case GLMMFamily::Binomial: return "binomial";
        case GLMMFamily::Poisson: return "poisson";
    }
    return "unknown";
}

void GLMM::fit(const dstruct::DataFrame& data) {
    const auto span = find_random_effect_span(formula_text_);
    if (!span.has_value())
        throw std::invalid_argument("GLMM: formula must contain exactly one random-effects term '(... | group)'");
    const std::size_t pipe_pos = span->content.find('|');
    if (pipe_pos == std::string::npos) throw std::invalid_argument("GLMM: random-effects term must contain '|'");
    const std::string random_rhs = mixed_model_trim(span->content.substr(0, pipe_pos));
    group_variable_ = mixed_model_trim(span->content.substr(pipe_pos + 1));
    if (group_variable_.empty()) throw std::invalid_argument("GLMM: missing grouping variable after '|'");

    const RandomEffectSpec spec = parse_random_effect_rhs(random_rhs);
    random_intercept_ = spec.intercept;
    random_slope_vars_ = spec.slope_vars;
    if (!random_intercept_ && random_slope_vars_.empty())
        throw std::invalid_argument("GLMM: random-effects term has no terms (no intercept, no slopes)");

    std::string fixed_text = formula_text_;
    fixed_text.erase(span->start, span->end - span->start + 1);
    fixed_text = mixed_model_trim(fixed_text);
    if (!fixed_text.empty() && fixed_text.back() == '+') {
        fixed_text.pop_back();
        fixed_text = mixed_model_trim(fixed_text);
    }
    if (!fixed_text.empty() && fixed_text.back() == '~') fixed_text += " 1";

    fixed_formula_ = Formula(fixed_text);
    design_ = fixed_formula_.resolve(data);
    const auto design_data = design_.build_matrix(data, /*require_response=*/true);

    if (!data.has_column(group_variable_))
        throw std::invalid_argument("GLMM: grouping column '" + group_variable_ + "' not found");
    for (const auto& v : random_slope_vars_) {
        if (!data.has_column(v)) throw std::invalid_argument("GLMM: random-slope column '" + v + "' not found");
        if (data.column_type(v) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("GLMM: random-slope column '" + v + "' must be numeric");
    }

    std::vector<std::size_t> final_rows;
    std::vector<std::size_t> final_positions;
    for (std::size_t i = 0; i < design_data.used_row_indices.size(); ++i) {
        const std::size_t row = design_data.used_row_indices[i];
        if (data.is_null(group_variable_, row)) continue;
        bool ok = true;
        for (const auto& v : random_slope_vars_) {
            if (data.is_null(v, row)) {
                ok = false;
                break;
            }
        }
        if (!ok) continue;
        final_rows.push_back(row);
        final_positions.push_back(i);
    }

    const std::size_t n = final_rows.size();
    const std::size_t p = design_data.X.cols();
    if (n <= p) throw std::invalid_argument("GLMM: not enough complete observations to fit the fixed effects");

    design_matrix_ = DenseMatrix<double>(n, p, 0.0);
    response_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) design_matrix_(i, j) = design_data.X(final_positions[i], j);
        response_[i] = design_data.y[final_positions[i]];
    }
    validate_response(options_.family, response_);

    const std::size_t q = (random_intercept_ ? 1 : 0) + random_slope_vars_.size();
    random_effect_names_.clear();
    if (random_intercept_) random_effect_names_.emplace_back("(Intercept)");
    for (const auto& v : random_slope_vars_) random_effect_names_.push_back(v);

    const bool group_is_numeric = data.column_type(group_variable_) == dstruct::DataFrame::ColumnType::Numeric;
    random_effect_design_ = DenseMatrix<double>(n, q, 0.0);
    group_index_.assign(n, 0);
    group_labels_.clear();
    std::unordered_map<std::string, std::size_t> group_lookup;

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t row = final_rows[i];
        std::string key;
        if (group_is_numeric) {
            std::ostringstream oss;
            oss << data.double_at(group_variable_, row);
            key = oss.str();
        } else {
            key = data.string_at(group_variable_, row);
        }
        const auto it = group_lookup.find(key);
        std::size_t gidx;
        if (it == group_lookup.end()) {
            gidx = group_labels_.size();
            group_labels_.push_back(key);
            group_lookup.emplace(key, gidx);
        } else {
            gidx = it->second;
        }
        group_index_[i] = gidx;

        std::size_t col = 0;
        if (random_intercept_) random_effect_design_(i, col++) = 1.0;
        for (const auto& v : random_slope_vars_) random_effect_design_(i, col++) = data.double_at(v, row);
    }

    if (group_labels_.size() < 2)
        throw std::invalid_argument("GLMM: grouping factor '" + group_variable_ + "' must have at least 2 distinct levels");

    group_rows_.assign(group_labels_.size(), {});
    for (std::size_t i = 0; i < n; ++i) group_rows_[group_index_[i]].push_back(i);

    observations_ = n;

    // ---- Penalized quasi-likelihood: outer IRLS re-linearization, inner theta search ----
    std::vector<double> beta(p, 0.0);
    {
        double mean_y = 0.0;
        for (const double v : response_) mean_y += v;
        mean_y = clamp_mu(options_.family, mean_y / static_cast<double>(n));
        const double eta0 = (options_.family == GLMMFamily::Binomial) ? std::log(mean_y / (1.0 - mean_y)) : std::log(mean_y);
        if (design_.has_intercept && p > 0) beta[0] = eta0;
    }
    std::vector<std::vector<double>> b(group_labels_.size(), std::vector<double>(q, 0.0));

    std::vector<double> theta(q * (q + 1) / 2);
    {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < q; ++i)
            for (std::size_t j = 0; j <= i; ++j) theta[idx++] = (i == j) ? 1.0 : 0.0;
    }
    std::vector<double> lower(theta.size()), upper(theta.size());
    {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < q; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                if (i == j) {
                    lower[idx] = 0.0;
                    upper[idx] = options_.theta_bound;
                } else {
                    lower[idx] = -options_.theta_bound;
                    upper[idx] = options_.theta_bound;
                }
                ++idx;
            }
        }
    }

    std::vector<double> prev_beta = beta;
    std::vector<double> z(n), w(n);

    for (std::size_t iter = 0; iter < options_.max_iterations; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            double e = 0.0;
            for (std::size_t j = 0; j < p; ++j) e += design_matrix_(i, j) * beta[j];
            const auto& bg = b[group_index_[i]];
            for (std::size_t k = 0; k < q; ++k) e += random_effect_design_(i, k) * bg[k];
            const double mu = clamp_mu(options_.family, inverse_link(options_.family, e));
            const double gprime = link_derivative(options_.family, mu);
            z[i] = e + (response_[i] - mu) * gprime;
            w[i] = 1.0 / (gprime * gprime * variance_function(options_.family, mu));
        }

        GLMMObjective objective(design_matrix_, z, w, random_effect_design_, group_rows_, q);
        optim::DEOptions de_options;
        de_options.population_size = options_.de_population_size;
        de_options.max_generations = options_.de_max_generations;
        de_options.seed = options_.seed;
        optim::DifferentialEvolution de(de_options);
        de.optimize(objective, theta, lower, upper);
        lambda_ = build_lambda(theta, q);

        const auto profile = compute_weighted_profile(theta, design_matrix_, z, w, random_effect_design_, group_rows_, q);
        beta = profile.beta;

        for (std::size_t g = 0; g < group_rows_.size(); ++g) {
            const auto& rows = group_rows_[g];
            const std::size_t ng = rows.size();
            std::vector<double> rg(ng);
            for (std::size_t a = 0; a < ng; ++a) {
                double f = 0.0;
                for (std::size_t j = 0; j < p; ++j) f += design_matrix_(rows[a], j) * beta[j];
                rg[a] = z[rows[a]] - f;
            }
            const auto minv_rg = profile.blocks[g].chol_m.solve(rg);

            std::vector<double> zt_minv_r(q, 0.0);
            for (std::size_t k = 0; k < q; ++k) {
                double s = 0.0;
                for (std::size_t a = 0; a < ng; ++a) s += random_effect_design_(rows[a], k) * minv_rg[a];
                zt_minv_r[k] = s;
            }
            std::vector<double> bg(q, 0.0);
            for (std::size_t i2 = 0; i2 < q; ++i2) {
                double s = 0.0;
                for (std::size_t k = 0; k < q; ++k) {
                    double lltk = 0.0;
                    for (std::size_t c = 0; c < q; ++c) lltk += lambda_(i2, c) * lambda_(k, c);
                    s += lltk * zt_minv_r[k];
                }
                bg[i2] = s;
            }
            b[g] = bg;
        }

        double max_change = 0.0;
        for (std::size_t j = 0; j < p; ++j) max_change = std::max(max_change, std::abs(beta[j] - prev_beta[j]));
        prev_beta = beta;
        iterations_used_ = iter + 1;
        if (max_change < options_.tol) break;
    }

    coefficients_ = beta;
    random_effects_ = b;

    linear_predictors_.resize(n);
    fitted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        double e = 0.0;
        for (std::size_t j = 0; j < p; ++j) e += design_matrix_(i, j) * coefficients_[j];
        const auto& bg = b[group_index_[i]];
        for (std::size_t k = 0; k < q; ++k) e += random_effect_design_(i, k) * bg[k];
        linear_predictors_[i] = e;
        const double mu = clamp_mu(options_.family, inverse_link(options_.family, e));
        fitted_[i] = mu;
        const double gprime = link_derivative(options_.family, mu);
        z[i] = e + (response_[i] - mu) * gprime;
        w[i] = 1.0 / (gprime * gprime * variance_function(options_.family, mu));
    }

    const auto final_profile = compute_weighted_profile(theta, design_matrix_, z, w, random_effect_design_, group_rows_, q);
    deviance_ = final_profile.criterion;

    fixed_effect_covariance_ = cholesky(final_profile.xtwx).solve(DenseMatrix<double>::identity(p));
    standard_errors_.resize(p);
    z_values_.resize(p);
    p_values_.resize(p);
    for (std::size_t j = 0; j < p; ++j) {
        standard_errors_[j] = std::sqrt(std::max(0.0, fixed_effect_covariance_(j, j)));
        z_values_[j] = coefficients_[j] / standard_errors_[j];
        p_values_[j] = 2.0 * (1.0 - random::normal_cdf(std::abs(z_values_[j])));
    }

    random_effect_covariance_ = DenseMatrix<double>(q, q, 0.0);
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j < q; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < q; ++k) s += lambda_(i, k) * lambda_(j, k);
            random_effect_covariance_(i, j) = s;
        }
}

std::vector<double> GLMM::random_effect_std_devs() const {
    std::vector<double> sd(random_effect_covariance_.rows());
    for (std::size_t i = 0; i < sd.size(); ++i) sd[i] = std::sqrt(std::max(0.0, random_effect_covariance_(i, i)));
    return sd;
}

double GLMM::random_effect_correlation(const std::size_t i, const std::size_t j) const {
    const double sii = random_effect_covariance_(i, i);
    const double sjj = random_effect_covariance_(j, j);
    if (sii <= 0.0 || sjj <= 0.0) return 0.0;
    return random_effect_covariance_(i, j) / std::sqrt(sii * sjj);
}

double GLMM::aic() const {
    const std::size_t q = random_effect_names_.size();
    const std::size_t k = coefficients_.size() + q * (q + 1) / 2;
    return deviance_ + 2.0 * static_cast<double>(k);
}

double GLMM::bic() const {
    const std::size_t q = random_effect_names_.size();
    const std::size_t k = coefficients_.size() + q * (q + 1) / 2;
    return deviance_ + std::log(static_cast<double>(observations_)) * static_cast<double>(k);
}

std::string GLMM::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void GLMM::print_summary() const { print_summary(std::cout); }

void GLMM::print_summary(std::ostream& os) const {
    const std::size_t q = random_effect_names_.size();
    os << "Generalized linear mixed model fit by PQL (" << family() << ", " << (options_.family == GLMMFamily::Binomial ? "logit" : "log")
       << " link)\n";
    os << "Formula: " << formula_text_ << "\n\n";

    os << "  AIC       BIC    deviance\n";
    os << std::fixed << std::setprecision(1);
    os << ' ' << std::setw(8) << aic() << ' ' << std::setw(8) << bic() << ' ' << std::setw(8) << deviance_ << "\n\n";

    const auto sd = random_effect_std_devs();
    os << "Random effects:\n";
    os << ' ' << std::left << std::setw(9) << "Groups" << std::setw(13) << "Name" << std::right << std::setw(10)
       << "Variance" << std::setw(10) << "Std.Dev." << std::setw(7) << "Corr" << "\n";
    os << std::setprecision(4);
    for (std::size_t i = 0; i < q; ++i) {
        os << ' ' << std::left << std::setw(9) << (i == 0 ? group_variable_ : std::string()) << std::setw(13)
           << random_effect_names_[i] << std::right << std::setw(10) << random_effect_covariance_(i, i) << std::setw(10)
           << sd[i];
        for (std::size_t j = 0; j < i; ++j) os << std::setw(7) << random_effect_correlation(i, j);
        os << "\n";
    }
    os << "Number of obs: " << observations_ << ", groups: " << group_variable_ << ", " << group_labels_.size() << "\n";
    os << "PQL outer iterations used: " << iterations_used_ << "\n\n";

    os << "Fixed effects:\n";
    os << std::left << std::setw(16) << "" << std::right << std::setw(12) << "Estimate" << std::setw(12) << "Std.Error"
       << std::setw(10) << "z value" << std::setw(12) << "Pr(>|z|)" << "\n";
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        os << std::left << std::setw(16) << design_.coefficient_names[j] << std::right << std::setw(12) << coefficients_[j]
           << std::setw(12) << standard_errors_[j] << std::setw(10) << z_values_[j] << std::setw(12) << p_values_[j] << "\n";
    }
}

std::vector<double> GLMM::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "GLMM::predict");
    const auto design_data = design_.build_matrix(newdata, /*require_response=*/false);
    const std::size_t p = coefficients_.size();

    std::unordered_map<std::string, std::size_t> group_lookup;
    for (std::size_t g = 0; g < group_labels_.size(); ++g) group_lookup.emplace(group_labels_[g], g);

    const bool group_is_numeric =
        newdata.has_column(group_variable_) && newdata.column_type(group_variable_) == dstruct::DataFrame::ColumnType::Numeric;

    std::vector<double> preds(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < design_data.used_row_indices.size(); ++i) {
        const std::size_t row = design_data.used_row_indices[i];
        double eta = 0.0;
        for (std::size_t j = 0; j < p; ++j) eta += design_data.X(i, j) * coefficients_[j];

        if (newdata.has_column(group_variable_) && !newdata.is_null(group_variable_, row)) {
            std::string key;
            if (group_is_numeric) {
                std::ostringstream oss;
                oss << newdata.double_at(group_variable_, row);
                key = oss.str();
            } else {
                key = newdata.string_at(group_variable_, row);
            }
            const auto it = group_lookup.find(key);
            if (it != group_lookup.end()) {
                const auto& bg = random_effects_[it->second];
                std::size_t col = 0;
                if (random_intercept_) eta += bg[col++];
                for (const auto& v : random_slope_vars_) {
                    if (newdata.has_column(v) && !newdata.is_null(v, row)) eta += newdata.double_at(v, row) * bg[col];
                    ++col;
                }
            }
        }
        preds[row] = inverse_link(options_.family, eta);
    }
    return preds;
}

} // namespace datamunge::stats
