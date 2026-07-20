#include <datamunge/stats/inla_mixed_model.hpp>

#include <datamunge/linalg/cholesky.hpp>
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
using linalg::DenseMatrix;

using detail::build_lambda;
using detail::find_random_effect_span;
using detail::mixed_model_trim;
using detail::parse_random_effect_rhs;
using detail::RandomEffectSpec;

constexpr double kMuEps = 1e-8;

double inverse_link(const INLAMixedModelFamily family, const double eta) {
    switch (family) {
        case INLAMixedModelFamily::Gaussian: return eta;
        case INLAMixedModelFamily::Binomial: return 1.0 / (1.0 + std::exp(-eta));
        case INLAMixedModelFamily::Poisson: return std::exp(eta);
    }
    return 0.0;
}

void validate_response(const INLAMixedModelFamily family, const std::vector<double>& y) {
    if (family == INLAMixedModelFamily::Binomial) {
        for (const double v : y)
            if (v < 0.0 || v > 1.0) throw std::invalid_argument("INLAMixedModel: binomial response must be in [0, 1]");
    } else if (family == INLAMixedModelFamily::Poisson) {
        for (const double v : y)
            if (v < 0.0) throw std::invalid_argument("INLAMixedModel: poisson response must be non-negative");
    }
}

// (Lambda * Lambda')^-1 for a lower-triangular Lambda with a strictly positive diagonal --
// exactly what CholeskyDecomposition::solve already implements, so just wrap Lambda as one.
// The grid integration strategy's numerical Hessian and grid points are built from raw
// theta arithmetic (theta_mode +/- perturbations), unaware of the DE search's box bounds, so
// they can probe theta slightly outside [lambda_diag_epsilon, lambda_bound] -- most commonly
// when the true mode sits right at that boundary (a near-zero variance component). Solving
// against a near-singular Lambda would silently divide by a near-zero pivot rather than fail
// cleanly, so guard it explicitly and let evaluate_laplace's try/catch treat it as an
// unusable point (exactly like it already does for a non-positive-definite Q).
DenseMatrix<double> precision_from_lambda(const DenseMatrix<double>& lambda) {
    for (std::size_t i = 0; i < lambda.rows(); ++i)
        if (lambda(i, i) <= 1e-6) throw std::domain_error("precision_from_lambda: Lambda diagonal is non-positive");
    const CholeskyDecomposition<double> as_chol{lambda, true};
    return as_chol.solve(DenseMatrix<double>::identity(lambda.rows()));
}

} // namespace

INLAMixedModel::INLAMixedModel(const dstruct::DataFrame& data, const std::string& formula, INLAMixedModelOptions options)
    : formula_text_(formula), fixed_formula_("y ~ 1"), options_(options) {
    fit(data);
}

std::string INLAMixedModel::family() const {
    switch (options_.family) {
        case INLAMixedModelFamily::Gaussian: return "gaussian";
        case INLAMixedModelFamily::Binomial: return "binomial";
        case INLAMixedModelFamily::Poisson: return "poisson";
    }
    return "unknown";
}

void INLAMixedModel::fit(const dstruct::DataFrame& data) {
    const auto span = find_random_effect_span(formula_text_);
    if (!span.has_value())
        throw std::invalid_argument("INLAMixedModel: formula must contain exactly one random-effects term '(... | group)'");
    const std::size_t pipe_pos = span->content.find('|');
    if (pipe_pos == std::string::npos) throw std::invalid_argument("INLAMixedModel: random-effects term must contain '|'");
    const std::string random_rhs = mixed_model_trim(span->content.substr(0, pipe_pos));
    group_variable_ = mixed_model_trim(span->content.substr(pipe_pos + 1));
    if (group_variable_.empty()) throw std::invalid_argument("INLAMixedModel: missing grouping variable after '|'");

    const RandomEffectSpec spec = parse_random_effect_rhs(random_rhs);
    random_intercept_ = spec.intercept;
    random_slope_vars_ = spec.slope_vars;
    if (!random_intercept_ && random_slope_vars_.empty())
        throw std::invalid_argument("INLAMixedModel: random-effects term has no terms (no intercept, no slopes)");

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
        throw std::invalid_argument("INLAMixedModel: grouping column '" + group_variable_ + "' not found");
    for (const auto& v : random_slope_vars_) {
        if (!data.has_column(v)) throw std::invalid_argument("INLAMixedModel: random-slope column '" + v + "' not found");
        if (data.column_type(v) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("INLAMixedModel: random-slope column '" + v + "' must be numeric");
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
    if (n <= p) throw std::invalid_argument("INLAMixedModel: not enough complete observations to fit the fixed effects");

    DenseMatrix<double> X(n, p, 0.0);
    std::vector<double> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) X(i, j) = design_data.X(final_positions[i], j);
        y[i] = design_data.y[final_positions[i]];
    }
    validate_response(options_.family, y);

    const std::size_t q = (random_intercept_ ? 1 : 0) + random_slope_vars_.size();
    random_effect_names_.clear();
    if (random_intercept_) random_effect_names_.emplace_back("(Intercept)");
    for (const auto& v : random_slope_vars_) random_effect_names_.push_back(v);

    const bool group_is_numeric = data.column_type(group_variable_) == dstruct::DataFrame::ColumnType::Numeric;
    DenseMatrix<double> Z(n, q, 0.0);
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
        if (random_intercept_) Z(i, col++) = 1.0;
        for (const auto& v : random_slope_vars_) Z(i, col++) = data.double_at(v, row);
    }

    if (group_labels_.size() < 2)
        throw std::invalid_argument("INLAMixedModel: grouping factor '" + group_variable_ + "' must have at least 2 distinct levels");

    group_rows_.assign(group_labels_.size(), {});
    for (std::size_t i = 0; i < n; ++i) group_rows_[group_index_[i]].push_back(i);

    observations_ = n;
    const std::size_t num_groups = group_labels_.size();
    const std::size_t q_total = q * num_groups;
    const std::size_t n_latent = p + q_total;

    // Latent-field design D = [X | block-placed Z]: row i's random-effect entries land in
    // its group's q-wide block of the q_total random-effect columns, zero elsewhere.
    DenseMatrix<double> D(n, n_latent, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) D(i, j) = X(i, j);
        const std::size_t offset = p + group_index_[i] * q;
        for (std::size_t k = 0; k < q; ++k) D(i, offset + k) = Z(i, k);
    }

    lambda_free_count_ = q * (q + 1) / 2;
    const bool has_sigma = (options_.family == INLAMixedModelFamily::Gaussian);
    const std::size_t theta_dim = lambda_free_count_ + (has_sigma ? 1 : 0);

    std::vector<double> theta_lower(theta_dim), theta_upper(theta_dim), theta_init(theta_dim);
    {
        std::size_t idx = 0;
        for (std::size_t i = 0; i < q; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                if (i == j) {
                    theta_lower[idx] = options_.lambda_diag_epsilon;
                    theta_upper[idx] = options_.lambda_bound;
                    theta_init[idx] = 1.0;
                } else {
                    theta_lower[idx] = -options_.lambda_bound;
                    theta_upper[idx] = options_.lambda_bound;
                    theta_init[idx] = 0.0;
                }
                ++idx;
            }
        }
    }
    double response_log_sd = 0.0;
    if (has_sigma) {
        double mean_y = 0.0;
        for (const double v : y) mean_y += v;
        mean_y /= static_cast<double>(n);
        double ss = 0.0;
        for (const double v : y) ss += (v - mean_y) * (v - mean_y);
        const double sd = std::sqrt(std::max(1e-6, ss / static_cast<double>(n > 1 ? n - 1 : 1)));
        response_log_sd = std::log(sd);
        theta_lower[lambda_free_count_] = response_log_sd - options_.log_sigma_search_radius;
        theta_upper[lambda_free_count_] = response_log_sd + options_.log_sigma_search_radius;
        theta_init[lambda_free_count_] = response_log_sd;
    }

    const double tau0 = 1.0 / (options_.fixed_effect_prior_sd * options_.fixed_effect_prior_sd);
    const std::size_t lambda_free_count = lambda_free_count_;

    const bayes::INLAPrecisionFn precision = [p, q, num_groups, n_latent, tau0, lambda_free_count](const std::vector<double>& theta) {
        DenseMatrix<double> Q(n_latent, n_latent, 0.0);
        for (std::size_t j = 0; j < p; ++j) Q(j, j) = tau0;

        const std::vector<double> lambda_theta(theta.begin(), theta.begin() + static_cast<long>(lambda_free_count));
        const DenseMatrix<double> lambda = build_lambda(lambda_theta, q);
        const DenseMatrix<double> block = precision_from_lambda(lambda);
        for (std::size_t g = 0; g < num_groups; ++g) {
            const std::size_t offset = p + g * q;
            for (std::size_t a = 0; a < q; ++a)
                for (std::size_t b = 0; b < q; ++b) Q(offset + a, offset + b) = block(a, b);
        }
        return Q;
    };

    bayes::INLALikelihoodFn likelihood;
    switch (options_.family) {
        case INLAMixedModelFamily::Gaussian:
            likelihood = bayes::inla_gaussian_likelihood(
                [lambda_free_count](const std::vector<double>& theta) { return std::exp(theta[lambda_free_count]); });
            break;
        case INLAMixedModelFamily::Binomial: likelihood = bayes::inla_binomial_logit_likelihood(); break;
        case INLAMixedModelFamily::Poisson: likelihood = bayes::inla_poisson_log_likelihood(); break;
    }

    const bayes::INLALogPriorFn log_prior = [](const std::vector<double>&) { return 0.0; };

    bayes::INLAOptions inla_options;
    inla_options.strategy = options_.strategy;
    inla_options.grid_points_per_dim = options_.grid_points_per_dim;
    inla_options.grid_span = options_.grid_span;
    inla_options.mode_population_size = options_.mode_population_size;
    inla_options.mode_max_generations = options_.mode_max_generations;
    inla_options.seed = options_.seed;
    const bayes::INLA inla(inla_options);

    const auto result = inla.fit(D, y, likelihood, precision, log_prior, theta_lower, theta_upper, theta_init);

    fixed_effects_mean_.assign(result.latent_mean.begin(), result.latent_mean.begin() + static_cast<long>(p));
    fixed_effects_sd_.assign(result.latent_sd.begin(), result.latent_sd.begin() + static_cast<long>(p));

    random_effects_mean_.assign(num_groups, std::vector<double>(q, 0.0));
    random_effects_sd_.assign(num_groups, std::vector<double>(q, 0.0));
    for (std::size_t g = 0; g < num_groups; ++g) {
        for (std::size_t k = 0; k < q; ++k) {
            random_effects_mean_[g][k] = result.latent_mean[p + g * q + k];
            random_effects_sd_[g][k] = result.latent_sd[p + g * q + k];
        }
    }

    theta_mean_ = result.theta_mean;
    log_marginal_likelihood_ = result.log_marginal_likelihood;
}

std::vector<double> INLAMixedModel::random_effect_std_devs() const {
    const std::size_t q = random_effect_names_.size();
    const std::vector<double> lambda_theta(theta_mean_.begin(), theta_mean_.begin() + static_cast<long>(lambda_free_count_));
    const DenseMatrix<double> lambda = build_lambda(lambda_theta, q);
    std::vector<double> sd(q);
    for (std::size_t i = 0; i < q; ++i) {
        double s = 0.0;
        for (std::size_t k = 0; k < q; ++k) s += lambda(i, k) * lambda(i, k);
        sd[i] = std::sqrt(s);
    }
    return sd;
}

double INLAMixedModel::residual_std_dev() const {
    if (options_.family != INLAMixedModelFamily::Gaussian)
        throw std::logic_error("INLAMixedModel::residual_std_dev: only defined for the Gaussian family");
    return std::exp(theta_mean_[lambda_free_count_]);
}

std::string INLAMixedModel::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void INLAMixedModel::print_summary() const { print_summary(std::cout); }

void INLAMixedModel::print_summary(std::ostream& os) const {
    os << "Generalized linear mixed model fit by INLA (" << family() << ")\n";
    os << "Formula: " << formula_text_ << "\n\n";

    os << "log marginal likelihood: " << log_marginal_likelihood_ << "\n\n";

    const auto sd = random_effect_std_devs();
    os << "Random effects (posterior mean std. dev., at theta's posterior mean):\n";
    os << ' ' << std::left << std::setw(9) << "Groups" << std::setw(13) << "Name" << std::right << std::setw(10) << "Std.Dev."
       << "\n";
    os << std::fixed << std::setprecision(4);
    for (std::size_t i = 0; i < sd.size(); ++i) {
        os << ' ' << std::left << std::setw(9) << (i == 0 ? group_variable_ : std::string()) << std::setw(13)
           << random_effect_names_[i] << std::right << std::setw(10) << sd[i] << "\n";
    }
    if (options_.family == INLAMixedModelFamily::Gaussian) os << "Residual Std.Dev.: " << residual_std_dev() << "\n";
    os << "Number of obs: " << observations_ << ", groups: " << group_variable_ << ", " << group_labels_.size() << "\n\n";

    os << "Fixed effects (posterior mean +/- posterior sd):\n";
    os << std::left << std::setw(16) << "" << std::right << std::setw(12) << "Mean" << std::setw(12) << "SD" << "\n";
    for (std::size_t j = 0; j < fixed_effects_mean_.size(); ++j) {
        os << std::left << std::setw(16) << design_.coefficient_names[j] << std::right << std::setw(12) << fixed_effects_mean_[j]
           << std::setw(12) << fixed_effects_sd_[j] << "\n";
    }
}

std::vector<double> INLAMixedModel::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "INLAMixedModel::predict");
    const auto design_data = design_.build_matrix(newdata, /*require_response=*/false);
    const std::size_t p = fixed_effects_mean_.size();

    std::unordered_map<std::string, std::size_t> group_lookup;
    for (std::size_t g = 0; g < group_labels_.size(); ++g) group_lookup.emplace(group_labels_[g], g);

    const bool group_is_numeric =
        newdata.has_column(group_variable_) && newdata.column_type(group_variable_) == dstruct::DataFrame::ColumnType::Numeric;

    std::vector<double> preds(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < design_data.used_row_indices.size(); ++i) {
        const std::size_t row = design_data.used_row_indices[i];
        double eta = 0.0;
        for (std::size_t j = 0; j < p; ++j) eta += design_data.X(i, j) * fixed_effects_mean_[j];

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
                const auto& bg = random_effects_mean_[it->second];
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
