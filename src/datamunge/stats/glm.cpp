#include <datamunge/stats/glm.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/regression.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

constexpr double kPi = 3.14159265358979323846;

std::vector<double> expand_to_full(const std::vector<std::size_t>& used_rows, std::size_t total_rows,
                                   const std::vector<double>& compact) {
    std::vector<double> out(total_rows, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < used_rows.size(); ++i) out[used_rows[i]] = compact[i];
    return out;
}

double quantile_type7(const std::vector<double>& sorted, double p) {
    if (sorted.size() == 1) return sorted.front();
    const double      n  = static_cast<double>(sorted.size());
    const double      h  = (n - 1.0) * p;
    const std::size_t lo = static_cast<std::size_t>(std::floor(h));
    const std::size_t hi = std::min(lo + 1, sorted.size() - 1);
    return sorted[lo] + (h - static_cast<double>(lo)) * (sorted[hi] - sorted[lo]);
}

std::string format_stat(double v, int precision = 5) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Inf" : "-Inf";
    std::ostringstream oss;
    const double mag = std::fabs(v);
    if (v != 0.0 && (mag >= 1e6 || mag < 1e-4)) {
        oss << std::scientific << std::setprecision(3) << v;
    } else {
        oss << std::fixed << std::setprecision(precision) << v;
    }
    return oss.str();
}

std::string format_pvalue(double p) {
    std::ostringstream oss;
    if (p < 1e-4) {
        oss << std::scientific << std::setprecision(3) << p;
    } else {
        oss << std::fixed << std::setprecision(4) << p;
    }
    return oss.str();
}

std::string significance_stars(double p) {
    if (p < 0.001) return "***";
    if (p < 0.01) return "**";
    if (p < 0.05) return "*";
    if (p < 0.1) return ".";
    return "";
}

const char* family_name(GLMFamily f) {
    switch (f) {
        case GLMFamily::Gaussian: return "gaussian";
        case GLMFamily::Binomial: return "binomial";
        case GLMFamily::Poisson: return "poisson";
        case GLMFamily::Gamma: return "Gamma";
    }
    return "unknown";
}

const char* link_name(GLMFamily f) {
    switch (f) {
        case GLMFamily::Gaussian: return "identity";
        case GLMFamily::Binomial: return "logit";
        case GLMFamily::Poisson: return "log";
        case GLMFamily::Gamma: return "inverse";
    }
    return "unknown";
}

double glm_link(GLMFamily f, double mu) {
    switch (f) {
        case GLMFamily::Gaussian: return mu;
        case GLMFamily::Binomial: return std::log(mu / (1.0 - mu));
        case GLMFamily::Poisson: return std::log(mu);
        case GLMFamily::Gamma: return 1.0 / mu;
    }
    return mu;
}

double glm_inverse_link(GLMFamily f, double eta) {
    switch (f) {
        case GLMFamily::Gaussian: return eta;
        case GLMFamily::Binomial: return 1.0 / (1.0 + std::exp(-eta));
        case GLMFamily::Poisson: return std::exp(eta);
        case GLMFamily::Gamma: return 1.0 / eta;
    }
    return eta;
}

// d(eta)/d(mu) = g'(mu)
double glm_link_derivative(GLMFamily f, double mu) {
    switch (f) {
        case GLMFamily::Gaussian: return 1.0;
        case GLMFamily::Binomial: return 1.0 / (mu * (1.0 - mu));
        case GLMFamily::Poisson: return 1.0 / mu;
        case GLMFamily::Gamma: return -1.0 / (mu * mu);
    }
    return 1.0;
}

double glm_variance(GLMFamily f, double mu) {
    switch (f) {
        case GLMFamily::Gaussian: return 1.0;
        case GLMFamily::Binomial: return mu * (1.0 - mu);
        case GLMFamily::Poisson: return mu;
        case GLMFamily::Gamma: return mu * mu;
    }
    return 1.0;
}

double glm_unit_deviance(GLMFamily f, double y, double mu) {
    switch (f) {
        case GLMFamily::Gaussian: {
            const double d = y - mu;
            return d * d;
        }
        case GLMFamily::Binomial: {
            const double a = (y > 0.0) ? y * std::log(y / mu) : 0.0;
            const double b = (y < 1.0) ? (1.0 - y) * std::log((1.0 - y) / (1.0 - mu)) : 0.0;
            return 2.0 * (a + b);
        }
        case GLMFamily::Poisson: {
            const double a = (y > 0.0) ? y * std::log(y / mu) : 0.0;
            return 2.0 * (a - (y - mu));
        }
        case GLMFamily::Gamma:
            return 2.0 * (-std::log(y / mu) + (y - mu) / mu);
    }
    return 0.0;
}

double glm_initial_mu(GLMFamily f, double y, double prior_weight) {
    switch (f) {
        case GLMFamily::Gaussian: return y;
        case GLMFamily::Binomial: return (prior_weight * y + 0.5) / (prior_weight + 1.0);
        case GLMFamily::Poisson: return y + 0.1;
        case GLMFamily::Gamma: return y;
    }
    return y;
}

void glm_validate_response(GLMFamily f, double y) {
    switch (f) {
        case GLMFamily::Gaussian: return;
        case GLMFamily::Binomial:
            if (y != 0.0 && y != 1.0)
                throw std::invalid_argument("GLM: binomial family requires a response of exactly 0 or 1, found "
                                            + std::to_string(y));
            return;
        case GLMFamily::Poisson:
            if (y < 0.0)
                throw std::invalid_argument("GLM: poisson family requires a non-negative response, found "
                                            + std::to_string(y));
            return;
        case GLMFamily::Gamma:
            if (y <= 0.0)
                throw std::invalid_argument("GLM: Gamma family requires a strictly positive response, found "
                                            + std::to_string(y));
            return;
    }
}

bool glm_uses_fixed_dispersion(GLMFamily f) { return f == GLMFamily::Binomial || f == GLMFamily::Poisson; }

// Matches R's family$aic() functions exactly (glm.fit reports
// object$aic = family$aic(y, n, mu, wt, dev) + 2*rank).
double glm_aic_contribution(GLMFamily f, const std::vector<double>& y, const std::vector<double>& weight,
                            const std::vector<double>& mu, double deviance) {
    const std::size_t n = y.size();
    switch (f) {
        case GLMFamily::Gaussian:
            // Matches R's gaussian()$aic exactly -- it ignores prior weights.
            return static_cast<double>(n) * (std::log(deviance / static_cast<double>(n) * 2.0 * kPi) + 1.0) + 2.0;
        case GLMFamily::Binomial: {
            double ll = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double m = std::clamp(mu[i], 1e-10, 1.0 - 1e-10);
                ll += weight[i] * (y[i] * std::log(m) + (1.0 - y[i]) * std::log(1.0 - m));
            }
            return -2.0 * ll;
        }
        case GLMFamily::Poisson: {
            double ll = 0.0;
            for (std::size_t i = 0; i < n; ++i) ll += weight[i] * (y[i] * std::log(mu[i]) - mu[i] - std::lgamma(y[i] + 1.0));
            return -2.0 * ll;
        }
        case GLMFamily::Gamma: {
            double wsum = 0.0;
            for (const double w : weight) wsum += w;
            const double disp = deviance / wsum;
            const double shape = 1.0 / disp;
            double        ll    = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double scale = mu[i] * disp;
                const double logpdf =
                    (shape - 1.0) * std::log(y[i]) - y[i] / scale - shape * std::log(scale) - std::lgamma(shape);
                ll += weight[i] * logpdf;
            }
            return -2.0 * ll + 2.0;
        }
    }
    return 0.0;
}

} // namespace

GLM::GLM(const dstruct::DataFrame& data, const std::string& formula, GLMOptions options) : formula_(formula) {
    fit(data, std::move(options));
}

void GLM::fit(const dstruct::DataFrame& data, GLMOptions options) {
    options_ = options;
    if (options_.max_iter == 0) throw std::invalid_argument("GLM: max_iter must be at least 1");

    design_        = formula_.resolve(data);
    const auto raw = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = raw.X.rows();
    const std::size_t p = raw.X.cols();

    design_matrix_ = raw.X;
    response_       = raw.y;
    for (const double y : response_) glm_validate_response(options_.family, y);

    prior_weights_.clear();
    if (options_.weights_column) {
        if (!data.has_column(*options_.weights_column))
            throw std::invalid_argument("GLM: weights column not found: " + *options_.weights_column);
        prior_weights_.reserve(raw.used_row_indices.size());
        for (const auto row : raw.used_row_indices) {
            const auto w = data.optional_double_at(*options_.weights_column, row);
            if (!w) throw std::invalid_argument("GLM: weights column has a null value in a fitted row");
            if (!(*w > 0.0)) throw std::invalid_argument("GLM: weights must be strictly positive");
            prior_weights_.push_back(*w);
        }
    }
    auto prior_weight_at = [&](std::size_t i) { return prior_weights_.empty() ? 1.0 : prior_weights_[i]; };

    std::vector<double> mu(n), eta(n);
    for (std::size_t i = 0; i < n; ++i) {
        mu[i]  = glm_initial_mu(options_.family, response_[i], prior_weight_at(i));
        eta[i] = glm_link(options_.family, mu[i]);
    }

    std::vector<double> beta(p, 0.0);
    std::vector<double> final_w(n, 1.0);
    double               deviance    = std::numeric_limits<double>::infinity();

    for (std::size_t iter = 0; iter < options_.max_iter; ++iter) {
        std::vector<double> z(n), w(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double gprime = glm_link_derivative(options_.family, mu[i]);
            z[i]                = eta[i] + (response_[i] - mu[i]) * gprime;
            const double v      = glm_variance(options_.family, mu[i]);
            w[i]                = prior_weight_at(i) / (gprime * gprime * v);
        }

        linalg::DenseMatrix<double> Xw(n, p, 0.0);
        std::vector<double>          zw(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double sw = std::sqrt(w[i]);
            for (std::size_t j = 0; j < p; ++j) Xw(i, j) = design_matrix_(i, j) * sw;
            zw[i] = z[i] * sw;
        }

        const auto wls = linalg::linear_regression(Xw, zw);
        beta            = wls.coefficients;

        for (std::size_t i = 0; i < n; ++i) {
            double s = 0.0;
            for (std::size_t j = 0; j < p; ++j) s += design_matrix_(i, j) * beta[j];
            eta[i] = s;
            mu[i]  = glm_inverse_link(options_.family, eta[i]);
        }

        double new_deviance = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            new_deviance += prior_weight_at(i) * glm_unit_deviance(options_.family, response_[i], mu[i]);

        final_w = w;
        const bool converged = std::fabs(new_deviance - deviance) / (std::fabs(deviance) + 0.1) < options_.tol;
        deviance              = new_deviance;
        if (converged) break;
    }

    coefficients_       = beta;
    fitted_              = mu;
    linear_predictor_    = eta;
    deviance_            = deviance;
    observations_        = n;
    degrees_of_freedom_  = n - p;
    final_weights_       = final_w;

    {
        double wsum = 0.0, ysum = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            wsum += prior_weight_at(i);
            ysum += prior_weight_at(i) * response_[i];
        }
        const double y_bar = ysum / wsum;
        null_deviance_       = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            null_deviance_ += prior_weight_at(i) * glm_unit_deviance(options_.family, response_[i], y_bar);
    }

    if (glm_uses_fixed_dispersion(options_.family)) {
        dispersion_ = 1.0;
    } else {
        double pearson_chisq = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const double v = glm_variance(options_.family, mu[i]);
            const double d = response_[i] - mu[i];
            pearson_chisq += prior_weight_at(i) * d * d / v;
        }
        dispersion_ = pearson_chisq / static_cast<double>(degrees_of_freedom_);
    }

    linalg::DenseMatrix<double> Xw_final(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const double sw = std::sqrt(final_weights_[i]);
        for (std::size_t j = 0; j < p; ++j) Xw_final(i, j) = design_matrix_(i, j) * sw;
    }
    const auto XtX  = Xw_final.transpose() * Xw_final;
    const auto chol = linalg::cholesky(XtX);
    if (!chol.ok) throw std::runtime_error("GLM: X^T W X is not positive definite");
    xtx_inverse_ = chol.solve(linalg::DenseMatrix<double>::identity(p));
    covariance_  = xtx_inverse_;
    covariance_ *= dispersion_;

    standard_errors_.resize(p);
    test_statistics_.resize(p);
    p_values_.resize(p);
    for (std::size_t j = 0; j < p; ++j) {
        standard_errors_[j] = std::sqrt(std::max(0.0, covariance_(j, j)));
        test_statistics_[j] = coefficients_[j] / standard_errors_[j];
        p_values_[j]         = glm_uses_fixed_dispersion(options_.family)
                                  ? 2.0 * (1.0 - random::normal_cdf(std::fabs(test_statistics_[j])))
                                  : 2.0 * (1.0 - random::student_t_cdf(std::fabs(test_statistics_[j]),
                                                                       static_cast<double>(degrees_of_freedom_)));
    }

    leverage_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        double q = 0.0;
        for (std::size_t a = 0; a < p; ++a) {
            double acc = 0.0;
            for (std::size_t b = 0; b < p; ++b) acc += xtx_inverse_(a, b) * design_matrix_(i, b);
            q += design_matrix_(i, a) * acc;
        }
        leverage_[i] = final_weights_[i] * q;
    }

    deviance_residuals_.resize(n);
    pearson_residuals_.resize(n);
    standardized_residuals_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double d    = glm_unit_deviance(options_.family, response_[i], mu[i]);
        const double sign = (response_[i] >= mu[i]) ? 1.0 : -1.0;
        deviance_residuals_[i] = sign * std::sqrt(std::max(0.0, d));

        const double v          = glm_variance(options_.family, mu[i]);
        pearson_residuals_[i]    = std::sqrt(prior_weight_at(i)) * (response_[i] - mu[i]) / std::sqrt(v);

        standardized_residuals_[i] =
            deviance_residuals_[i] / std::sqrt(dispersion_ * std::max(1e-12, 1.0 - leverage_[i]));
    }

    std::vector<double> weight_vec(n);
    for (std::size_t i = 0; i < n; ++i) weight_vec[i] = prior_weight_at(i);
    aic_ = glm_aic_contribution(options_.family, response_, weight_vec, mu, deviance_) + 2.0 * static_cast<double>(p);
}

std::vector<GLM::Interval> GLM::confidence_intervals(double level) const {
    const double alpha = 1.0 - level;
    const double crit  = glm_uses_fixed_dispersion(options_.family)
                          ? random::normal_quantile(1.0 - alpha / 2.0)
                          : random::student_t_quantile(1.0 - alpha / 2.0, static_cast<double>(degrees_of_freedom_));
    std::vector<Interval> out(coefficients_.size());
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        const double margin = crit * standard_errors_[j];
        out[j]              = {coefficients_[j] - margin, coefficients_[j] + margin};
    }
    return out;
}

std::string GLM::summary() const {
    std::ostringstream out;
    out << "Call:\nglm(formula = " << formula_.text() << ", family = " << family_name(options_.family) << "(link = \""
        << link_name(options_.family) << "\"))\n\n";

    std::vector<double> sorted_resid = deviance_residuals_;
    std::sort(sorted_resid.begin(), sorted_resid.end());
    out << "Deviance Residuals:\n";
    out << "     Min       1Q   Median       3Q      Max\n";
    out << std::right << std::setw(9) << format_stat(sorted_resid.front(), 4) << " " << std::setw(8)
        << format_stat(quantile_type7(sorted_resid, 0.25), 4) << " " << std::setw(8)
        << format_stat(quantile_type7(sorted_resid, 0.5), 4) << " " << std::setw(8)
        << format_stat(quantile_type7(sorted_resid, 0.75), 4) << " " << std::setw(8)
        << format_stat(sorted_resid.back(), 4) << "\n\n";

    const bool fixed_dispersion = glm_uses_fixed_dispersion(options_.family);
    out << "Coefficients:\n";
    out << std::left << std::setw(16) << "" << std::right << std::setw(11) << "Estimate" << " " << std::setw(11)
        << "Std. Error" << " " << std::setw(9) << (fixed_dispersion ? "z value" : "t value") << " " << std::setw(11)
        << (fixed_dispersion ? "Pr(>|z|)" : "Pr(>|t|)") << "\n";
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        out << std::left << std::setw(16) << design_.coefficient_names[j] << std::right << std::setw(11)
            << format_stat(coefficients_[j]) << " " << std::setw(11) << format_stat(standard_errors_[j]) << " "
            << std::setw(9) << format_stat(test_statistics_[j], 4) << " " << std::setw(11)
            << format_pvalue(p_values_[j]) << " " << significance_stars(p_values_[j]) << "\n";
    }
    out << "---\n";
    out << "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\n";

    if (!glm_uses_fixed_dispersion(options_.family))
        out << "(Dispersion parameter estimated: " << format_stat(dispersion_, 4) << ")\n\n";

    out << "    Null deviance: " << format_stat(null_deviance_, 4) << "  on " << (observations_ - 1)
        << "  degrees of freedom\n";
    out << "Residual deviance: " << format_stat(deviance_, 4) << "  on " << degrees_of_freedom_
        << "  degrees of freedom\n";
    out << "AIC: " << format_stat(aic_, 4) << "\n";

    return out.str();
}

void GLM::print_summary(std::ostream& os) const { os << summary(); }

void GLM::print_summary() const { print_summary(std::cout); }

void GLM::validate_categorical_levels(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "GLM::predict");
}

std::vector<double> GLM::predict(const dstruct::DataFrame& newdata) const {
    return predict(newdata, GLMPredictionInterval::None).fit;
}

GLMPrediction GLM::predict(const dstruct::DataFrame& newdata, GLMPredictionInterval interval, double level) const {
    validate_categorical_levels(newdata);
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    std::vector<double> compact_eta(dm.X.rows());
    std::vector<double> compact_fit(dm.X.rows());
    for (std::size_t i = 0; i < dm.X.rows(); ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < dm.X.cols(); ++j) s += dm.X(i, j) * coefficients_[j];
        compact_eta[i] = s;
        compact_fit[i] = glm_inverse_link(options_.family, s);
    }

    GLMPrediction result;
    result.fit = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_fit);
    if (interval == GLMPredictionInterval::None) return result;

    const double alpha = 1.0 - level;
    const double crit  = glm_uses_fixed_dispersion(options_.family)
                          ? random::normal_quantile(1.0 - alpha / 2.0)
                          : random::student_t_quantile(1.0 - alpha / 2.0, static_cast<double>(degrees_of_freedom_));

    std::vector<double> compact_se(dm.X.rows());
    std::vector<double> compact_lo(dm.X.rows());
    std::vector<double> compact_hi(dm.X.rows());
    for (std::size_t i = 0; i < dm.X.rows(); ++i) {
        double q = 0.0;
        for (std::size_t a = 0; a < dm.X.cols(); ++a) {
            double acc = 0.0;
            for (std::size_t b = 0; b < dm.X.cols(); ++b) acc += xtx_inverse_(a, b) * dm.X(i, b);
            q += dm.X(i, a) * acc;
        }
        const double se_eta = std::sqrt(dispersion_ * q);
        const double lo_eta = compact_eta[i] - crit * se_eta;
        const double hi_eta = compact_eta[i] + crit * se_eta;

        // Response-scale SE via the delta method; the interval itself is
        // computed by back-transforming the link-scale bounds so it always
        // respects the family's natural range (e.g. binomial stays in [0, 1]).
        double dmu_deta;
        switch (options_.family) {
            case GLMFamily::Gaussian: dmu_deta = 1.0; break;
            case GLMFamily::Binomial: {
                const double p = compact_fit[i];
                dmu_deta        = p * (1.0 - p);
                break;
            }
            case GLMFamily::Poisson: dmu_deta = compact_fit[i]; break;
            case GLMFamily::Gamma: dmu_deta = -compact_fit[i] * compact_fit[i]; break;
            default: dmu_deta = 1.0;
        }
        compact_se[i] = std::fabs(dmu_deta) * se_eta;
        compact_lo[i] = glm_inverse_link(options_.family, lo_eta);
        compact_hi[i] = glm_inverse_link(options_.family, hi_eta);
        if (compact_lo[i] > compact_hi[i]) std::swap(compact_lo[i], compact_hi[i]);
    }

    result.se_fit = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_se);
    result.lower  = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_lo);
    result.upper  = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_hi);
    return result;
}

dstruct::DataFrame GLM::predict_frame(const dstruct::DataFrame& newdata, GLMPredictionInterval interval,
                                      double level) const {
    const auto pred = predict(newdata, interval, level);
    dstruct::DataFrame frame;
    frame.add_column("fit", pred.fit);
    if (interval != GLMPredictionInterval::None) {
        frame.add_column("se_fit", pred.se_fit);
        frame.add_column("lwr", pred.lower);
        frame.add_column("upr", pred.upper);
    }
    return frame;
}

plot::RPlot GLM::plot_residuals_vs_fitted() const {
    auto p = plot::RPlot::create();
    p.points(fitted_, deviance_residuals_, "deviance residuals");
    p.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return p;
}

plot::RPlot GLM::plot_normal_qq() const {
    std::vector<double> sorted = standardized_residuals_;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t   n = sorted.size();
    std::vector<double> theoretical(n);
    for (std::size_t i = 0; i < n; ++i)
        theoretical[i] = random::normal_quantile((static_cast<double>(i) + 0.5) / static_cast<double>(n));

    auto p = plot::RPlot::create();
    p.points(theoretical, sorted, "standardized deviance residuals");
    p.title("Normal Q-Q").x_label("Theoretical Quantiles").y_label("Standardized residuals");
    return p;
}

plot::RPlot GLM::plot_scale_location() const {
    std::vector<double> sqrt_abs_std(standardized_residuals_.size());
    for (std::size_t i = 0; i < standardized_residuals_.size(); ++i)
        sqrt_abs_std[i] = std::sqrt(std::fabs(standardized_residuals_[i]));

    auto p = plot::RPlot::create();
    p.points(fitted_, sqrt_abs_std, "sqrt(|standardized residuals|)");
    p.title("Scale-Location").x_label("Fitted values").y_label("sqrt(|Standardized residuals|)");
    return p;
}

plot::RPlot GLM::plot_residuals_vs_leverage() const {
    auto p = plot::RPlot::create();
    p.points(leverage_, standardized_residuals_, "standardized residuals");
    p.title("Residuals vs Leverage").x_label("Leverage").y_label("Standardized residuals");
    return p;
}

void GLM::save_diagnostic_plots(const std::string& path_prefix) const {
    plot_residuals_vs_fitted().save(path_prefix + "_residuals_vs_fitted.svg");
    plot_normal_qq().save(path_prefix + "_normal_qq.svg");
    plot_scale_location().save(path_prefix + "_scale_location.svg");
    plot_residuals_vs_leverage().save(path_prefix + "_residuals_vs_leverage.svg");
}

} // namespace datamunge::stats
