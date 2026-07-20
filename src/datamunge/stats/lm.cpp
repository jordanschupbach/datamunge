#include <datamunge/stats/lm.hpp>

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

// Fixed-precision formatting overflows its column width (and runs
// together with the next column) for very large or very small magnitudes —
// e.g. a near-perfect fit drives standard errors toward 0 and t-values
// toward astronomical numbers. Switch to scientific notation in that case
// so the string stays short regardless of magnitude.
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

} // namespace

LM::LM(const dstruct::DataFrame& data, const std::string& formula, LmOptions options) : formula_(formula) {
    fit(data, std::move(options));
}

void LM::fit(const dstruct::DataFrame& data, LmOptions options) {
    options_       = options;
    design_        = formula_.resolve(data);
    auto raw_design = design_.build_matrix(data, /*require_response=*/true);

    if (options_.weights_column) {
        if (!data.has_column(*options_.weights_column))
            throw std::invalid_argument("LM: weights column not found: " + *options_.weights_column);
        weights_.reserve(raw_design.used_row_indices.size());
        for (const auto row : raw_design.used_row_indices) {
            const auto w = data.optional_double_at(*options_.weights_column, row);
            if (!w) throw std::invalid_argument("LM: weights column has a null value in a fitted row");
            if (!(*w > 0.0)) throw std::invalid_argument("LM: weights must be strictly positive");
            weights_.push_back(*w);
        }
    }

    design_matrix_ = raw_design.X;
    response_      = raw_design.y;
    if (!weights_.empty()) {
        for (std::size_t i = 0; i < design_matrix_.rows(); ++i) {
            const double s = std::sqrt(weights_[i]);
            for (std::size_t j = 0; j < design_matrix_.cols(); ++j) design_matrix_(i, j) *= s;
            response_[i] *= s;
        }
    }

    auto result = linalg::linear_regression(design_matrix_, response_);

    coefficients_        = result.coefficients;
    standard_errors_     = result.standard_errors;
    covariance_          = result.covariance;
    sigma2_              = result.sigma2;
    sigma_               = result.sigma;
    r_squared_           = result.r_squared;
    adjusted_r_squared_  = result.adjusted_r_squared;
    observations_        = result.observations;
    degrees_of_freedom_  = result.degrees_of_freedom;

    fitted_ = raw_design.X * coefficients_;
    residuals_.resize(raw_design.y.size());
    for (std::size_t i = 0; i < raw_design.y.size(); ++i) residuals_[i] = raw_design.y[i] - fitted_[i];

    if (sigma2_ > 0.0) {
        xtx_inverse_ = covariance_;
        xtx_inverse_ *= (1.0 / sigma2_);
    } else {
        auto xtx  = design_matrix_.transpose() * design_matrix_;
        auto chol = linalg::cholesky(xtx);
        if (!chol.ok) throw std::runtime_error("LM: X^T X is not positive definite");
        xtx_inverse_ = chol.solve(linalg::DenseMatrix<double>::identity(design_matrix_.cols()));
    }

    t_values_.resize(coefficients_.size());
    p_values_.resize(coefficients_.size());
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        t_values_[j] = (standard_errors_[j] > 0.0) ? coefficients_[j] / standard_errors_[j]
                                                    : std::numeric_limits<double>::infinity();
        p_values_[j] =
            2.0 * (1.0 - random::student_t_cdf(std::fabs(t_values_[j]), static_cast<double>(degrees_of_freedom_)));
    }

    // linear_regression()'s R^2/ESS/TSS treat the *transformed* response
    // (sqrt(w) * y) as the quantity of interest, which is not the weighted
    // R^2 R reports (R weights deviations from the weighted mean of the raw
    // y). RSS is unaffected (sqrt(w_i)*residual_i squared sums to the same
    // weighted RSS either way), so only R^2/adjusted R^2/F need correcting.
    double explained_sum_of_squares = result.explained_sum_of_squares;
    if (!weights_.empty()) {
        double sum_w = 0.0, sum_wy = 0.0;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            sum_w += weights_[i];
            sum_wy += weights_[i] * raw_design.y[i];
        }
        const double weighted_mean = sum_wy / sum_w;
        double       tss = 0.0, rss = 0.0;
        for (std::size_t i = 0; i < weights_.size(); ++i) {
            const double d = raw_design.y[i] - weighted_mean;
            tss += weights_[i] * d * d;
            rss += weights_[i] * residuals_[i] * residuals_[i];
        }
        explained_sum_of_squares = tss - rss;
        r_squared_          = (tss == 0.0) ? (rss == 0.0 ? 1.0 : 0.0) : (1.0 - rss / tss);
        adjusted_r_squared_ = (observations_ > 1 && tss != 0.0)
                                 ? 1.0 - (1.0 - r_squared_)
                                       * (static_cast<double>(observations_ - 1) / static_cast<double>(degrees_of_freedom_))
                                 : r_squared_;
    }

    const std::size_t num_predictors = coefficients_.size() - (design_.has_intercept ? 1u : 0u);
    if (num_predictors > 0 && degrees_of_freedom_ > 0 && sigma2_ > 0.0) {
        f_statistic_ = (explained_sum_of_squares / static_cast<double>(num_predictors)) / sigma2_;
        f_p_value_   = 1.0 - random::f_cdf(f_statistic_, static_cast<double>(num_predictors),
                                          static_cast<double>(degrees_of_freedom_));
    } else {
        f_statistic_ = std::numeric_limits<double>::quiet_NaN();
        f_p_value_   = std::numeric_limits<double>::quiet_NaN();
    }

    compute_diagnostics();
}

void LM::compute_diagnostics() {
    const std::size_t n = design_matrix_.rows();
    const std::size_t p = coefficients_.size();
    leverage_.assign(n, 0.0);
    standardized_residuals_.assign(n, 0.0);
    studentized_residuals_.assign(n, 0.0);
    cooks_distance_.assign(n, 0.0);

    std::vector<double> working_residuals(n);
    for (std::size_t i = 0; i < n; ++i)
        working_residuals[i] = weights_.empty() ? residuals_[i] : residuals_[i] * std::sqrt(weights_[i]);

    for (std::size_t i = 0; i < n; ++i) {
        double h = 0.0;
        for (std::size_t a = 0; a < p; ++a) {
            double acc = 0.0;
            for (std::size_t b = 0; b < p; ++b) acc += xtx_inverse_(a, b) * design_matrix_(i, b);
            h += design_matrix_(i, a) * acc;
        }
        h            = std::clamp(h, 0.0, 1.0 - 1e-12);
        leverage_[i] = h;

        const double denom     = sigma_ * std::sqrt(std::max(1.0 - h, 1e-12));
        const double std_resid = (denom > 0.0) ? working_residuals[i] / denom : 0.0;
        standardized_residuals_[i] = std_resid;

        const double df           = static_cast<double>(degrees_of_freedom_);
        const double ext_variance = (df - 1.0) / std::max(df - std_resid * std_resid, 1e-12);
        studentized_residuals_[i] = (ext_variance > 0.0) ? std_resid * std::sqrt(ext_variance) : std_resid;

        cooks_distance_[i] =
            (p > 0) ? (std_resid * std_resid / static_cast<double>(p)) * (h / std::max(1.0 - h, 1e-12)) : 0.0;
    }
}

std::vector<LM::Interval> LM::confidence_intervals(double level) const {
    const double alpha  = 1.0 - level;
    const double t_crit = random::student_t_quantile(1.0 - alpha / 2.0, static_cast<double>(degrees_of_freedom_));
    std::vector<Interval> out(coefficients_.size());
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        const double margin = t_crit * standard_errors_[j];
        out[j]              = {coefficients_[j] - margin, coefficients_[j] + margin};
    }
    return out;
}

std::vector<AnovaRow> LM::anova() const {
    std::vector<AnovaRow> rows;
    if (design_.term_groups.empty()) return rows;

    const std::size_t n = design_matrix_.rows();

    auto rss_for_prefix = [&](std::size_t num_cols) -> double {
        if (num_cols == 0) {
            double s = 0.0;
            for (const double v : response_) s += v * v;
            return s;
        }
        linalg::DenseMatrix<double> prefix(n, num_cols);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < num_cols; ++j) prefix(i, j) = design_matrix_(i, j);
        return linalg::linear_regression(prefix, response_).residual_sum_of_squares;
    };

    const std::size_t intercept_cols = design_.has_intercept ? 1u : 0u;
    double            previous_rss   = rss_for_prefix(intercept_cols);
    std::size_t       running_cols   = intercept_cols;

    for (const auto& group : design_.term_groups) {
        running_cols += group.column_indices.size();
        const double      rss    = rss_for_prefix(running_cols);
        const double      seq_ss = previous_rss - rss;
        const std::size_t df     = group.column_indices.size();

        AnovaRow row;
        row.term                = group.label;
        row.degrees_of_freedom  = df;
        row.sum_sq              = seq_ss;
        row.mean_sq             = (df > 0) ? seq_ss / static_cast<double>(df) : 0.0;
        row.f_value             = row.mean_sq / sigma2_;
        row.p_value             = 1.0 - random::f_cdf(row.f_value, static_cast<double>(df),
                                                       static_cast<double>(degrees_of_freedom_));
        rows.push_back(row);

        previous_rss = rss;
    }

    AnovaRow residual_row;
    residual_row.term               = "Residuals";
    residual_row.degrees_of_freedom = degrees_of_freedom_;
    residual_row.sum_sq             = previous_rss;
    residual_row.mean_sq            = sigma2_;
    rows.push_back(residual_row);

    return rows;
}

std::string LM::summary() const {
    std::ostringstream out;
    out << "Call:\nlm(formula = " << formula_.text() << ")\n\n";

    std::vector<double> sorted_resid = residuals_;
    std::sort(sorted_resid.begin(), sorted_resid.end());
    out << "Residuals:\n";
    out << "     Min       1Q   Median       3Q      Max\n";
    out << std::right << std::setw(9) << format_stat(sorted_resid.front(), 4) << " " << std::setw(8)
        << format_stat(quantile_type7(sorted_resid, 0.25), 4) << " " << std::setw(8)
        << format_stat(quantile_type7(sorted_resid, 0.5), 4) << " " << std::setw(8)
        << format_stat(quantile_type7(sorted_resid, 0.75), 4) << " " << std::setw(8)
        << format_stat(sorted_resid.back(), 4) << "\n\n";

    out << "Coefficients:\n";
    out << std::left << std::setw(16) << "" << std::right << std::setw(11) << "Estimate" << " " << std::setw(11)
        << "Std. Error" << " " << std::setw(9) << "t value" << " " << std::setw(11) << "Pr(>|t|)"
        << "\n";
    for (std::size_t j = 0; j < coefficients_.size(); ++j) {
        out << std::left << std::setw(16) << design_.coefficient_names[j] << std::right << std::setw(11)
            << format_stat(coefficients_[j]) << " " << std::setw(11) << format_stat(standard_errors_[j]) << " "
            << std::setw(9) << format_stat(t_values_[j], 4) << " " << std::setw(11) << format_pvalue(p_values_[j])
            << " " << significance_stars(p_values_[j]) << "\n";
    }
    out << "---\n";
    out << "Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n\n";

    out << "Residual standard error: " << format_stat(sigma_, 4) << " on " << degrees_of_freedom_
        << " degrees of freedom\n";
    out << "Multiple R-squared:  " << format_stat(r_squared_, 4)
        << ",  Adjusted R-squared:  " << format_stat(adjusted_r_squared_, 4) << "\n";
    if (!std::isnan(f_statistic_)) {
        const std::size_t num_predictors = coefficients_.size() - (design_.has_intercept ? 1u : 0u);
        out << "F-statistic: " << format_stat(f_statistic_, 4) << " on " << num_predictors << " and "
            << degrees_of_freedom_ << " DF,  p-value: " << format_pvalue(f_p_value_) << "\n";
    }
    return out.str();
}

void LM::print_summary(std::ostream& os) const { os << summary(); }

void LM::print_summary() const { print_summary(std::cout); }

void LM::validate_categorical_levels(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "LM::predict");
}

std::vector<double> LM::predict(const dstruct::DataFrame& newdata) const {
    return predict(newdata, PredictionInterval::None).fit;
}

LmPrediction LM::predict(const dstruct::DataFrame& newdata, PredictionInterval interval, double level) const {
    validate_categorical_levels(newdata);
    auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    std::vector<double> compact_fit(dm.X.rows());
    for (std::size_t i = 0; i < dm.X.rows(); ++i) {
        double s = 0.0;
        for (std::size_t j = 0; j < dm.X.cols(); ++j) s += dm.X(i, j) * coefficients_[j];
        compact_fit[i] = s;
    }

    LmPrediction result;
    result.fit = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_fit);
    if (interval == PredictionInterval::None) return result;

    const double alpha  = 1.0 - level;
    const double t_crit = random::student_t_quantile(1.0 - alpha / 2.0, static_cast<double>(degrees_of_freedom_));

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
        const double var_fit = sigma2_ * q;
        const double se = (interval == PredictionInterval::Confidence) ? std::sqrt(var_fit)
                                                                        : std::sqrt(sigma2_ + var_fit);
        compact_se[i] = se;
        compact_lo[i] = compact_fit[i] - t_crit * se;
        compact_hi[i] = compact_fit[i] + t_crit * se;
    }

    result.se_fit = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_se);
    result.lower  = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_lo);
    result.upper  = expand_to_full(dm.used_row_indices, newdata.nrows(), compact_hi);
    return result;
}

dstruct::DataFrame LM::predict_frame(const dstruct::DataFrame& newdata, PredictionInterval interval,
                                     double level) const {
    const auto pred = predict(newdata, interval, level);

    auto to_optional = [](const std::vector<double>& values) {
        std::vector<std::optional<double>> out(values.size());
        for (std::size_t i = 0; i < values.size(); ++i)
            out[i] = std::isnan(values[i]) ? std::nullopt : std::optional<double>(values[i]);
        return out;
    };

    dstruct::DataFrame out;
    out.add_column("fit", to_optional(pred.fit));
    if (interval != PredictionInterval::None) {
        out.add_column("se_fit", to_optional(pred.se_fit));
        out.add_column("lwr", to_optional(pred.lower));
        out.add_column("upr", to_optional(pred.upper));
    }
    return out;
}

plot::RPlot LM::plot_residuals_vs_fitted() const {
    auto p = plot::RPlot::create();
    p.points(fitted_, residuals_, "residuals");
    p.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return p;
}

plot::RPlot LM::plot_normal_qq() const {
    std::vector<double> sorted = standardized_residuals_;
    std::sort(sorted.begin(), sorted.end());
    const std::size_t   n = sorted.size();
    std::vector<double> theoretical(n);
    for (std::size_t i = 0; i < n; ++i)
        theoretical[i] = random::normal_quantile((static_cast<double>(i) + 0.5) / static_cast<double>(n));

    auto p = plot::RPlot::create();
    p.points(theoretical, sorted, "standardized residuals");
    p.title("Normal Q-Q").x_label("Theoretical Quantiles").y_label("Standardized residuals");
    return p;
}

plot::RPlot LM::plot_scale_location() const {
    std::vector<double> sqrt_abs_std(standardized_residuals_.size());
    for (std::size_t i = 0; i < standardized_residuals_.size(); ++i)
        sqrt_abs_std[i] = std::sqrt(std::fabs(standardized_residuals_[i]));

    auto p = plot::RPlot::create();
    p.points(fitted_, sqrt_abs_std, "sqrt(|standardized residuals|)");
    p.title("Scale-Location").x_label("Fitted values").y_label("sqrt(|Standardized residuals|)");
    return p;
}

plot::RPlot LM::plot_residuals_vs_leverage() const {
    auto p = plot::RPlot::create();
    p.points(leverage_, standardized_residuals_, "points");

    std::vector<double> flagged_leverage;
    std::vector<double> flagged_residuals;
    const double         threshold = (leverage_.empty()) ? 0.0 : 4.0 / static_cast<double>(leverage_.size());
    for (std::size_t i = 0; i < leverage_.size(); ++i) {
        if (cooks_distance_[i] > threshold) {
            flagged_leverage.push_back(leverage_[i]);
            flagged_residuals.push_back(standardized_residuals_[i]);
        }
    }
    if (!flagged_leverage.empty())
        p.points(flagged_leverage, flagged_residuals, "high Cook's distance", {220, 38, 38}, 6.0);

    p.title("Residuals vs Leverage").x_label("Leverage").y_label("Standardized residuals");
    return p;
}

void LM::save_diagnostic_plots(const std::string& path_prefix) const {
    plot_residuals_vs_fitted().save(path_prefix + "_residuals_vs_fitted.svg");
    plot_normal_qq().save(path_prefix + "_normal_qq.svg");
    plot_scale_location().save(path_prefix + "_scale_location.svg");
    plot_residuals_vs_leverage().save(path_prefix + "_residuals_vs_leverage.svg");
}

} // namespace datamunge::stats
