/// @file lm.cpp
/// @brief Fitting, inference, ANOVA, prediction and diagnostic plots for stats::LM.

#include <datamunge/stats/lm.hpp>

#include <datamunge/stats/lowess.hpp>

#include <datamunge/linalg/regression.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Helpers private to this file.
namespace {

/// @brief Scatters per-fitted-row values back to all @p total_rows rows, NaN for dropped rows.
/// @param used_rows  Data row of each compact entry.
/// @param total_rows  Rows in the data frame.
/// @param compact  One value per used row.
/// @return @p total_rows values.
std::vector<double> expand_to_full(const std::vector<std::size_t>& used_rows, std::size_t total_rows,
                                   const std::vector<double>& compact) {
    std::vector<double> out(total_rows, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < used_rows.size(); ++i) out[used_rows[i]] = compact[i];
    return out;
}

/// @brief R's default (type 7) sample quantile of sorted data.
/// @param sorted  Nonempty values in increasing order.
/// @param p  Probability in [0, 1].
/// @return The quantile.
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
/// @brief Formats a statistic in fixed notation, or scientific for very large or small magnitudes.
/// @param v  Value.
/// @param precision  Digits after the point in fixed notation.
/// @return The text.
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

/// @brief Formats a p-value with 4 decimals, or scientific below 1e-4.
/// @param p  The p-value.
/// @return The text.
std::string format_pvalue(double p) {
    std::ostringstream oss;
    if (p < 1e-4) {
        oss << std::scientific << std::setprecision(3) << p;
    } else {
        oss << std::fixed << std::setprecision(4) << p;
    }
    return oss.str();
}

/// @brief R's significance code for a p-value: "***", "**", "*", "." or "".
/// @param p  The p-value.
/// @return The code.
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

    fitted_rows_   = raw_design.used_row_indices;
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

    // R's summary.lm: TSS is the weighted spread about the weighted mean of the raw y when the
    // model has an intercept, and about 0 when it has none. linear_regression() always centres
    // on the mean of the transformed response, so recompute R^2, adjusted R^2 and the ESS here.
    double tss = 0.0, rss = 0.0, sum_w = 0.0, sum_wy = 0.0;
    for (std::size_t i = 0; i < raw_design.y.size(); ++i) {
        const double w = weights_.empty() ? 1.0 : weights_[i];
        sum_w += w;
        sum_wy += w * raw_design.y[i];
    }
    const double centre = design_.has_intercept ? sum_wy / sum_w : 0.0;
    for (std::size_t i = 0; i < raw_design.y.size(); ++i) {
        const double w = weights_.empty() ? 1.0 : weights_[i];
        const double d = raw_design.y[i] - centre;
        tss += w * d * d;
        rss += w * residuals_[i] * residuals_[i];
    }
    const double explained_sum_of_squares = tss - rss;
    const double intercept_df             = design_.has_intercept ? 1.0 : 0.0;
    r_squared_          = (tss == 0.0) ? (rss == 0.0 ? 1.0 : 0.0) : (1.0 - rss / tss);
    adjusted_r_squared_ = (degrees_of_freedom_ > 0 && tss != 0.0)
                              ? 1.0 - (1.0 - r_squared_) * ((static_cast<double>(observations_) - intercept_df) /
                                                            static_cast<double>(degrees_of_freedom_))
                              : r_squared_;

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

double LM::log_likelihood() const {
    const double n       = static_cast<double>(observations_);
    double       sum_log = 0.0;
    for (const double w : weights_) sum_log += std::log(w);
    return 0.5 * (sum_log - n * (std::log(2.0 * std::numbers::pi) + 1.0 - std::log(n) + std::log(rss())));
}

double LM::aic() const { return -2.0 * log_likelihood() + 2.0 * static_cast<double>(rank() + 1); }

double LM::bic() const {
    return -2.0 * log_likelihood() + std::log(static_cast<double>(observations_)) * static_cast<double>(rank() + 1);
}

dstruct::DataFrame LM::coefficient_frame(double level) const {
    const auto          ci = confidence_intervals(level);
    std::vector<double> lower(ci.size()), upper(ci.size());
    for (std::size_t j = 0; j < ci.size(); ++j) {
        lower[j] = ci[j].lower;
        upper[j] = ci[j].upper;
    }
    dstruct::DataFrame out;
    out.add_column("term", design_.coefficient_names);
    out.add_column("estimate", coefficients_);
    out.add_column("std_error", standard_errors_);
    out.add_column("t_value", t_values_);
    out.add_column("p_value", p_values_);
    out.add_column("lower", lower);
    out.add_column("upper", upper);
    return out;
}

dstruct::DataFrame LM::anova_frame() const {
    std::vector<std::string>           terms;
    std::vector<double>                df, sum_sq, mean_sq;
    std::vector<std::optional<double>> f_value, p_value;
    for (const auto& row : anova()) {
        const bool residual = row.term == "Residuals";
        terms.push_back(row.term);
        df.push_back(static_cast<double>(row.degrees_of_freedom));
        sum_sq.push_back(row.sum_sq);
        mean_sq.push_back(row.mean_sq);
        f_value.push_back(residual ? std::nullopt : std::optional<double>(row.f_value));
        p_value.push_back(residual ? std::nullopt : std::optional<double>(row.p_value));
    }
    dstruct::DataFrame out;
    out.add_column("term", terms);
    out.add_column("df", df);
    out.add_column("sum_sq", sum_sq);
    out.add_column("mean_sq", mean_sq);
    out.add_column("f_value", f_value);
    out.add_column("p_value", p_value);
    return out;
}

namespace {

/// @brief Colour of the diagnostic plots' points (R's default black, softened).
constexpr graphics::RGB kPointColor{55, 65, 81};
/// @brief Colour of the lowess smoothers (R's red).
constexpr graphics::RGB kSmoothColor{220, 38, 38};
/// @brief Colour of reference lines and Cook's contours.
constexpr graphics::RGB kGuideColor{148, 163, 184};

/// @brief Padded [lo, hi] range of @p values, widened by @p pad of the span on each side.
/// @param values  Nonempty values.
/// @param pad  Fraction of the span to add on each side.
/// @return The padded range; a zero span becomes +/- 1.
std::pair<double, double> padded_range(const std::vector<double>& values, double pad) {
    const auto [lo_it, hi_it] = std::minmax_element(values.begin(), values.end());
    double     lo = *lo_it, hi = *hi_it;
    const double span = hi - lo;
    if (span <= 0.0) return {lo - 1.0, hi + 1.0};
    return {lo - pad * span, hi + pad * span};
}

/// @brief Indices of the @p k largest values of @p score (fewer if @p score is shorter).
/// @param score  Values to rank.
/// @param k  How many to return.
/// @return Indices, largest score first.
std::vector<std::size_t> top_k(const std::vector<double>& score, std::size_t k) {
    std::vector<std::size_t> idx(score.size());
    std::iota(idx.begin(), idx.end(), std::size_t{0});
    k = std::min(k, idx.size());
    std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                      [&](std::size_t a, std::size_t b) { return score[a] > score[b]; });
    idx.resize(k);
    return idx;
}

/// @brief Labels points @p ids of (x, y) with their 1-based data row numbers, placed just right of
///        each point, or just left of it when that would leave the x range.
/// @param plot  Plot to annotate.
/// @param x  Point x coordinates.
/// @param y  Point y coordinates.
/// @param ids  Indices to label.
/// @param x_range  The plot's x limits.
/// @param rows  Data row (0-based) of each point.
void label_points(graphics::RPlot& plot, const std::vector<double>& x, const std::vector<double>& y,
                  const std::vector<std::size_t>& ids, std::pair<double, double> x_range,
                  const std::vector<std::size_t>& rows) {
    const double dx = 0.035 * (x_range.second - x_range.first);
    for (const auto i : ids) {
        const bool   left = x[i] + 2.0 * dx > x_range.second;
        const double tx   = left ? x[i] - dx : x[i] + dx;
        plot.text(tx, y[i], std::to_string(rows[i] + 1), {75, 85, 99}, 13.0);
    }
}

/// @brief Adds a lowess smoother of (x, y), R's `panel.smooth` (span 2/3, 3 iterations).
/// @param plot  Plot to draw on.
/// @param x  Predictor values.
/// @param y  Response values.
/// @return The smoothed y values (to include in the axis range).
std::vector<double> add_smoother(graphics::RPlot& plot, const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() < 3) return {};
    auto smooth = lowess(x, y);
    plot.lines(smooth.x, smooth.y, kSmoothColor, 2.0);
    return smooth.y;
}

/// @brief Applies explicit limits so every layer (including text and smoothers) stays inside.
/// @param plot  Plot to limit.
/// @param x_range  x limits.
/// @param y_range  y limits.
void set_limits(graphics::RPlot& plot, std::pair<double, double> x_range, std::pair<double, double> y_range) {
    plot.x_limits(x_range.first, x_range.second).y_limits(y_range.first, y_range.second);
}

} // namespace

graphics::RPlot LM::plot_residuals_vs_fitted() const {
    auto p = graphics::RPlot::create();
    p.abline_h(0.0, kGuideColor, 1.0);
    p.points(fitted_, residuals_, kPointColor, 3.0);
    auto smooth = add_smoother(p, fitted_, residuals_);

    std::vector<double> ys = residuals_;
    ys.insert(ys.end(), smooth.begin(), smooth.end());
    const auto x_range = padded_range(fitted_, 0.06);
    const auto y_range = padded_range(ys, 0.06);
    std::vector<double> abs_res(residuals_.size());
    for (std::size_t i = 0; i < residuals_.size(); ++i) abs_res[i] = std::fabs(residuals_[i]);
    label_points(p, fitted_, residuals_, top_k(abs_res, 3), x_range, fitted_rows_);
    set_limits(p, x_range, y_range);
    p.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return p;
}

graphics::RPlot LM::plot_normal_qq() const {
    const std::size_t        n = standardized_residuals_.size();
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return standardized_residuals_[a] < standardized_residuals_[b]; });
    // R's ppoints(): (i - a) / (n + 1 - 2a) with a = 3/8 for n <= 10, else 1/2.
    const double        a = n <= 10 ? 0.375 : 0.5;
    std::vector<double> theoretical(n), sample(n), x_of(n), y_of(n);
    for (std::size_t i = 0; i < n; ++i) {
        theoretical[i] = random::normal_quantile((static_cast<double>(i + 1) - a) / (static_cast<double>(n) + 1.0 - 2.0 * a));
        sample[i]      = standardized_residuals_[order[i]];
        x_of[order[i]] = theoretical[i];
        y_of[order[i]] = sample[i];
    }

    auto       p       = graphics::RPlot::create();
    const auto x_range = padded_range(theoretical, 0.06);
    const auto y_range = padded_range(sample, 0.06);
    if (n >= 2) {
        // Line through the sample and normal quartiles (R's qqline), clipped to the plot area.
        const double q1 = quantile_type7(sample, 0.25), q3 = quantile_type7(sample, 0.75);
        const double z1 = random::normal_quantile(0.25), z3 = random::normal_quantile(0.75);
        const double slope = (q3 - q1) / (z3 - z1), intercept = q1 - slope * z1;
        double       x0 = x_range.first, x1 = x_range.second;
        if (slope != 0.0) {
            const double xa = (y_range.first - intercept) / slope, xb = (y_range.second - intercept) / slope;
            x0 = std::max(x0, std::min(xa, xb));
            x1 = std::min(x1, std::max(xa, xb));
        }
        if (x1 > x0) p.lines({x0, x1}, {intercept + slope * x0, intercept + slope * x1}, kGuideColor, 1.5);
    }
    p.points(theoretical, sample, kPointColor, 3.0);
    std::vector<double> abs_res(n);
    for (std::size_t i = 0; i < n; ++i) abs_res[i] = std::fabs(standardized_residuals_[i]);
    label_points(p, x_of, y_of, top_k(abs_res, 3), x_range, fitted_rows_);
    set_limits(p, x_range, y_range);
    p.title("Normal Q-Q").x_label("Theoretical quantiles").y_label("Standardized residuals");
    return p;
}

graphics::RPlot LM::plot_scale_location() const {
    std::vector<double> sqrt_abs_std(standardized_residuals_.size());
    for (std::size_t i = 0; i < standardized_residuals_.size(); ++i)
        sqrt_abs_std[i] = std::sqrt(std::fabs(standardized_residuals_[i]));

    auto p = graphics::RPlot::create();
    p.points(fitted_, sqrt_abs_std, kPointColor, 3.0);
    auto smooth = add_smoother(p, fitted_, sqrt_abs_std);

    std::vector<double> ys = sqrt_abs_std;
    ys.insert(ys.end(), smooth.begin(), smooth.end());
    ys.push_back(0.0);
    const auto x_range = padded_range(fitted_, 0.06);
    const auto y_range = padded_range(ys, 0.04);
    label_points(p, fitted_, sqrt_abs_std, top_k(sqrt_abs_std, 3), x_range, fitted_rows_);
    set_limits(p, x_range, y_range);
    p.title("Scale-Location").x_label("Fitted values").y_label("sqrt(|standardized residuals|)");
    return p;
}

graphics::RPlot LM::plot_residuals_vs_leverage() const {
    auto       p       = graphics::RPlot::create();
    const double p_coef = static_cast<double>(coefficients_.size());

    std::vector<double> xs = leverage_;
    xs.push_back(0.0);
    const auto x_range = padded_range(xs, 0.06);
    const auto y_range = padded_range(standardized_residuals_, 0.08);
    p.abline_h(0.0, kGuideColor, 1.0);

    // Cook's distance D = r^2 h / (p (1 - h)), so the contour is r = +/- sqrt(D p (1 - h) / h).
    // |r| falls as h grows, so the part inside the y limits starts where |r| = Y.
    for (const double level : {0.5, 1.0}) {
        for (const double sign : {1.0, -1.0}) {
            const double y_edge = sign > 0 ? y_range.second : -y_range.first;
            if (y_edge <= 0.0) continue;
            const double c      = level * p_coef;
            const double h_from = std::max(c / (y_edge * y_edge + c), 1e-6);
            const double h_to   = std::min(x_range.second, 1.0 - 1e-6);
            if (h_from >= h_to) continue;
            std::vector<double> hx, ry;
            for (int k = 0; k <= 60; ++k) {
                const double h = h_from + (h_to - h_from) * k / 60.0;
                hx.push_back(h);
                ry.push_back(std::clamp(sign * std::sqrt(c * (1.0 - h) / h), y_range.first, y_range.second));
            }
            p.lines(hx, ry, kGuideColor, 1.5);
            const std::string label = level == 0.5 ? "0.5" : "1";
            const double span    = y_range.second - y_range.first;
            // The D = 1 curve lies outside the D = 0.5 curve: label each on its outer or inner side.
            const double offset  = (level == 1.0 ? 0.045 : -0.02) * span;
            const double label_y = std::clamp(ry.back() + sign * offset, y_range.first + 0.04 * span,
                                              y_range.second - 0.04 * span);
            p.text(h_to - 0.04 * (x_range.second - x_range.first), label_y, label, {100, 116, 139}, 12.0);
        }
    }

    p.points(leverage_, standardized_residuals_, kPointColor, 3.0);
    label_points(p, leverage_, standardized_residuals_, top_k(cooks_distance_, 3), x_range, fitted_rows_);
    set_limits(p, x_range, y_range);
    p.title("Residuals vs Leverage").x_label("Leverage").y_label("Standardized residuals");
    return p;
}

graphics::RLayout LM::plot_diagnostics() const {
    auto fig = graphics::RLayout::create(2, 2);
    fig.add(plot_residuals_vs_fitted())
        .add(plot_normal_qq())
        .add(plot_scale_location())
        .add(plot_residuals_vs_leverage())
        .size(1100, 860)
        .fit_panels();
    return fig;
}

void LM::save_diagnostic_plots(const std::string& path_prefix) const {
    plot_residuals_vs_fitted().save(path_prefix + "_residuals_vs_fitted.svg");
    plot_normal_qq().save(path_prefix + "_normal_qq.svg");
    plot_scale_location().save(path_prefix + "_scale_location.svg");
    plot_residuals_vs_leverage().save(path_prefix + "_residuals_vs_leverage.svg");
}

NestedAnova anova(const LM& reduced, const LM& full) {
    if (reduced.observations() != full.observations())
        throw std::invalid_argument("anova: the models were fit to different numbers of rows");
    if (full.degrees_of_freedom() >= reduced.degrees_of_freedom())
        throw std::invalid_argument("anova: the full model must have fewer residual degrees of freedom");
    NestedAnova out;
    out.reduced_df  = reduced.degrees_of_freedom();
    out.full_df     = full.degrees_of_freedom();
    out.reduced_rss = reduced.rss();
    out.full_rss    = full.rss();
    out.df          = out.reduced_df - out.full_df;
    out.sum_sq      = out.reduced_rss - out.full_rss;
    const double d0 = static_cast<double>(out.df), d1 = static_cast<double>(out.full_df);
    out.f_value     = (out.sum_sq / d0) / (out.full_rss / d1);
    out.p_value     = 1.0 - random::f_cdf(out.f_value, d0, d1);
    return out;
}

} // namespace datamunge::stats
