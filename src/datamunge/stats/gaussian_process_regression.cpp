#include <datamunge/stats/gaussian_process_regression.hpp>

#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

constexpr double kJitter = 1e-8; // numerical-stability nugget added to every diagonal, regardless of noise_ratio
constexpr double kPi     = 3.14159265358979323846;

std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

std::vector<double> geometric_grid(double lo, double hi, std::size_t count) {
    std::vector<double> grid(count);
    for (std::size_t k = 0; k < count; ++k) {
        const double t = (count == 1) ? 0.0 : static_cast<double>(k) / static_cast<double>(count - 1);
        grid[k]        = std::exp(std::log(lo) + t * (std::log(hi) - std::log(lo)));
    }
    return grid;
}

} // namespace

GaussianProcessRegression::GaussianProcessRegression(const dstruct::DataFrame& data, const std::string& formula,
                                                      GaussianProcessRegressionOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

void GaussianProcessRegression::fit(const dstruct::DataFrame& data) {
    if (options_.length_scale == 0.0)
        throw std::invalid_argument("GaussianProcessRegression: length_scale must not be zero");
    if (options_.n_length_scale_grid == 0 || options_.n_noise_grid == 0)
        throw std::invalid_argument("GaussianProcessRegression: grid sizes must be at least 1");

    design_       = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    const auto dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = dm.X.rows();
    const std::size_t p = dm.X.cols();
    if (p == 0) throw std::invalid_argument("GaussianProcessRegression: at least one predictor is required");

    predictor_names_ = design_.coefficient_names;
    observations_     = n;
    training_y_       = dm.y;

    feature_mean_.assign(p, 0.0);
    feature_scale_.assign(p, 1.0);
    if (options_.standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += dm.X(i, j);
            feature_mean_[j] = sum / static_cast<double>(n);
        }
        for (std::size_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = dm.X(i, j) - feature_mean_[j];
                ss += d * d;
            }
            const double scale = std::sqrt(ss / static_cast<double>(n));
            feature_scale_[j]  = (scale < 1e-12) ? 1.0 : scale;
        }
    }
    training_X_std_ = linalg::DenseMatrix<double>(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) training_X_std_(i, j) = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    y_mean_ = std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(n);
    double y_ss = 0.0;
    for (const double y : training_y_) y_ss += (y - y_mean_) * (y - y_mean_);
    y_scale_ = std::sqrt(y_ss / static_cast<double>(n));
    if (y_scale_ < 1e-12) y_scale_ = 1.0;
    training_y_std_.resize(n);
    for (std::size_t i = 0; i < n; ++i) training_y_std_[i] = (training_y_[i] - y_mean_) / y_scale_;

    linalg::DenseMatrix<double> sq_dist(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            double d2 = 0.0;
            for (std::size_t k = 0; k < p; ++k) {
                const double diff = training_X_std_(i, k) - training_X_std_(j, k);
                d2 += diff * diff;
            }
            sq_dist(i, j) = d2;
            sq_dist(j, i) = d2;
        }
    }

    if (options_.length_scale > 0.0) {
        length_scale_grid_ = {options_.length_scale};
    } else {
        length_scale_grid_ =
            geometric_grid(0.1 * std::sqrt(static_cast<double>(p)), 3.0 * std::sqrt(static_cast<double>(p)),
                           options_.n_length_scale_grid);
    }
    if (options_.noise_ratio >= 0.0) {
        noise_ratio_grid_ = {options_.noise_ratio};
    } else {
        noise_ratio_grid_ = geometric_grid(1e-4, 1.0, options_.n_noise_grid);
    }

    length_scale_profile_ll_.assign(length_scale_grid_.size(), -std::numeric_limits<double>::infinity());

    double best_ll = -std::numeric_limits<double>::infinity();
    double best_l = length_scale_grid_.front(), best_eta = noise_ratio_grid_.front(), best_sf2 = 1.0;

    linalg::DenseMatrix<double> C(n, n, 0.0);
    for (std::size_t li = 0; li < length_scale_grid_.size(); ++li) {
        const double l           = length_scale_grid_[li];
        const double inv_two_l2  = 1.0 / (2.0 * l * l);
        double       best_for_l  = -std::numeric_limits<double>::infinity();

        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) C(i, j) = std::exp(-sq_dist(i, j) * inv_two_l2);

        for (const double eta : noise_ratio_grid_) {
            for (std::size_t i = 0; i < n; ++i) C(i, i) = 1.0 + eta + kJitter;

            const auto chol = linalg::cholesky(C);
            if (!chol.ok) continue;
            const auto   alpha_c = chol.solve(training_y_std_);
            const double quad    = std::inner_product(training_y_std_.begin(), training_y_std_.end(),
                                                       alpha_c.begin(), 0.0);
            double       sf2     = quad / static_cast<double>(n);
            if (sf2 < 1e-12) sf2 = 1e-12;

            double log_det_C = 0.0;
            for (std::size_t i = 0; i < n; ++i) log_det_C += 2.0 * std::log(chol.L(i, i));

            const double ll = -0.5 * static_cast<double>(n) * std::log(sf2) - 0.5 * log_det_C
                             - 0.5 * static_cast<double>(n) * (1.0 + std::log(2.0 * kPi));

            if (ll > best_for_l) best_for_l = ll;
            if (ll > best_ll) {
                best_ll  = ll;
                best_l   = l;
                best_eta = eta;
                best_sf2 = sf2;
            }
            for (std::size_t i = 0; i < n; ++i) C(i, i) = 1.0; // restore off-diagonal-only matrix for reuse
        }
        length_scale_profile_ll_[li] = best_for_l;
    }

    length_scale_used_           = best_l;
    noise_ratio_used_            = best_eta;
    signal_variance_used_        = best_sf2;
    log_marginal_likelihood_used_ = best_ll;

    linalg::DenseMatrix<double> K(n, n, 0.0);
    const double inv_two_l2 = 1.0 / (2.0 * length_scale_used_ * length_scale_used_);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            K(i, j) = signal_variance_used_ * std::exp(-sq_dist(i, j) * inv_two_l2);
    for (std::size_t i = 0; i < n; ++i) K(i, i) += noise_variance() + kJitter;

    chol_  = linalg::cholesky(K);
    if (!chol_->ok) throw std::runtime_error("GaussianProcessRegression: final kernel matrix was not SPD");
    alpha_ = chol_->solve(training_y_std_);

    const auto Kinv = chol_->solve(linalg::DenseMatrix<double>::identity(n));
    fitted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double kinv_ii   = Kinv(i, i);
        const double mu_loo    = training_y_std_[i] - alpha_[i] / kinv_ii;
        fitted_[i]              = mu_loo * y_scale_ + y_mean_;
    }
}

std::vector<double> GaussianProcessRegression::kernel_vector(const std::vector<double>& query_std) const {
    const std::size_t n = training_X_std_.rows();
    const std::size_t p = training_X_std_.cols();
    const double inv_two_l2 = 1.0 / (2.0 * length_scale_used_ * length_scale_used_);
    std::vector<double> k(n);
    for (std::size_t i = 0; i < n; ++i) {
        double d2 = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            const double diff = training_X_std_(i, j) - query_std[j];
            d2 += diff * diff;
        }
        k[i] = signal_variance_used_ * std::exp(-d2 * inv_two_l2);
    }
    return k;
}

double GaussianProcessRegression::r_squared() const {
    const double mean =
        std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    double tss = 0.0, rss = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double dt = training_y_[i] - mean;
        const double dr = training_y_[i] - fitted_[i];
        tss += dt * dt;
        rss += dr * dr;
    }
    return (tss > 0.0) ? (1.0 - rss / tss) : 1.0;
}

double GaussianProcessRegression::rmse() const {
    double sse = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double d = training_y_[i] - fitted_[i];
        sse += d * d;
    }
    return std::sqrt(sse / static_cast<double>(training_y_.size()));
}

std::string GaussianProcessRegression::summary() const {
    std::ostringstream out;
    out << "Call:\ngp(formula = " << formula_.text() << ", kernel = \"rbf\")\n\n";

    out << "Length scale: " << format_stat(length_scale_used_, 4)
        << (length_scale_was_selected() ? "  (selected by log marginal likelihood over "
                                          + std::to_string(length_scale_grid_.size()) + " values)"
                                        : "  (fixed)")
        << "\n";
    out << "Signal variance: " << format_stat(signal_variance_used_, 4) << "  (profiled maximum-likelihood value)\n";
    out << "Noise variance: " << format_stat(noise_variance(), 4)
        << (noise_ratio_was_selected() ? "  (selected by log marginal likelihood over "
                                        + std::to_string(noise_ratio_grid_.size()) + " values)"
                                       : "  (fixed)")
        << "\n";
    out << "Log marginal likelihood: " << format_stat(log_marginal_likelihood_used_, 4) << "\n";

    out << "\nLeave-one-out R-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "Leave-one-out RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";

    return out.str();
}

void GaussianProcessRegression::print_summary(std::ostream& os) const { os << summary(); }

void GaussianProcessRegression::print_summary() const { print_summary(std::cout); }

std::vector<double> GaussianProcessRegression::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).fit;
}

GaussianProcessRegressionPrediction GaussianProcessRegression::predict_detail(const dstruct::DataFrame& newdata,
                                                                              double level) const {
    design_.validate_categorical_levels(newdata, "GaussianProcessRegression::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    GaussianProcessRegressionPrediction result;
    result.fit.assign(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    result.se_fit.assign(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    result.lower.assign(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    result.upper.assign(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());

    const double z = random::normal_quantile(1.0 - (1.0 - level) / 2.0);

    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

        const auto k_star = kernel_vector(query);
        const double mean_std = std::inner_product(k_star.begin(), k_star.end(), alpha_.begin(), 0.0);

        const auto v = chol_->solve(k_star);
        double var_std = signal_variance_used_ - std::inner_product(k_star.begin(), k_star.end(), v.begin(), 0.0);
        if (var_std < 0.0) var_std = 0.0;

        const auto original_row = dm.used_row_indices[i];
        const double mean       = mean_std * y_scale_ + y_mean_;
        const double se         = std::sqrt(var_std) * y_scale_;

        result.fit[original_row]   = mean;
        result.se_fit[original_row] = se;
        result.lower[original_row]  = mean - z * se;
        result.upper[original_row]  = mean + z * se;
    }
    return result;
}

plot::ScatterPlot GaussianProcessRegression::plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution,
                                                       double level) const {
    if (predictor_names_.size() != 1)
        throw std::invalid_argument("GaussianProcessRegression::plot_fit: only supported for a single-predictor "
                                    "model");
    const auto& x_feature = predictor_names_[0];

    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < training_X_std_.rows(); ++i) {
        const double raw = training_X_std_(i, 0) * feature_scale_[0] + feature_mean_[0];
        x_min             = std::min(x_min, raw);
        x_max             = std::max(x_max, raw);
    }
    const double pad = std::max(1e-6, (x_max - x_min) * 0.05);
    x_min -= pad;
    x_max += pad;

    std::vector<double> grid_x(grid_resolution);
    for (std::size_t k = 0; k < grid_resolution; ++k)
        grid_x[k] = x_min + (x_max - x_min) * static_cast<double>(k) / static_cast<double>(grid_resolution - 1);

    dstruct::DataFrame grid;
    grid.add_column(x_feature, grid_x);
    const auto detail = predict_detail(grid, level);

    auto plot = plot::ScatterPlot::create();

    std::vector<double> actual_x, actual_y;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const auto x = data.optional_double_at(x_feature, i);
        const auto y = data.optional_double_at(design_.response_name, i);
        if (!x || !y) continue;
        actual_x.push_back(*x);
        actual_y.push_back(*y);
    }
    plot.points(actual_x, actual_y, "observed", {156, 163, 175}, 4.0);
    plot.line(grid_x, detail.lower, "lower", {252, 165, 165}, 1.5);
    plot.line(grid_x, detail.upper, "upper", {252, 165, 165}, 1.5);
    plot.line(grid_x, detail.fit, "posterior mean", {220, 38, 38}, 2.5);

    plot.title("Gaussian Process Fit").x_label(x_feature).y_label(design_.response_name);
    return plot;
}

plot::ScatterPlot GaussianProcessRegression::plot_predicted_vs_actual() const {
    auto plot = plot::ScatterPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual (leave-one-out)").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::ScatterPlot GaussianProcessRegression::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_[i];

    auto plot = plot::ScatterPlot::create();
    plot.points(fitted_, residuals, "residuals");
    plot.title("Residuals vs Fitted (leave-one-out)").x_label("Fitted values").y_label("Residuals");
    return plot;
}

plot::ScatterPlot GaussianProcessRegression::plot_length_scale_profile() const {
    if (!length_scale_was_selected())
        throw std::invalid_argument("GaussianProcessRegression::plot_length_scale_profile: length_scale was not "
                                    "auto-selected");

    auto plot = plot::ScatterPlot::create();
    plot.line(length_scale_grid_, length_scale_profile_ll_, "profile log marginal likelihood");
    const auto best_it = std::max_element(length_scale_profile_ll_.begin(), length_scale_profile_ll_.end());
    const auto best_k   = static_cast<std::size_t>(best_it - length_scale_profile_ll_.begin());
    plot.points({length_scale_grid_[best_k]}, {length_scale_profile_ll_[best_k]}, "selected length scale",
               {220, 38, 38}, 8.0);
    plot.title("Length Scale Profile").x_label("Length scale").y_label("Profile log marginal likelihood");
    return plot;
}

} // namespace datamunge::stats
