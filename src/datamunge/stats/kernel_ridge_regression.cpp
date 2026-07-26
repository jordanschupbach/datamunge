#include <datamunge/stats/kernel_ridge_regression.hpp>

#include <datamunge/linalg/eigen.hpp>

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

std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

// A whole leave-one-out evaluation of ridge in the eigenbasis of K, for one lambda: returns the
// LOO mean squared error and (optionally) the dual coefficients, LOO fitted values, and effective
// degrees of freedom. b = U^T yc is precomputed since it does not depend on lambda.
struct LooResult {
    double loo_mse{0.0};
    double effective_dof{0.0};
    std::vector<double> alpha;
    std::vector<double> fitted; // leave-one-out, centered scale (add y_mean for original)
};

LooResult ridge_loo(const linalg::DenseMatrix<double>& U, const std::vector<double>& D,
                    const std::vector<double>& b, const std::vector<double>& yc, double lambda, double y_mean,
                    bool want_detail) {
    const std::size_t n = yc.size();
    LooResult out;
    // coefficients c_k = (D_k/(D_k+lambda)) b_k for the fit, a_k = b_k/(D_k+lambda) for alpha.
    std::vector<double> shrink(n), acoef(n);
    for (std::size_t k = 0; k < n; ++k) {
        const double denom = std::max(D[k] + lambda, 1e-12);
        shrink[k] = D[k] / denom;
        acoef[k] = b[k] / denom;
        out.effective_dof += D[k] / denom;
    }
    double sse = 0.0;
    if (want_detail) {
        out.alpha.assign(n, 0.0);
        out.fitted.assign(n, 0.0);
    }
    for (std::size_t i = 0; i < n; ++i) {
        double yhat = 0.0, hii = 0.0, alpha_i = 0.0;
        for (std::size_t k = 0; k < n; ++k) {
            const double uik = U(i, k);
            yhat += uik * shrink[k] * b[k];
            hii += uik * uik * shrink[k];
            if (want_detail) alpha_i += uik * acoef[k];
        }
        const double one_minus_h = std::max(1.0 - hii, 1e-9);
        const double loo_resid = (yc[i] - yhat) / one_minus_h;
        sse += loo_resid * loo_resid;
        if (want_detail) {
            out.alpha[i] = alpha_i;
            out.fitted[i] = y_mean + (yc[i] - loo_resid);
        }
    }
    out.loo_mse = sse / static_cast<double>(n);
    return out;
}

} // namespace

KernelRidgeRegression::KernelRidgeRegression(const dstruct::DataFrame& data, const std::string& formula,
                                             KernelRidgeRegressionOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

double KernelRidgeRegression::kernel_scalar(const std::vector<double>& a, const std::vector<double>& b,
                                            double gamma) const {
    if (options_.kernel == "linear") {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        return dot;
    }
    if (options_.kernel == "polynomial") {
        double dot = 0.0;
        for (std::size_t j = 0; j < a.size(); ++j) dot += a[j] * b[j];
        return std::pow(gamma * dot + options_.coef0, options_.degree);
    }
    // rbf
    double d2 = 0.0;
    for (std::size_t j = 0; j < a.size(); ++j) {
        const double d = a[j] - b[j];
        d2 += d * d;
    }
    return std::exp(-gamma * d2);
}

std::vector<double> KernelRidgeRegression::kernel_vector(const std::vector<double>& query_std) const {
    const std::size_t n = training_X_std_.rows();
    std::vector<double> k(n, 0.0);
    std::vector<double> row(training_X_std_.cols());
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t c = 0; c < training_X_std_.cols(); ++c) row[c] = training_X_std_(j, c);
        k[j] = kernel_scalar(query_std, row, gamma_used_);
    }
    return k;
}

void KernelRidgeRegression::fit(const dstruct::DataFrame& data) {
    if (options_.kernel != "rbf" && options_.kernel != "linear" && options_.kernel != "polynomial")
        throw std::invalid_argument("KernelRidgeRegression: kernel must be 'rbf', 'linear', or 'polynomial'");
    if (options_.n_lambda_grid == 0) throw std::invalid_argument("KernelRidgeRegression: n_lambda_grid must be >= 1");
    if (options_.lambda == 0.0 && !(options_.kernel == "linear"))
        ; // lambda == 0 is legal (interpolation); no error.

    design_       = formula_.resolve(data, ResponseKind::Numeric, /*force_no_intercept=*/true);
    const auto dm = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n = dm.X.rows();
    const std::size_t p = dm.X.cols();
    if (p == 0) throw std::invalid_argument("KernelRidgeRegression: at least one predictor is required");
    if (n < 2) throw std::invalid_argument("KernelRidgeRegression: fewer than 2 complete observations");

    predictor_names_ = design_.coefficient_names;
    observations_     = n;
    training_y_       = dm.y;

    // Standardize predictors (population sd, matching the other kernel methods); record raw range.
    feature_min_.assign(p, std::numeric_limits<double>::infinity());
    feature_max_.assign(p, -std::numeric_limits<double>::infinity());
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            feature_min_[j] = std::min(feature_min_[j], dm.X(i, j));
            feature_max_[j] = std::max(feature_max_[j], dm.X(i, j));
        }
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
            const double sd = std::sqrt(ss / static_cast<double>(n));
            feature_scale_[j] = (sd < 1e-12) ? 1.0 : sd;
        }
    }
    training_X_std_ = linalg::DenseMatrix<double>(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j)
            training_X_std_(i, j) = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    // Center the response; the RKHS penalty does not shrink the intercept.
    y_mean_ = std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(n);
    std::vector<double> yc(n);
    for (std::size_t i = 0; i < n; ++i) yc[i] = training_y_[i] - y_mean_;

    // Candidate gammas: fixed if given, else a log grid around the median-squared-distance heuristic.
    std::vector<double> gamma_candidates;
    gamma_was_selected_ = false;
    if (options_.kernel == "linear") {
        gamma_candidates = {1.0};
    } else if (options_.gamma > 0.0) {
        gamma_candidates = {options_.gamma};
    } else {
        std::vector<double> sqd;
        sqd.reserve(n * (n - 1) / 2);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) {
                double d2 = 0.0;
                for (std::size_t c = 0; c < p; ++c) {
                    const double d = training_X_std_(i, c) - training_X_std_(j, c);
                    d2 += d * d;
                }
                sqd.push_back(d2);
            }
        std::sort(sqd.begin(), sqd.end());
        const double median_d2 = sqd.empty() ? 1.0 : std::max(sqd[sqd.size() / 2], 1e-9);
        const double g0 = 1.0 / median_d2;
        const std::size_t ng = std::max<std::size_t>(options_.n_gamma_grid, 1);
        gamma_candidates.assign(ng, g0);
        gamma_was_selected_ = ng > 1;
        for (std::size_t k = 0; k < ng; ++k) {
            const double t = (ng == 1) ? 0.0 : static_cast<double>(k) / static_cast<double>(ng - 1);
            gamma_candidates[k] = g0 * std::exp(std::log(0.03) + t * (std::log(30.0) - std::log(0.03)));
        }
    }

    // Candidate lambdas: fixed if given (>= 0), else an absolute log grid.
    std::vector<double> lambda_candidates;
    if (options_.lambda >= 0.0) {
        lambda_candidates = {options_.lambda};
        lambda_was_selected_ = false;
    } else {
        lambda_was_selected_ = true;
        const std::size_t nl = options_.n_lambda_grid;
        lambda_candidates.assign(nl, 0.0);
        for (std::size_t k = 0; k < nl; ++k) {
            const double t = (nl == 1) ? 0.0 : static_cast<double>(k) / static_cast<double>(nl - 1);
            lambda_candidates[k] = std::exp(std::log(1e-4) + t * (std::log(1e3) - std::log(1e-4)));
        }
    }

    // Search: for each gamma build & eigendecompose K once, then sweep lambda cheaply in that basis.
    double best_mse = std::numeric_limits<double>::infinity();
    double best_gamma = gamma_candidates.front(), best_lambda = lambda_candidates.front();
    std::vector<double> best_lambda_curve;
    for (double g : gamma_candidates) {
        linalg::DenseMatrix<double> K(n, n, 0.0);
        std::vector<double> ai(p), aj(p);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t c = 0; c < p; ++c) ai[c] = training_X_std_(i, c);
            for (std::size_t j = i; j < n; ++j) {
                for (std::size_t c = 0; c < p; ++c) aj[c] = training_X_std_(j, c);
                const double v = kernel_scalar(ai, aj, g);
                K(i, j) = v;
                K(j, i) = v;
            }
        }
        auto eig = linalg::jacobi_eigen(K);
        std::vector<double> D = eig.eigenvalues;
        for (double& d : D) d = std::max(d, 0.0);
        std::vector<double> b(n, 0.0);
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t i = 0; i < n; ++i) b[k] += eig.eigenvectors(i, k) * yc[i];

        std::vector<double> curve(lambda_candidates.size(), 0.0);
        for (std::size_t li = 0; li < lambda_candidates.size(); ++li) {
            const double mse = ridge_loo(eig.eigenvectors, D, b, yc, lambda_candidates[li], y_mean_, false).loo_mse;
            curve[li] = mse;
            if (mse < best_mse) {
                best_mse = mse;
                best_gamma = g;
                best_lambda = lambda_candidates[li];
            }
        }
        if (g == best_gamma) best_lambda_curve = curve;
    }
    gamma_used_ = best_gamma;
    lambda_used_ = best_lambda;
    if (lambda_was_selected_) {
        lambda_grid_ = lambda_candidates;
        cv_mse_ = best_lambda_curve;
    }

    // Final fit at the chosen (gamma, lambda): recompute K's eigenbasis and read off everything.
    linalg::DenseMatrix<double> K(n, n, 0.0);
    std::vector<double> ai(p), aj(p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < p; ++c) ai[c] = training_X_std_(i, c);
        for (std::size_t j = i; j < n; ++j) {
            for (std::size_t c = 0; c < p; ++c) aj[c] = training_X_std_(j, c);
            const double v = kernel_scalar(ai, aj, gamma_used_);
            K(i, j) = v;
            K(j, i) = v;
        }
    }
    auto eig = linalg::jacobi_eigen(K);
    std::vector<double> D = eig.eigenvalues;
    for (double& d : D) d = std::max(d, 0.0);
    std::vector<double> b(n, 0.0);
    for (std::size_t k = 0; k < n; ++k)
        for (std::size_t i = 0; i < n; ++i) b[k] += eig.eigenvectors(i, k) * yc[i];

    const LooResult res = ridge_loo(eig.eigenvectors, D, b, yc, lambda_used_, y_mean_, true);
    alpha_ = res.alpha;
    fitted_ = res.fitted;
    effective_dof_ = res.effective_dof;
}

double KernelRidgeRegression::r_squared() const {
    const std::size_t n = training_y_.size();
    const double ybar = std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(n);
    double ss_res = 0.0, ss_tot = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        ss_res += (training_y_[i] - fitted_[i]) * (training_y_[i] - fitted_[i]);
        ss_tot += (training_y_[i] - ybar) * (training_y_[i] - ybar);
    }
    return (ss_tot > 0.0) ? 1.0 - ss_res / ss_tot : 0.0;
}

double KernelRidgeRegression::rmse() const {
    const std::size_t n = training_y_.size();
    double ss_res = 0.0;
    for (std::size_t i = 0; i < n; ++i) ss_res += (training_y_[i] - fitted_[i]) * (training_y_[i] - fitted_[i]);
    return std::sqrt(ss_res / static_cast<double>(n));
}

std::vector<double> KernelRidgeRegression::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "KernelRidgeRegression::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    std::vector<double> result(newdata.nrows(), std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        std::vector<double> query(dm.X.cols());
        for (std::size_t j = 0; j < dm.X.cols(); ++j) query[j] = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];
        const auto k = kernel_vector(query);
        double pred = y_mean_;
        for (std::size_t j = 0; j < k.size(); ++j) pred += alpha_[j] * k[j];
        result[dm.used_row_indices[i]] = pred;
    }
    return result;
}

std::string KernelRidgeRegression::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void KernelRidgeRegression::print_summary() const { print_summary(std::cout); }

void KernelRidgeRegression::print_summary(std::ostream& os) const {
    os << "Kernel Ridge Regression\n";
    os << "Formula: " << formula_.text() << "\n";
    os << "Kernel: " << options_.kernel << ", observations: " << observations_ << "\n";
    os << std::fixed << std::setprecision(4);
    os << "lambda: " << lambda_used_ << (lambda_was_selected_ ? " (LOO-CV selected)" : " (fixed)");
    if (options_.kernel != "linear")
        os << ", gamma: " << gamma_used_ << (gamma_was_selected_ ? " (LOO-CV selected)" : " (fixed)");
    os << "\n";
    os << "Effective degrees of freedom: " << effective_dof_ << " (of " << observations_ << ")\n";
    os << "Leave-one-out R^2: " << format_stat(r_squared()) << ", RMSE: " << format_stat(rmse()) << "\n";
}

plot::RPlot KernelRidgeRegression::plot_fit(const dstruct::DataFrame& data, std::size_t grid_resolution) const {
    if (predictor_names_.size() != 1)
        throw std::invalid_argument("KernelRidgeRegression::plot_fit: only supported for a single-predictor model");
    const auto& x_feature = predictor_names_[0];
    double x_min = feature_min_[0], x_max = feature_max_[0];
    const double pad = std::max(1e-6, (x_max - x_min) * 0.05);
    x_min -= pad;
    x_max += pad;

    std::vector<double> grid_x(grid_resolution);
    for (std::size_t k = 0; k < grid_resolution; ++k)
        grid_x[k] = x_min + (x_max - x_min) * static_cast<double>(k) / static_cast<double>(grid_resolution - 1);
    dstruct::DataFrame grid;
    grid.add_column(x_feature, grid_x);
    const auto grid_y = predict(grid);

    auto plot = plot::RPlot::create();
    std::vector<double> actual_x, actual_y;
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        const auto x = data.optional_double_at(x_feature, i);
        const auto y = data.optional_double_at(design_.response_name, i);
        if (!x || !y) continue;
        actual_x.push_back(*x);
        actual_y.push_back(*y);
    }
    plot.points(actual_x, actual_y, "observed", {156, 163, 175}, 4.0);
    plot.line(grid_x, grid_y, "KRR fit", {220, 38, 38}, 2.5);
    plot.title("Kernel Ridge Regression Fit").x_label(x_feature).y_label(design_.response_name);
    return plot;
}

plot::RPlot KernelRidgeRegression::plot_predicted_vs_actual() const {
    auto plot = plot::RPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual (leave-one-out)").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::RPlot KernelRidgeRegression::plot_residuals_vs_fitted() const {
    std::vector<double> residuals(training_y_.size());
    for (std::size_t i = 0; i < training_y_.size(); ++i) residuals[i] = training_y_[i] - fitted_[i];
    auto plot = plot::RPlot::create();
    plot.points(fitted_, residuals, "residuals");
    plot.title("Residuals vs Fitted (leave-one-out)").x_label("Fitted values").y_label("Residuals");
    return plot;
}

plot::RPlot KernelRidgeRegression::plot_cv_curve() const {
    if (!lambda_was_selected_)
        throw std::invalid_argument("KernelRidgeRegression::plot_cv_curve: no CV curve for a fixed-lambda fit");
    const auto best_it = std::min_element(cv_mse_.begin(), cv_mse_.end());
    const auto best_k  = static_cast<std::size_t>(best_it - cv_mse_.begin());
    auto plot = plot::RPlot::create();
    plot.line(lambda_grid_, cv_mse_, "LOO CV mean squared error");
    plot.points({lambda_grid_[best_k]}, {cv_mse_[best_k]}, "selected lambda", {220, 38, 38}, 8.0);
    plot.title("Regularization Cross-Validation Curve").x_label("lambda").y_label("LOO mean squared error");
    return plot;
}

} // namespace datamunge::stats
