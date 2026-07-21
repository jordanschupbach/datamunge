#include <datamunge/stats/elastic_net.hpp>

#include <datamunge/random/random.hpp>

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

std::string format_stat(double v, int precision = 6) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

std::vector<double> expand_to_full(const std::vector<std::size_t>& used_rows, std::size_t total_rows,
                                   const std::vector<double>& compact) {
    std::vector<double> out(total_rows, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t i = 0; i < used_rows.size(); ++i) out[used_rows[i]] = compact[i];
    return out;
}

// Cyclic coordinate descent for the glmnet-style elastic net objective on
// already-standardized columns (mean 0, (1/n)*sum(x_ij^2) = 1):
//   (1/2n)*||y - X*beta||^2 + lambda*alpha*||beta||_1 + lambda*(1-alpha)/2*||beta||_2^2
// `beta` is both the warm-start input and the returned solution.
std::vector<double> coordinate_descent(const linalg::DenseMatrix<double>& X, const std::vector<double>& y,
                                       double alpha, double lambda, std::vector<double> beta,
                                       std::size_t max_iter, double tol) {
    const std::size_t n = X.rows();
    const std::size_t p = X.cols();
    if (p == 0) return beta;

    std::vector<double> r(n);
    for (std::size_t i = 0; i < n; ++i) {
        double pred = 0.0;
        for (std::size_t j = 0; j < p; ++j) pred += X(i, j) * beta[j];
        r[i] = y[i] - pred;
    }

    for (std::size_t iter = 0; iter < max_iter; ++iter) {
        double max_change = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            const double old_beta = beta[j];
            if (old_beta != 0.0)
                for (std::size_t i = 0; i < n; ++i) r[i] += X(i, j) * old_beta;

            double rho = 0.0;
            for (std::size_t i = 0; i < n; ++i) rho += X(i, j) * r[i];
            rho /= static_cast<double>(n);

            double new_beta;
            if (alpha > 0.0) {
                const double thresh = lambda * alpha;
                double       s      = 0.0;
                if (rho > thresh)
                    s = rho - thresh;
                else if (rho < -thresh)
                    s = rho + thresh;
                new_beta = s / (1.0 + lambda * (1.0 - alpha));
            } else {
                new_beta = rho / (1.0 + lambda);
            }

            beta[j] = new_beta;
            if (new_beta != 0.0)
                for (std::size_t i = 0; i < n; ++i) r[i] -= X(i, j) * new_beta;
            max_change = std::max(max_change, std::fabs(new_beta - old_beta));
        }
        if (max_change < tol) break;
    }
    return beta;
}

} // namespace

ElasticNet::ElasticNet(const dstruct::DataFrame& data, const std::string& formula, ElasticNetOptions options)
    : formula_(formula) {
    options_ = options;
    fit(data);
}

ElasticNet::ElasticNet(const dstruct::DataFrame& data, const std::string& formula, double fixed_alpha,
                       ElasticNetOptions options)
    : formula_(formula) {
    options_       = options;
    options_.alpha = fixed_alpha;
    fit(data);
}

void ElasticNet::fit(const dstruct::DataFrame& data) {
    if (options_.alpha < 0.0 || options_.alpha > 1.0)
        throw std::invalid_argument("ElasticNet: alpha must be in [0, 1]");
    if (options_.n_lambda == 0) throw std::invalid_argument("ElasticNet: n_lambda must be at least 1");
    if (options_.cv_folds < 2) throw std::invalid_argument("ElasticNet: cv_folds must be at least 2");

    design_        = formula_.resolve(data, ResponseKind::Numeric);
    const auto dm  = design_.build_matrix(data, /*require_response=*/true);
    const std::size_t n       = dm.X.rows();
    const bool         has_int = design_.has_intercept;
    const std::size_t p       = has_int ? dm.X.cols() - 1 : dm.X.cols();

    predictor_names_.clear();
    predictor_names_.reserve(p);
    for (std::size_t j = (has_int ? 1 : 0); j < design_.coefficient_names.size(); ++j)
        predictor_names_.push_back(design_.coefficient_names[j]);

    observations_ = n;
    training_y_   = dm.y;

    linalg::DenseMatrix<double> Xp(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) Xp(i, j) = dm.X(i, j + (has_int ? 1 : 0));

    feature_mean_.assign(p, 0.0);
    feature_scale_.assign(p, 1.0);
    if (has_int) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += Xp(i, j);
            feature_mean_[j] = sum / static_cast<double>(n);
        }
    }
    if (options_.standardize) {
        for (std::size_t j = 0; j < p; ++j) {
            double ss = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = Xp(i, j) - feature_mean_[j];
                ss += d * d;
            }
            const double scale = std::sqrt(ss / static_cast<double>(n));
            feature_scale_[j]  = (scale < 1e-12) ? 1.0 : scale;
        }
    }

    linalg::DenseMatrix<double> Xstd(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) Xstd(i, j) = (Xp(i, j) - feature_mean_[j]) / feature_scale_[j];

    y_mean_ = has_int ? (std::accumulate(dm.y.begin(), dm.y.end(), 0.0) / static_cast<double>(n)) : 0.0;
    std::vector<double> y_centered(n);
    for (std::size_t i = 0; i < n; ++i) y_centered[i] = dm.y[i] - y_mean_;

    auto unstandardize = [&](const std::vector<double>& beta_std) {
        std::vector<double> beta_orig(p);
        for (std::size_t j = 0; j < p; ++j) beta_orig[j] = beta_std[j] / feature_scale_[j];
        return beta_orig;
    };

    if (options_.lambda >= 0.0) {
        lambda_was_selected_ = false;
        lambda_path_.clear();
        cv_mse_path_.clear();
        path_coefficients_ = linalg::DenseMatrix<double>();

        std::vector<double> beta(p, 0.0);
        beta         = coordinate_descent(Xstd, y_centered, options_.alpha, options_.lambda, std::move(beta),
                                          options_.max_iter, options_.tol);
        lambda_used_ = options_.lambda;
        coefficients_ = unstandardize(beta);
    } else {
        lambda_was_selected_ = true;

        double lambda_max = 0.0;
        for (std::size_t j = 0; j < p; ++j) {
            double rho = 0.0;
            for (std::size_t i = 0; i < n; ++i) rho += Xstd(i, j) * y_centered[i];
            rho /= static_cast<double>(n);
            lambda_max = std::max(lambda_max, std::fabs(rho));
        }
        const double alpha_for_max = std::max(options_.alpha, 1e-3);
        lambda_max /= alpha_for_max;
        if (lambda_max <= 0.0) lambda_max = 1.0;

        const double lambda_min_ratio = (n < p) ? 0.01 : 0.0001;
        const double lambda_min       = lambda_max * lambda_min_ratio;

        lambda_path_.assign(options_.n_lambda, 0.0);
        for (std::size_t k = 0; k < options_.n_lambda; ++k) {
            const double t = (options_.n_lambda == 1)
                               ? 0.0
                               : static_cast<double>(k) / static_cast<double>(options_.n_lambda - 1);
            lambda_path_[k] = std::exp(std::log(lambda_max) + t * (std::log(lambda_min) - std::log(lambda_max)));
        }

        path_coefficients_ = linalg::DenseMatrix<double>(options_.n_lambda, p, 0.0);
        {
            std::vector<double> beta(p, 0.0);
            for (std::size_t k = 0; k < options_.n_lambda; ++k) {
                beta               = coordinate_descent(Xstd, y_centered, options_.alpha, lambda_path_[k],
                                                        std::move(beta), options_.max_iter, options_.tol);
                const auto orig    = unstandardize(beta);
                for (std::size_t j = 0; j < p; ++j) path_coefficients_(k, j) = orig[j];
            }
        }

        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        random::SplitMix64 rng(options_.seed);
        for (std::size_t i = n; i-- > 1;) {
            const std::size_t j = static_cast<std::size_t>(rng.next_u64() % (i + 1));
            std::swap(order[i], order[j]);
        }
        std::vector<std::size_t> fold_of(n);
        for (std::size_t i = 0; i < n; ++i) fold_of[order[i]] = i % options_.cv_folds;

        std::vector<double>      cv_sq_err(options_.n_lambda, 0.0);
        std::vector<std::size_t> cv_count(options_.n_lambda, 0);

        for (std::size_t f = 0; f < options_.cv_folds; ++f) {
            std::vector<std::size_t> train_rows, test_rows;
            for (std::size_t i = 0; i < n; ++i) (fold_of[i] == f ? test_rows : train_rows).push_back(i);
            if (train_rows.empty() || test_rows.empty()) continue;

            linalg::DenseMatrix<double> Xtrain(train_rows.size(), p, 0.0);
            std::vector<double>          ytrain(train_rows.size());
            for (std::size_t r = 0; r < train_rows.size(); ++r) {
                for (std::size_t j = 0; j < p; ++j) Xtrain(r, j) = Xstd(train_rows[r], j);
                ytrain[r] = y_centered[train_rows[r]];
            }

            std::vector<double> beta_f(p, 0.0);
            for (std::size_t k = 0; k < options_.n_lambda; ++k) {
                beta_f = coordinate_descent(Xtrain, ytrain, options_.alpha, lambda_path_[k], std::move(beta_f),
                                            options_.max_iter, options_.tol);
                for (const auto row : test_rows) {
                    double pred = 0.0;
                    for (std::size_t j = 0; j < p; ++j) pred += Xstd(row, j) * beta_f[j];
                    const double err = y_centered[row] - pred;
                    cv_sq_err[k] += err * err;
                    ++cv_count[k];
                }
            }
        }

        cv_mse_path_.assign(options_.n_lambda, std::numeric_limits<double>::quiet_NaN());
        for (std::size_t k = 0; k < options_.n_lambda; ++k)
            if (cv_count[k] > 0) cv_mse_path_[k] = cv_sq_err[k] / static_cast<double>(cv_count[k]);

        std::size_t best_k     = 0;
        double      best_error = std::numeric_limits<double>::infinity();
        for (std::size_t k = 0; k < options_.n_lambda; ++k) {
            if (std::isnan(cv_mse_path_[k])) continue;
            if (cv_mse_path_[k] < best_error) {
                best_error = cv_mse_path_[k];
                best_k     = k;
            }
        }

        best_lambda_index_ = best_k;
        lambda_used_       = lambda_path_[best_k];
        coefficients_.assign(p, 0.0);
        for (std::size_t j = 0; j < p; ++j) coefficients_[j] = path_coefficients_(best_k, j);
    }

    intercept_ = has_int ? y_mean_ - std::inner_product(feature_mean_.begin(), feature_mean_.end(),
                                                        coefficients_.begin(), 0.0)
                          : 0.0;

    fitted_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        double s = intercept_;
        for (std::size_t j = 0; j < p; ++j) s += Xp(i, j) * coefficients_[j];
        fitted_[i] = s;
    }
    residuals_.resize(n);
    for (std::size_t i = 0; i < n; ++i) residuals_[i] = dm.y[i] - fitted_[i];
}

std::size_t ElasticNet::non_zero_coefficients() const {
    return static_cast<std::size_t>(
        std::count_if(coefficients_.begin(), coefficients_.end(), [](const double c) { return std::fabs(c) > 1e-10; }));
}

double ElasticNet::r_squared() const {
    const double mean =
        std::accumulate(training_y_.begin(), training_y_.end(), 0.0) / static_cast<double>(training_y_.size());
    double tss = 0.0, rss = 0.0;
    for (std::size_t i = 0; i < training_y_.size(); ++i) {
        const double dt = training_y_[i] - mean;
        const double dr = residuals_[i];
        tss += dt * dt;
        rss += dr * dr;
    }
    return (tss > 0.0) ? (1.0 - rss / tss) : 1.0;
}

double ElasticNet::rmse() const {
    double sse = 0.0;
    for (const double r : residuals_) sse += r * r;
    return std::sqrt(sse / static_cast<double>(residuals_.size()));
}

std::string ElasticNet::summary() const {
    std::ostringstream out;
    out << "Call:\nelasticNet(formula = " << formula_.text() << ", alpha = " << format_stat(options_.alpha, 3)
        << ")\n\n";

    if (lambda_was_selected_)
        out << "Lambda: " << format_stat(lambda_used_, 6) << "  (selected by " << options_.cv_folds
            << "-fold cross-validation over " << lambda_path_.size() << " values)\n";
    else
        out << "Lambda: " << format_stat(lambda_used_, 6) << "  (fixed)\n";

    out << "Non-zero coefficients: " << non_zero_coefficients() << " of " << predictor_names_.size() << "\n\n";

    out << "Coefficients:\n";
    if (design_.has_intercept) out << "  (Intercept): " << format_stat(intercept_) << "\n";
    for (std::size_t j = 0; j < predictor_names_.size(); ++j)
        out << "  " << predictor_names_[j] << ": " << format_stat(coefficients_[j]) << "\n";

    out << "\nR-squared: " << format_stat(r_squared(), 4) << "\n";
    out << "RMSE: " << format_stat(rmse(), 4) << "  (" << observations_ << " observations)\n";

    return out.str();
}

void ElasticNet::print_summary(std::ostream& os) const { os << summary(); }

void ElasticNet::print_summary() const { print_summary(std::cout); }

std::vector<double> ElasticNet::predict(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "ElasticNet::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);

    const bool         has_int = design_.has_intercept;
    const std::size_t p       = predictor_names_.size();

    std::vector<double> compact(dm.X.rows());
    for (std::size_t i = 0; i < dm.X.rows(); ++i) {
        double s = intercept_;
        for (std::size_t j = 0; j < p; ++j) s += dm.X(i, j + (has_int ? 1 : 0)) * coefficients_[j];
        compact[i] = s;
    }
    return expand_to_full(dm.used_row_indices, newdata.nrows(), compact);
}

plot::RPlot ElasticNet::plot_coefficient_path() const {
    if (!lambda_was_selected_)
        throw std::invalid_argument("ElasticNet::plot_coefficient_path: no path available for a fixed-lambda fit");

    std::vector<double> log_lambda(lambda_path_.size());
    for (std::size_t k = 0; k < lambda_path_.size(); ++k) log_lambda[k] = std::log10(lambda_path_[k]);

    const plot::RGB colors[] = {
        {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
    };

    auto plot = plot::RPlot::create();
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        std::vector<double> trace(lambda_path_.size());
        for (std::size_t k = 0; k < lambda_path_.size(); ++k) trace[k] = path_coefficients_(k, j);
        plot.line(log_lambda, trace, predictor_names_[j], colors[j % (sizeof(colors) / sizeof(colors[0]))]);
    }
    plot.title("Coefficient Path").x_label("log10(lambda)").y_label("Coefficient");
    return plot;
}

plot::RPlot ElasticNet::plot_cv_curve() const {
    if (!lambda_was_selected_)
        throw std::invalid_argument("ElasticNet::plot_cv_curve: no cross-validation curve for a fixed-lambda fit");

    // A handful of folds can occasionally diverge to a numerically infinite,
    // or merely astronomically large but still technically finite, squared
    // error at the smallest lambdas on the path (most often for lasso/
    // elastic-net folds with near-collinear predictors). Either corrupts
    // axis auto-ranging for the whole plot, so both are excluded: first the
    // non-finite entries, then any finite entry many orders of magnitude
    // above the best (minimum) achieved error, which is itself guaranteed
    // finite and representative since it is what lambda selection is based
    // on.
    double min_finite_mse = std::numeric_limits<double>::infinity();
    for (const double mse : cv_mse_path_)
        if (std::isfinite(mse) && mse < min_finite_mse) min_finite_mse = mse;

    std::vector<double> log_lambda;
    std::vector<double> finite_mse;
    log_lambda.reserve(lambda_path_.size());
    finite_mse.reserve(lambda_path_.size());
    for (std::size_t k = 0; k < lambda_path_.size(); ++k) {
        if (!std::isfinite(cv_mse_path_[k])) continue;
        if (cv_mse_path_[k] > 1000.0 * min_finite_mse) continue;
        log_lambda.push_back(std::log10(lambda_path_[k]));
        finite_mse.push_back(cv_mse_path_[k]);
    }

    auto plot = plot::RPlot::create();
    plot.line(log_lambda, finite_mse, "CV mean squared error");
    if (std::isfinite(cv_mse_path_[best_lambda_index_]))
        plot.points({std::log10(lambda_path_[best_lambda_index_])}, {cv_mse_path_[best_lambda_index_]},
                   "selected lambda", {220, 38, 38}, 8.0);
    plot.title("Cross-Validation Curve").x_label("log10(lambda)").y_label("CV mean squared error");
    return plot;
}

plot::RPlot ElasticNet::plot_predicted_vs_actual() const {
    auto plot = plot::RPlot::create();
    plot.points(training_y_, fitted_, "predictions");
    plot.title("Predicted vs Actual").x_label("Actual").y_label("Predicted");
    return plot;
}

plot::RPlot ElasticNet::plot_residuals_vs_fitted() const {
    auto plot = plot::RPlot::create();
    plot.points(fitted_, residuals_, "residuals");
    plot.title("Residuals vs Fitted").x_label("Fitted values").y_label("Residuals");
    return plot;
}

Ridge::Ridge(const dstruct::DataFrame& data, const std::string& formula, RidgeOptions options)
    : ElasticNet(data, formula, 0.0, [&] {
        ElasticNetOptions o;
        o.lambda      = options.lambda;
        o.n_lambda    = options.n_lambda;
        o.cv_folds    = options.cv_folds;
        o.max_iter    = options.max_iter;
        o.tol         = options.tol;
        o.seed        = options.seed;
        o.standardize = options.standardize;
        return o;
      }()) {}

Lasso::Lasso(const dstruct::DataFrame& data, const std::string& formula, LassoOptions options)
    : ElasticNet(data, formula, 1.0, [&] {
        ElasticNetOptions o;
        o.lambda      = options.lambda;
        o.n_lambda    = options.n_lambda;
        o.cv_folds    = options.cv_folds;
        o.max_iter    = options.max_iter;
        o.tol         = options.tol;
        o.seed        = options.seed;
        o.standardize = options.standardize;
        return o;
      }()) {}

} // namespace datamunge::stats
