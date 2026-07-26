#include <datamunge/stats/quantile_regression.hpp>

#include <datamunge/linalg/qr.hpp>

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

double check_loss(double residual, double tau) {
    return residual >= 0.0 ? tau * residual : (tau - 1.0) * residual;
}

double l2_norm(const std::vector<double>& values) {
    double sum = 0.0;
    for (const double value : values) sum += value * value;
    return std::sqrt(sum);
}

double weighted_quantile(std::vector<std::pair<double, double>> values, double tau) {
    std::sort(values.begin(), values.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
    double total_weight = 0.0;
    for (const auto& [_, weight] : values) total_weight += weight;
    const double target = tau * total_weight;
    double       cumulative = 0.0;
    for (const auto& [value, weight] : values) {
        cumulative += weight;
        if (cumulative >= target) return value;
    }
    return values.back().first;
}

} // namespace

QuantileRegression::QuantileRegression(const dstruct::DataFrame& data, const std::string& formula,
                                       QuantileRegressionOptions options)
    : formula_(formula), options_(std::move(options)) {
    fit(data);
}

void QuantileRegression::fit(const dstruct::DataFrame& data) {
    const double tau = options_.quantile;
    if (!(tau > 0.0 && tau < 1.0))
        throw std::invalid_argument("QuantileRegression: quantile must be strictly between 0 and 1");
    if (!(options_.rho > 0.0))
        throw std::invalid_argument("QuantileRegression: rho must be strictly positive");
    if (options_.max_iterations == 0)
        throw std::invalid_argument("QuantileRegression: max_iterations must be positive");
    if (!(options_.absolute_tolerance > 0.0) || !(options_.relative_tolerance > 0.0))
        throw std::invalid_argument("QuantileRegression: tolerances must be strictly positive");

    design_         = formula_.resolve(data);
    const auto raw  = design_.build_matrix(data, /*require_response=*/true);
    observations_   = raw.X.rows();
    const auto cols = raw.X.cols();
    if (observations_ == 0 || cols == 0)
        throw std::invalid_argument("QuantileRegression: the resolved design matrix is empty");
    if (observations_ < cols)
        throw std::invalid_argument("QuantileRegression: require at least as many observations as coefficients");

    weights_.assign(observations_, 1.0);
    if (options_.weights_column) {
        if (!data.has_column(*options_.weights_column))
            throw std::invalid_argument("QuantileRegression: weights column not found: "
                                        + *options_.weights_column);
        for (std::size_t i = 0; i < observations_; ++i) {
            const auto weight = data.optional_double_at(*options_.weights_column, raw.used_row_indices[i]);
            if (!weight)
                throw std::invalid_argument(
                    "QuantileRegression: weights column has a null value in a fitted row");
            if (!(*weight > 0.0) || !std::isfinite(*weight))
                throw std::invalid_argument("QuantileRegression: weights must be finite and strictly positive");
            weights_[i] = *weight;
        }
    }

    const auto decomposition = linalg::qr(raw.X);
    if (!decomposition.is_full_rank())
        throw std::runtime_error("QuantileRegression: design matrix is rank-deficient");

    // Scaled ADMM for y - X beta = r:
    // beta <- argmin ||y-r+u-X beta||^2
    // r    <- prox_{sum w_i rho_tau / rho}(y-X beta+u)
    // u    <- u + y-X beta-r
    coefficients_ = decomposition.solve(raw.y); // OLS is an effective warm start.
    std::vector<double> r(observations_, 0.0);
    std::vector<double> previous_r(observations_, 0.0);
    std::vector<double> u(observations_, 0.0);
    std::vector<double> rhs(observations_, 0.0);
    std::vector<double> xb(observations_, 0.0);
    std::vector<double> primal(observations_, 0.0);
    std::vector<double> dual_argument(observations_, 0.0);
    double              rho = options_.rho;

    for (std::size_t iteration = 1; iteration <= options_.max_iterations; ++iteration) {
        for (std::size_t i = 0; i < observations_; ++i) rhs[i] = raw.y[i] - r[i] + u[i];
        coefficients_ = decomposition.solve(rhs);
        xb            = raw.X * coefficients_;
        previous_r    = r;

        for (std::size_t i = 0; i < observations_; ++i) {
            const double value           = raw.y[i] - xb[i] + u[i];
            const double upper_threshold = weights_[i] * tau / rho;
            const double lower_threshold = weights_[i] * (1.0 - tau) / rho;
            if (value > upper_threshold)
                r[i] = value - upper_threshold;
            else if (value < -lower_threshold)
                r[i] = value + lower_threshold;
            else
                r[i] = 0.0;

            primal[i]        = raw.y[i] - xb[i] - r[i];
            u[i]            += primal[i];
            dual_argument[i] = r[i] - previous_r[i];
        }

        primal_residual_norm_ = l2_norm(primal);
        const auto dual_projected = raw.X.transpose() * dual_argument;
        dual_residual_norm_       = rho * l2_norm(dual_projected);
        const double primal_scale =
            std::max({l2_norm(raw.y), l2_norm(xb), l2_norm(r)});
        const double primal_tolerance =
            std::sqrt(static_cast<double>(observations_)) * options_.absolute_tolerance
            + options_.relative_tolerance * primal_scale;
        const auto projected_u = raw.X.transpose() * u;
        const double dual_tolerance =
            std::sqrt(static_cast<double>(cols)) * options_.absolute_tolerance
            + options_.relative_tolerance * rho * l2_norm(projected_u);

        iterations_ = iteration;
        if (primal_residual_norm_ <= primal_tolerance
            && dual_residual_norm_ <= dual_tolerance) {
            converged_ = true;
            break;
        }

        // Residual balancing changes only the augmented-Lagrangian penalty,
        // not the quantile-regression optimum. Rescale the scaled dual
        // variable so the unscaled multiplier remains unchanged.
        if (options_.adaptive_rho && iteration % 10 == 0) {
            constexpr double balance = 10.0;
            if (primal_residual_norm_ > balance * dual_residual_norm_ && rho < 1e8) {
                rho *= 2.0;
                for (double& value : u) value *= 0.5;
            } else if (dual_residual_norm_ > balance * primal_residual_norm_ && rho > 1e-8) {
                rho *= 0.5;
                for (double& value : u) value *= 2.0;
            }
        }
    }

    fitted_ = raw.X * coefficients_;
    residuals_.resize(observations_);
    objective_ = 0.0;
    for (std::size_t i = 0; i < observations_; ++i) {
        residuals_[i] = raw.y[i] - fitted_[i];
        objective_ += weights_[i] * check_loss(residuals_[i], tau);
    }

    std::vector<std::pair<double, double>> response_and_weight(observations_);
    for (std::size_t i = 0; i < observations_; ++i)
        response_and_weight[i] = {raw.y[i], weights_[i]};
    const double null_quantile = weighted_quantile(std::move(response_and_weight), tau);
    double       null_objective = 0.0;
    for (std::size_t i = 0; i < observations_; ++i)
        null_objective += weights_[i] * check_loss(raw.y[i] - null_quantile, tau);
    pseudo_r_squared_ =
        null_objective > 0.0 ? 1.0 - objective_ / null_objective
                             : (objective_ == 0.0 ? 1.0 : 0.0);
}

void QuantileRegression::validate_categorical_levels(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "QuantileRegression::predict");
}

std::vector<double> QuantileRegression::predict(const dstruct::DataFrame& newdata) const {
    validate_categorical_levels(newdata);
    const auto matrix = design_.build_matrix(newdata, /*require_response=*/false);
    return matrix.X * coefficients_;
}

std::string QuantileRegression::summary() const {
    std::ostringstream out;
    out << "Call:\nquantile_regression(formula = " << formula_.text()
        << ", quantile = " << options_.quantile << ")\n\n";
    out << "Coefficients:\n";
    out << std::left << std::setw(22) << "" << std::right << std::setw(14) << "Estimate" << "\n";
    for (std::size_t i = 0; i < coefficients_.size(); ++i)
        out << std::left << std::setw(22) << design_.coefficient_names[i] << std::right
            << std::setw(14) << std::fixed << std::setprecision(6) << coefficients_[i] << "\n";
    out << "\nQuantile: " << options_.quantile << "\n";
    out << "Objective (check loss): " << objective_ << "\n";
    out << "Pseudo R-squared: " << pseudo_r_squared_ << "\n";
    out << "ADMM iterations: " << iterations_ << "\n";
    out << "Converged: " << (converged_ ? "yes" : "no") << "\n";
    return out.str();
}

void QuantileRegression::print_summary(std::ostream& os) const { os << summary(); }

void QuantileRegression::print_summary() const { print_summary(std::cout); }

} // namespace datamunge::stats
