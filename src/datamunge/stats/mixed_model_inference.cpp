/// @file mixed_model_inference.cpp
/// @brief REML information, contrast moments, and Kenward–Roger covariance adjustment.

#include <datamunge/stats/mixed_model_inference.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/svd.hpp>
#include <datamunge/random/distributions.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::stats {
/// @brief Dense REML information and contrast-space calculations.
namespace {
/// @brief Working matrix with widened products.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Working vector.
using Vector = std::vector<long double>;

/// @brief Converts a finite statistic while detecting overflow.
double output(long double x) {
    if (!std::isfinite(x) || std::abs(x) > std::numeric_limits<double>::max())
        throw std::overflow_error("mixed-model inference: result exceeds finite reporting range");
    return static_cast<double>(x);
}

/// @brief Inner product in working precision.
long double dot(const Vector& x, const Vector& y) {
    return std::inner_product(x.begin(), x.end(), y.begin(), 0.0L);
}

/// @brief Trace of a square working matrix.
long double trace(const Matrix& x) {
    long double result = 0;
    for (std::size_t i = 0; i < x.rows(); ++i)
        result += x(i, i);
    return result;
}

/// @brief Trace of two matrix factors without storing their product.
long double trace_product(const Matrix& x, const Matrix& y) {
    long double result = 0;
    for (std::size_t i = 0; i < x.rows(); ++i)
        for (std::size_t j = 0; j < x.cols(); ++j)
            result += x(i, j) * y(j, i);
    return result;
}

/// @brief Inverts positive information after equilibration to its diagonal.
Matrix inverse_spd(const Matrix& a) {
    const auto d = a.rows();
    Vector scale(d);
    Matrix corr(d, d);
    for (std::size_t i = 0; i < d; ++i) {
        if (!(a(i, i) > 0) || !std::isfinite(a(i, i)))
            throw std::runtime_error("mixed-model inference: information has an invalid diagonal");
        scale[i] = std::sqrt(a(i, i));
        corr(i, i) = 1;
        for (std::size_t j = 0; j < i; ++j) {
            const auto entry = .5L * (a(i, j) + a(j, i)) / scale[i] / scale[j];
            if (!std::isfinite(entry))
                throw std::runtime_error("mixed-model inference: nonfinite scaled information");
            corr(i, j) = entry;
        }
    }
    const auto factor = linalg::cholesky(corr);
    if (!factor.ok)
        throw std::runtime_error("mixed-model inference: covariance or information is singular or indefinite");
    for (std::size_t j = 0; j < d; ++j)
        if (!(factor.L(j, j) > 16 * std::sqrt(std::numeric_limits<long double>::epsilon())))
            throw std::runtime_error("mixed-model inference: covariance or information is ill-conditioned");
    auto result = factor.solve(Matrix::identity(d));
    for (std::size_t i = 0; i < d; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            result(i, j) = result(j, i) = result(i, j) / scale[i] / scale[j];
    return result;
}

/// @brief Computes covariance of the active variance parameters from REML information.
Matrix variance_covariance(const std::vector<Matrix>& kernels, const Matrix& projector,
    const std::vector<double>& response, long double sigma2, bool observed) {
    std::vector<Matrix> products;
    std::vector<Vector> error_parts, projected_error_parts;
    Vector error;
    if (observed) {
        error.assign(response.begin(), response.end());
        const auto scale = std::sqrt(sigma2);
        for (auto& x : error)
            x /= scale;
        error = projector * error;
    }
    for (const auto& k : kernels) {
        products.push_back(projector * k);
        if (observed) {
            error_parts.push_back(k * error);
            projected_error_parts.push_back(projector * error_parts.back());
        }
    }
    Matrix information(kernels.size(), kernels.size());
    for (std::size_t i = 0; i < kernels.size(); ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            const auto expected = .5L * trace_product(products[i], products[j]);
            information(i, j) = information(j, i) = observed ?
                dot(error_parts[i], projected_error_parts[j]) - expected : expected;
        }
    return inverse_spd(information);
}

/// @brief Matches the moments of one or more independent eigen-contrast t statistics.
long double satterthwaite(const Matrix& covariance, const std::vector<Matrix>& derivative,
                         const Matrix& variance) {
    const auto eigen = linalg::symmetric_eigen(covariance);
    Vector indices;
    for (std::size_t j = 0; j < eigen.eigenvalues.size(); ++j) {
        const auto value = eigen.eigenvalues[j];
        if (!(value > 0))
            throw std::runtime_error("mixed-model inference: nonpositive eigen-contrast variance");
        const auto direction = eigen.eigenvectors.col(j);
        Vector gradient;
        for (const auto& g : derivative)
            gradient.push_back(dot(direction, g * direction) / value);
        const auto relative_variance = dot(gradient, variance * gradient);
        if (!(relative_variance >= 0) || !std::isfinite(relative_variance))
            throw std::runtime_error("mixed-model inference: invalid contrast variance uncertainty");
        indices.push_back(relative_variance == 0 ? std::numeric_limits<long double>::infinity() : 2 / relative_variance);
    }
    if (indices.size() == 1)
        return indices.front();
    const auto [lo, hi] = std::ranges::minmax_element(indices);
    if (*lo == *hi)
        return *lo;
    if (*hi - *lo < 1e-8L)
        return std::midpoint(*lo, *hi);
    if (*lo <= 2)
        return 2;
    long double excess = 0;
    for (const auto nu : indices)
        excess += 2 / (nu - 2);
    return 2 + 2 * indices.size() / excess;
}

/// @brief Matching denominator degrees of freedom and F-scale for Kenward–Roger.
std::pair<long double,long double> kenward_roger(const Matrix& covariance,
    const std::vector<Matrix>& derivative, const Matrix& variance) {
    const auto inverse = inverse_spd(covariance);
    std::vector<Matrix> transformed;
    Vector traces;
    for (const auto& term : derivative) {
        transformed.push_back(inverse * term);
        traces.push_back(trace(transformed.back()));
    }
    const auto a1 = dot(traces, variance * traces);
    long double a2 = 0;
    for (std::size_t i = 0; i < transformed.size(); ++i)
        for (std::size_t j = 0; j < transformed.size(); ++j)
            a2 += variance(i, j) * trace_product(transformed[i], transformed[j]);
    if (!(a1 >= 0) || !(a2 >= 0) || !std::isfinite(a1) || !std::isfinite(a2))
        throw std::runtime_error("mixed-model inference: invalid Kenward–Roger moment");
    if (a2 == 0)
        return {std::numeric_limits<long double>::infinity(), 1};
    if (covariance.rows() == 1)
        return {2 / a2, 1};

    const long double q = covariance.rows();
    const auto b = (a1 + 6*a2) / (2*q);
    const auto g = ((q+1)*a1 - (q+4)*a2) / ((q+2)*a2);
    const auto common = 3*q + 2*(1-g);
    const auto c1 = g / common, c2 = (q-g) / common, c3 = (q+2-g) / common;
    const auto v0 = 1 + c1*b, v1 = 1 - c2*b, v2 = 1 - c3*b;
    const auto inverse_mean = 1 - a2 / q;
    const auto denominator = v1*v1*v2;
    const auto rho_remainder = (inverse_mean*inverse_mean*v0 - denominator) / denominator;
    const auto df = 4 + (q+2) / rho_remainder;
    const auto scale = inverse_mean / (1 - 2 / df);
    if (!(df > 0) || !(scale > 0) || !std::isfinite(scale))
        throw std::runtime_error("mixed-model inference: Kenward–Roger F moments are undefined");
    return {df, scale};
}

/// @brief Evaluates the F survival function, using the chi-squared limit for infinite degrees of freedom.
double f_upper(long double statistic, std::size_t numerator, long double denominator) {
    if (!std::isfinite(denominator) || denominator > 1e10L)
        return random::detail::gamma_probability(.5L * numerator * statistic, .5 * numerator, false);
    return random::beta_cdf(static_cast<double>(1 / (1 + numerator * statistic / denominator)),
                            static_cast<double>(.5L * denominator), .5 * numerator);
}

} // namespace

MixedModelContrastTest mixed_model_contrast_test(const VarianceComponentsFit& fit,
    const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const std::vector<linalg::DenseMatrix<double>>& random_designs, const linalg::DenseMatrix<double>& contrast,
    MixedModelDF method, const std::vector<double>& null_value) {
    const auto n = design.rows(), p = design.cols(), q = contrast.rows();
    const auto finite = [](double value) { return std::isfinite(value); };
    const bool adjusted = method == MixedModelDF::KenwardRoger;
    if (n <= p || p == 0 || response.size() != n || q == 0 || q > p || contrast.cols() != p ||
        (!null_value.empty() && null_value.size() != q) ||
        (!adjusted && method != MixedModelDF::Satterthwaite) ||
        !fit.reml || (fit.status != VarianceComponentsStatus::Converged && fit.status != VarianceComponentsStatus::Boundary) ||
        fit.coefficients.size() != p || fit.fitted.size() != n || fit.variances.size() != random_designs.size() ||
        fit.effects.size() != random_designs.size() || !finite(fit.residual_variance) || !(fit.residual_variance > 0) ||
        !std::ranges::all_of(fit.coefficients, finite) || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(contrast.data(), finite) || !std::ranges::all_of(response, finite) ||
        !std::ranges::all_of(null_value, finite))
        throw std::invalid_argument("mixed-model inference: invalid training data, contrast, or REML fit");
    const auto capacity = Vector{}.max_size(), components = fit.variances.size() + 1;
    if (n > capacity/n || components > capacity/components)
        throw std::length_error("mixed-model inference: dense working matrices exceed storage capacity");

    Matrix x(n, p);
    std::ranges::copy(design.data(), x.data().begin());
    Matrix l(q, p);
    Vector delta(q);
    for (std::size_t i = 0; i < q; ++i) {
        long double scale = 0;
        for (std::size_t j = 0; j < p; ++j)
            scale = std::max(scale, std::abs(static_cast<long double>(contrast(i, j))));
        if (scale == 0)
            throw std::invalid_argument("mixed-model inference: zero contrast row");
        delta[i] = null_value.empty() ? 0 : -null_value[i] / scale;
        for (std::size_t j = 0; j < p; ++j) {
            l(i, j) = contrast(i, j) / scale;
            delta[i] += l(i, j) * fit.coefficients[j];
        }
    }
    const auto rank = linalg::svd(l);
    if (!rank.converged || rank.rank() != q)
        throw std::invalid_argument("mixed-model inference: contrast rows must be independent");

    const long double sigma2 = fit.residual_variance;
    Matrix shape = Matrix::identity(n);
    std::vector<Matrix> kernels{shape};
    for (std::size_t j = 0; j < random_designs.size(); ++j) {
        const auto& design = random_designs[j];
        if (design.rows() != n || design.cols() == 0 || design.cols() != fit.effects[j].size() ||
            !finite(fit.variances[j]) || fit.variances[j] < 0 || !std::ranges::all_of(design.data(), finite))
            throw std::invalid_argument("mixed-model inference: invalid random-effect design or variance");
        if (fit.variances[j] == 0)
            continue;
        Matrix z(n, design.cols());
        const auto scale = std::sqrt(fit.variances[j] / sigma2);
        for (std::size_t i = 0; i < design.data().size(); ++i)
            z.data()[i] = scale * design.data()[i];
        kernels.push_back(z * z.transpose());
        shape = shape + kernels.back();
    }

    const auto precision = inverse_spd(shape);
    const auto precision_x = precision * x;
    auto covariance = inverse_spd(x.transpose() * precision_x);
    long double degrees = n-p, scale = 1;
    if (kernels.size() > 1) {
        const auto projector = precision - precision_x * covariance * precision_x.transpose();
        const auto weight = variance_covariance(kernels, projector, response, sigma2, !adjusted);
        auto moment_contrast = l;
        if (!adjusted && q > 1) {
            // The Satterthwaite eigenvectors use the supplied contrast
            // coordinates. Scale all rows together, preserving those axes.
            long double largest = 0;
            for (const auto value : contrast.data())
                largest = std::max(largest, std::abs(static_cast<long double>(value)));
            for (std::size_t i = 0; i < contrast.data().size(); ++i)
                moment_contrast.data()[i] = contrast.data()[i] / largest;
        }
        const auto transposed_contrast = moment_contrast.transpose();
        const auto contrast_covariance = moment_contrast * covariance * transposed_contrast;
        const auto mapped = precision_x * covariance;
        std::vector<Matrix> derivatives, parts;
        for (const auto& k : kernels) {
            parts.push_back(k * mapped);
            derivatives.push_back(moment_contrast * mapped.transpose() * parts.back() * transposed_contrast);
        }
        if (adjusted) {
            std::tie(degrees, scale) = kenward_roger(contrast_covariance, derivatives, weight);
            std::vector<Matrix> projected;
            for (const auto& part : parts)
                projected.push_back(projector * part);
            auto corrected = covariance;
            for (std::size_t i = 0; i < kernels.size(); ++i)
                for (std::size_t j = 0; j < kernels.size(); ++j) {
                    const auto term = parts[i].transpose() * projected[j];
                    for (std::size_t k = 0; k < term.data().size(); ++k)
                        corrected.data()[k] += 2 * weight(i, j) * term.data()[k];
                }
            covariance = std::move(corrected);
        } else {
            degrees = satterthwaite(contrast_covariance, derivatives, weight);
        }
    }

    const auto inverse_contrast = inverse_spd(l * covariance * l.transpose());
    const auto statistic = dot(delta, inverse_contrast * delta) * scale / sigma2 / q;
    if (!(statistic >= 0))
        throw std::runtime_error("mixed-model inference: invalid F statistic");
    MixedModelContrastTest result{linalg::DenseMatrix<double>(p, p), output(statistic), q,
        std::isinf(degrees) ? std::numeric_limits<double>::infinity() : output(degrees),
        output(scale), f_upper(statistic, q, degrees)};
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = 0; j <= i; ++j)
            result.covariance(i,j) = result.covariance(j,i) = output(.5L * sigma2 * (covariance(i,j) + covariance(j,i)));
    return result;
}

} // namespace datamunge::stats
