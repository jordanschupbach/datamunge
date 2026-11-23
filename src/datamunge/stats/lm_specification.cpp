/// @file lm_specification.cpp
/// @brief Auxiliary-regression diagnostics for fitted linear models.

#include <datamunge/stats/lm.hpp>

#include <datamunge/random/random.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

namespace datamunge::stats {

/// @brief Bounded scaling and rank-aware projections for the auxiliary regressions.
namespace {

/// @brief Auxiliary design in working precision.
using Design = linalg::DenseMatrix<long double>;

/// @brief Sum of squares and rank of an auxiliary regression.
struct Projection {
    std::size_t rank;       ///< Numerical dimension of the fitted subspace.
    long double explained;  ///< Squared norm of the fitted part.
    long double residual;   ///< Squared norm of the part orthogonal to the fitted subspace.
};

/// @brief Stores a pivoted QR factor in the design and returns its reflector multipliers.
std::vector<long double> factor_design(Design& design) {
    for (std::size_t j = 0; j < design.cols(); ++j) {
        long double scale = 0;
        for (std::size_t i = 0; i < design.rows(); ++i)
            scale = std::max(scale, std::abs(design(i, j)));
        if (scale > 0)
            for (std::size_t i = 0; i < design.rows(); ++i)
                design(i, j) /= scale;
    }
    // The active part of each eliminated column stores its Householder vector.
    std::vector<long double> multipliers;
    multipliers.reserve(std::min(design.rows(), design.cols()));
    std::size_t rank = 0;
    long double cutoff = 0;
    for (; rank < std::min(design.rows(), design.cols()); ++rank) {
        std::size_t pivot = rank;
        long double largest_norm2 = 0;
        for (std::size_t j = rank; j < design.cols(); ++j) {
            long double norm2 = 0;
            for (std::size_t i = rank; i < design.rows(); ++i)
                norm2 += design(i, j) * design(i, j);
            if (norm2 > largest_norm2) {
                pivot = j;
                largest_norm2 = norm2;
            }
        }
        const long double norm = std::sqrt(largest_norm2);
        if (rank == 0)
            cutoff = norm * std::max(design.rows(), design.cols()) * std::numeric_limits<double>::epsilon();
        if (!(norm > cutoff))
            break;
        for (std::size_t i = rank; i < design.rows(); ++i) {
            std::swap(design(i, rank), design(i, pivot));
            design(i, rank) /= norm;
        }
        design(rank, rank) += std::copysign(1.0L, design(rank, rank));
        long double reflector_norm2 = 0;
        for (std::size_t i = rank; i < design.rows(); ++i)
            reflector_norm2 += design(i, rank) * design(i, rank);
        const long double multiplier = 2 / reflector_norm2;
        multipliers.push_back(multiplier);
        for (std::size_t j = rank + 1; j < design.cols(); ++j) {
            long double dot = 0;
            for (std::size_t i = rank; i < design.rows(); ++i)
                dot += design(i, rank) * design(i, j);
            for (std::size_t i = rank; i < design.rows(); ++i)
                design(i, j) -= design(i, rank) * (multiplier * dot);
        }
    }
    return multipliers;
}

/// @brief Applies one stored Householder reflector to a vector.
void reflect(const Design& design, std::size_t column, long double multiplier, std::vector<long double>& values) {
    long double dot = 0;
    for (std::size_t i = column; i < values.size(); ++i)
        dot += design(i, column) * values[i];
    for (std::size_t i = column; i < values.size(); ++i)
        values[i] -= design(i, column) * (multiplier * dot);
}

/// @brief Projects a response onto the rescaled auxiliary column space.
Projection project(Design& design, const std::vector<long double>& response) {
    const auto multipliers = factor_design(design);
    const auto rank = multipliers.size();
    auto transformed = response;
    for (std::size_t j = 0; j < rank; ++j)
        reflect(design, j, multipliers[j], transformed);
    const auto split = transformed.begin() + static_cast<std::ptrdiff_t>(rank);
    return {rank, std::inner_product(transformed.begin(), split, transformed.begin(), 0.0L),
            std::inner_product(split, transformed.end(), split, 0.0L)};
}

/// @brief Evaluates the ratio of successive squared differences to residual sum of squares.
long double durbin_watson_ratio(const std::vector<long double>& residuals) {
    long double numerator = 0;
    for (std::size_t i = 1; i < residuals.size(); ++i) {
        const long double difference = residuals[i] - residuals[i - 1];
        numerator += difference * difference;
    }
    const auto denominator = std::inner_product(residuals.begin(), residuals.end(), residuals.begin(), 0.0L);
    if (!(denominator > 0))
        throw std::domain_error("LM Durbin-Watson: residual statistic is undefined");
    return numerator / denominator;
}

/// @brief Evaluates the partial F ratio with a roundoff guard for exact fits.
double partial_f(const Projection& projection, double df1, double df2) {
    const long double precision = 16 * std::numeric_limits<double>::epsilon();
    return projection.residual <= precision * precision * projection.explained
        ? std::numeric_limits<double>::infinity()
        : static_cast<double>((projection.explained / df1) / (projection.residual / df2));
}

/// @brief Centers a column so the auxiliary regression uses an implicit intercept.
void center_column(Design& design, std::size_t j) {
    const long double origin = design(0, j);
    long double mean = 0;
    for (std::size_t i = 0; i < design.rows(); ++i) {
        design(i, j) -= origin;
        mean += design(i, j) / static_cast<long double>(design.rows());
    }
    for (std::size_t i = 0; i < design.rows(); ++i)
        design(i, j) -= mean;
}

/// @brief Builds the centered auxiliary variance predictors and optional White expansion.
Design variance_design(const linalg::DenseMatrix<double>& input, std::size_t observations, bool expand) {
    const auto p = input.cols();
    if (input.rows() != observations || p == 0 ||
        !std::ranges::all_of(input.data(), [](double x) { return std::isfinite(x); }))
        throw std::invalid_argument("LM variance test: predictors must be finite and align with fitted rows");
    const auto max = std::numeric_limits<std::size_t>::max();
    if (expand && (p == max || p > (max - p) / (p + 1)))
        throw std::length_error("LM White test: auxiliary column count exceeds index range");
    const auto columns = expand ? p + p * (p + 1) / 2 : p;
    if (columns > max / observations)
        throw std::length_error("LM variance test: auxiliary matrix size exceeds index range");
    Design design(observations, columns);
    for (std::size_t j = 0; j < p; ++j) {
        long double scale = 0;
        for (std::size_t i = 0; i < observations; ++i)
            scale = std::max(scale, std::abs(static_cast<long double>(input(i, j))));
        if (scale == 0)
            continue;
        for (std::size_t i = 0; i < observations; ++i)
            design(i, j) = input(i, j) / scale;
        center_column(design, j);
    }
    if (expand) {
        std::size_t column = p;
        for (std::size_t j = 0; j < p; ++j) {
            for (std::size_t k = j; k < p; ++k, ++column) {
                for (std::size_t i = 0; i < observations; ++i)
                    design(i, column) = design(i, j) * design(i, k);
                center_column(design, column);
            }
        }
    }
    return design;
}

/// @brief Computes the Gaussian or studentized statistic on a centered auxiliary design.
HypothesisTestResult variance_test(Design design, std::vector<long double> residuals,
                                   bool studentized, std::string method) {
    long double mean = 0;
    for (auto& value : residuals) {
        value *= value;
        mean += value;
    }
    mean /= residuals.size();
    for (auto& value : residuals)
        value -= mean;
    const auto projection = project(design, residuals);
    const auto total = projection.explained + projection.residual;
    const long double precision = 16 * std::numeric_limits<double>::epsilon();
    if (!(total > residuals.size() * precision * precision) ||
        projection.rank == 0 || projection.rank >= residuals.size() - 1)
        throw std::domain_error("LM variance test: auxiliary regression has undefined inference");
    const long double r2 = projection.explained / total;
    const double statistic = static_cast<double>(studentized
        ? residuals.size() * r2 : projection.explained / (2 * mean * mean));
    const auto df = static_cast<double>(projection.rank);
    return {.statistic = statistic, .parameter1 = df, .p_value = detail::chi_squared_test_survival(statistic, df),
            .estimate1 = static_cast<double>(r2), .alternative = Alternative::Greater, .method = std::move(method)};
}

} // namespace

std::vector<long double> LM::scaled_working_residuals() const {
    std::vector<long double> values(observations());
    long double maximum = 0;
    long double response_scale = 0;
    for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] = static_cast<long double>(residuals_[i]) *
                    (weights_.empty() ? 1 : std::sqrt(static_cast<long double>(weights_[i])));
        maximum = std::max(maximum, std::abs(values[i]));
        response_scale = std::max(response_scale, std::abs(static_cast<long double>(response_[i])));
    }
    if (!std::isfinite(maximum) || !(maximum > 16 * std::numeric_limits<double>::epsilon() * response_scale))
        throw std::domain_error("LM specification test: residuals are numerically zero or nonfinite");
    for (auto& value : values)
        value /= maximum;
    return values;
}

linalg::DenseMatrix<long double> LM::augmented_working_design(std::size_t additional_columns) const {
    const auto max_columns = std::numeric_limits<std::size_t>::max() / observations();
    if (additional_columns > max_columns - rank())
        throw std::length_error("LM specification test: augmented matrix size exceeds index range");
    Design design(observations(), rank() + additional_columns);
    for (std::size_t i = 0; i < observations(); ++i) {
        if (sparse_design_matrix_) {
            const auto& sparse = *sparse_design_matrix_;
            for (std::size_t k = sparse.row_ptr[i]; k < sparse.row_ptr[i + 1]; ++k)
                design(i, sparse.col_idx[k]) = sparse.values[k];
        } else {
            for (std::size_t j = 0; j < rank(); ++j)
                design(i, j) = design_matrix_(i, j);
        }
    }
    return design;
}

HypothesisTestResult LM::breusch_pagan(const linalg::DenseMatrix<double>& variance_predictors, bool studentized) const {
    auto design = variance_design(variance_predictors, observations(), false);
    return variance_test(std::move(design), scaled_working_residuals(), studentized,
        studentized ? "Studentized Breusch-Pagan test" : "Gaussian Breusch-Pagan test");
}

HypothesisTestResult LM::white_heteroscedasticity(const linalg::DenseMatrix<double>& variance_predictors) const {
    auto design = variance_design(variance_predictors, observations(), true);
    return variance_test(std::move(design), scaled_working_residuals(), true, "White heteroscedasticity test");
}

HypothesisTestResult LM::ramsey_reset(const std::vector<unsigned>& powers) const {
    auto sorted = powers;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.empty() || sorted.front() < 2 || std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end())
        throw std::invalid_argument("LM RESET: specify distinct powers of at least two");
    long double fitted_scale = 0;
    for (double value : fitted_)
        fitted_scale = std::max(fitted_scale, std::abs(static_cast<long double>(value)));
    if (!(fitted_scale > 0) || !std::isfinite(fitted_scale))
        throw std::domain_error("LM RESET: fitted values give no usable powers");
    const auto residuals = scaled_working_residuals();
    auto design = augmented_working_design(powers.size());
    for (std::size_t i = 0; i < observations(); ++i) {
        const auto value = fitted_[i] / fitted_scale;
        const long double weight = weights_.empty() ? 1 : std::sqrt(static_cast<long double>(weights_[i]));
        for (std::size_t j = 0; j < powers.size(); ++j)
            design(i, rank() + j) = weight * std::pow(value, powers[j]);
    }
    const auto projection = project(design, residuals);
    if (projection.rank <= rank() || projection.rank >= observations())
        throw std::domain_error("LM RESET: additional terms must add rank and leave residual degrees of freedom");
    const auto df1 = static_cast<double>(projection.rank - rank());
    const auto df2 = static_cast<double>(observations() - projection.rank);
    const double statistic = partial_f(projection, df1, df2);
    return {.statistic = statistic, .parameter1 = df1, .parameter2 = df2,
            .p_value = detail::f_test_survival(statistic, df1, df2),
            .alternative = Alternative::Greater, .method = "Ramsey RESET"};
}

HypothesisTestResult LM::durbin_watson(random::RandomGenerator& rng, std::size_t simulations,
                                     Alternative alternative) const {
    if (simulations == 0 || (alternative != Alternative::Less && alternative != Alternative::Greater &&
                             alternative != Alternative::TwoSided))
        throw std::invalid_argument("LM Durbin-Watson: invalid simulation count or alternative");
    auto observed = scaled_working_residuals();
    auto design = augmented_working_design();
    const auto multipliers = factor_design(design);
    if (multipliers.size() != rank() || rank() >= observations())
        throw std::domain_error("LM Durbin-Watson: no usable numerical residual space");
    const auto from_residual_space = [&](std::vector<long double>& values) {
        std::fill_n(values.begin(), multipliers.size(), 0.0L);
        for (std::size_t j = multipliers.size(); j > 0; --j)
            reflect(design, j - 1, multipliers[j - 1], values);
    };
    for (std::size_t j = 0; j < multipliers.size(); ++j)
        reflect(design, j, multipliers[j], observed);
    from_residual_space(observed);
    const auto statistic = durbin_watson_ratio(observed);

    std::size_t lower = 0, upper = 0;
    std::vector<long double> draw(observations());
    const long double tolerance = 32 * std::numeric_limits<long double>::epsilon() * std::max(1.0L, statistic);
    for (std::size_t i = 0; i < simulations; ++i) {
        for (std::size_t j = multipliers.size(); j < draw.size(); ++j)
            draw[j] = rng.normal();
        from_residual_space(draw);
        const auto value = durbin_watson_ratio(draw);
        lower += value <= statistic + tolerance;
        upper += value >= statistic - tolerance;
    }
    const long double denominator = static_cast<long double>(simulations) + 1;
    const auto lower_tail = (static_cast<long double>(lower) + 1) / denominator;
    const auto upper_tail = (static_cast<long double>(upper) + 1) / denominator;
    const auto probability = alternative == Alternative::Greater ? lower_tail
        : alternative == Alternative::Less ? upper_tail : std::min(1.0L, 2 * std::min(lower_tail, upper_tail));
    return {.statistic = static_cast<double>(statistic), .parameter1 = static_cast<double>(simulations),
            .p_value = static_cast<double>(probability), .alternative = alternative,
            .method = "Durbin-Watson test (conditional Gaussian simulation)"};
}

HypothesisTestResult LM::breusch_godfrey(std::size_t lags, bool use_f) const {
    if (lags == 0 || lags >= observations())
        throw std::invalid_argument("LM Breusch-Godfrey: lag must be positive and less than the fitted row count");
    const auto residuals = scaled_working_residuals();
    auto design = augmented_working_design(lags);
    for (std::size_t lag = 1; lag <= lags; ++lag)
        for (std::size_t i = lag; i < observations(); ++i)
            design(i, rank() + lag - 1) = residuals[i - lag];
    const auto projection = project(design, residuals);
    if (projection.rank <= rank() || projection.rank >= observations())
        throw std::domain_error("LM Breusch-Godfrey: lags must add rank and leave residual degrees of freedom");
    const double df1 = static_cast<double>(projection.rank - rank());
    const double df2 = static_cast<double>(observations() - projection.rank);
    const auto explained = projection.explained / (projection.explained + projection.residual);
    const double statistic = use_f ? partial_f(projection, df1, df2)
                                   : static_cast<double>(observations() * explained);
    return {.statistic = statistic, .parameter1 = df1, .parameter2 = use_f ? df2 : 0,
            .p_value = use_f ? detail::f_test_survival(statistic, df1, df2)
                             : detail::chi_squared_test_survival(statistic, df1),
            .estimate1 = static_cast<double>(explained), .alternative = Alternative::Greater,
            .method = use_f ? "Breusch-Godfrey F test" : "Breusch-Godfrey LM test"};
}

} // namespace datamunge::stats
