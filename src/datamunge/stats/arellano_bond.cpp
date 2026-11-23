/// @file arellano_bond.cpp
/// @brief First-difference GMM, Windmeijer covariance, and serial correlation tests.

#include <datamunge/stats/arellano_bond.hpp>

#include <datamunge/linalg/eigen.hpp>
#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace datamunge::stats {
/// @brief Working observations, sparse instruments and GMM calculations.
namespace {

/// @brief Dense matrix for the estimating moments and their weights.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Vector of extended-precision observations or moments.
using Vector = std::vector<long double>;
/// @brief Original unit or period labels.
using Labels = std::vector<std::int64_t>;
/// @brief Instrument entries at a single differenced equation.
using Instruments = std::vector<std::pair<std::size_t, long double>>;
/// @brief Instrument identifier by variable, equation period (when uncollapsed), and lag.
using InstrumentKey = std::tuple<std::size_t, std::int64_t, std::uint64_t>;

/// @brief One usable equation at a specific unit and period.
struct Equation {
    std::size_t original;         ///< Associated original observation index.
    std::int64_t time;             ///< Period of the left-hand-side difference.
    long double response;          ///< First-differenced outcome.
    Vector design;                 ///< Differenced predictors and optional period indicators.
    Instruments instruments;       ///< Sparse matched instruments.
};

/// @brief Scaled equations grouped by unit.
struct Panel {
    std::vector<Equation> equations;                        ///< All usable equations, sorted within unit.
    std::vector<std::pair<std::size_t, std::size_t>> groups;  ///< Half-open equation ranges by unit.
    Labels time_levels;                                      ///< Sorted dates with optional differenced effects.
    Vector design_scale;                                     ///< Absolute peak of each unscaled differenced design column.
    long double response_scale{1};                          ///< Absolute peak of the unscaled differenced response, or one.
    std::size_t instrument_columns{};                       ///< Number of columns in the sparse instrument design.
};

/// @brief Positive weighting subspace and square root of the associated inverse.
struct Weight {
    Matrix half; ///< Rank-by-instrument matrix whose cross-product is the generalized inverse.
};

/// @brief One estimation step and its coefficient response to the moment vector.
struct StepFit {
    Vector beta;                ///< Fitted coefficients in working coordinates.
    Matrix gain;                ///< Map from instrument total to coefficients.
    Matrix bread;               ///< Inverse weighted coefficient information.
    Vector residuals;           ///< Fitted differenced residuals.
    std::vector<Vector> scores; ///< Residual instrument totals by unit.
};

/// @brief Positive distance between sorted period labels, without signed overflow.
std::uint64_t distance(std::int64_t later, std::int64_t earlier) {
    return static_cast<std::uint64_t>(later) - static_cast<std::uint64_t>(earlier);
}

/// @brief Checks that a dense working matrix fits its storage type.
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols > 0 && rows > Vector{}.max_size() / cols)
        throw std::length_error("Arellano-Bond: dense moment workspace exceeds storage capacity");
}

/// @brief Creates a sparse entry, assigning an instrument column to a new key.
void add_instrument(std::map<InstrumentKey, std::size_t>& columns, Equation& row, InstrumentKey key,
    long double value) {
    const auto it = columns.try_emplace(key, columns.size()).first;
    row.instruments.emplace_back(it->second, value);
}

/// @brief Builds consecutive differences and eligible observed level instruments.
Panel build_panel(const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
    const Labels& unit, const Labels& time, const ArellanoBondOptions& options) {
    const auto p = predictors.cols();
    std::map<std::int64_t, std::vector<std::size_t>> rows;
    for (std::size_t i = 0; i < unit.size(); ++i)
        rows[unit[i]].push_back(i);
    auto types = options.regressor_types;
    if (types.empty())
        types.resize(p, ArellanoBondRegressor::StrictlyExogenous);
    std::map<InstrumentKey, std::size_t> columns;
    Panel panel;
    for (auto& [id, indices] : rows) {
        std::ranges::sort(indices, {}, [&](std::size_t i) { return time[i]; });
        for (std::size_t i = 1; i < indices.size(); ++i)
            if (time[indices[i]] == time[indices[i-1]])
                throw std::invalid_argument("Arellano-Bond: duplicate period within a unit");
        const auto begin = panel.equations.size();
        for (std::size_t j = 2; j < indices.size(); ++j) {
            const auto i = indices[j], past = indices[j-1], earlier = indices[j-2];
            const auto period = time[i];
            if (distance(period, time[past]) != 1 || distance(time[past], time[earlier]) != 1)
                continue;
            Equation row{i, period, static_cast<long double>(response[i]) - response[past], Vector(p+1), {}};
            row.design[0] = static_cast<long double>(response[past]) - response[earlier];
            for (std::size_t k = 0; k < p; ++k) {
                row.design[k+1] = static_cast<long double>(predictors(i,k)) - predictors(past,k);
                if (types[k] == ArellanoBondRegressor::StrictlyExogenous)
                    add_instrument(columns, row, {k+1, 0, 0}, row.design[k+1]);
            }
            const auto key_period = options.collapse ? 0 : period;
            for (std::size_t h = j; h-- > 0;) {
                const auto prior = indices[h];
                const auto lag = distance(period, time[prior]);
                if (options.maximum_lag && lag > options.maximum_lag)
                    break;
                if (lag >= options.minimum_outcome_lag)
                    add_instrument(columns, row, {0, key_period, lag}, response[prior]);
                for (std::size_t k = 0; k < p; ++k)
                    if (types[k] == ArellanoBondRegressor::Predetermined ||
                        (types[k] == ArellanoBondRegressor::Endogenous && lag >= 2))
                        add_instrument(columns, row, {k+1, key_period, lag}, predictors(prior,k));
            }
            if (row.instruments.empty() && !options.time_effects)
                continue;
            panel.equations.push_back(std::move(row));
        }
        if (begin != panel.equations.size())
            panel.groups.emplace_back(begin, panel.equations.size());
    }
    if (panel.groups.size() < 2)
        throw std::invalid_argument("Arellano-Bond: need usable equations from at least two units");
    if (options.time_effects) {
        auto& levels = panel.time_levels;
        for (const auto& row : panel.equations)
            levels.push_back(row.time);
        std::ranges::sort(levels);
        levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
        check_storage(panel.equations.size(), p+1+levels.size());
        for (auto& row : panel.equations) {
            const auto position = std::ranges::lower_bound(levels, row.time) - levels.begin();
            row.design.resize(p+1+levels.size());
            row.design[p+1+position] = 1;
            add_instrument(columns, row, {p+1, row.time, 0}, 1);
        }
    }
    panel.instrument_columns = columns.size();
    const auto n = panel.equations.size(), slopes = p+1+panel.time_levels.size();
    if (n <= slopes)
        throw std::invalid_argument("Arellano-Bond: too few differenced equations for the requested coefficients");
    check_storage(n, slopes);
    check_storage(panel.instrument_columns, panel.instrument_columns);
    panel.design_scale.resize(slopes);
    Vector z_scale(columns.size());
    long double y_scale = 0;
    for (const auto& row : panel.equations) {
        y_scale = std::max(y_scale, std::abs(row.response));
        for (std::size_t j = 0; j < slopes; ++j)
            panel.design_scale[j] = std::max(panel.design_scale[j], std::abs(row.design[j]));
        for (const auto& [j, value] : row.instruments)
            z_scale[j] = std::max(z_scale[j], std::abs(value));
    }
    if (std::ranges::any_of(panel.design_scale, [](long double value) { return value == 0; }))
        throw std::domain_error("Arellano-Bond: a design column has no differenced variation");
    panel.response_scale = y_scale > 0 ? y_scale : 1;
    for (auto& row : panel.equations) {
        row.response /= panel.response_scale;
        for (std::size_t j = 0; j < slopes; ++j)
            row.design[j] /= panel.design_scale[j];
        for (auto& [j, value] : row.instruments)
            if (z_scale[j] > 0)
                value /= z_scale[j];
    }
    return panel;
}

/// @brief Adds a weighted outer product of two sparse instrument rows.
void add_sparse_product(Matrix& matrix, const Instruments& a, const Instruments& b, long double multiplier) {
    for (const auto& [i, x] : a)
        for (const auto& [j, y] : b)
            matrix(i,j) += multiplier * (x*y);
}

/// @brief Forms the initial covariance of the aggregate moments under spherical level disturbances.
Matrix initial_meat(const Panel& panel) {
    const auto q = panel.instrument_columns;
    Matrix result(q,q);
    for (const auto& [begin, end] : panel.groups) {
        for (std::size_t i = begin; i < end; ++i) {
            const auto& row = panel.equations[i];
            add_sparse_product(result, row.instruments, row.instruments, 2);
            if (i > begin && distance(row.time, panel.equations[i-1].time) == 1) {
                const auto& previous = panel.equations[i-1].instruments;
                add_sparse_product(result, row.instruments, previous, -1);
                add_sparse_product(result, previous, row.instruments, -1);
            }
        }
    }
    return result;
}

/// @brief Numerical square root of the inverse on the positive moment covariance subspace.
Weight weight(const Matrix& covariance, long double tolerance) {
    const auto q = covariance.rows();
    const auto eigen = linalg::symmetric_eigen(covariance);
    const auto cutoff = tolerance * std::max(0.0L, eigen.eigenvalues.front());
    const auto first_zero = std::ranges::find_if(eigen.eigenvalues, [&](auto value) { return value <= cutoff; });
    const auto rank = static_cast<std::size_t>(first_zero - eigen.eigenvalues.begin());
    Matrix half(rank,q);
    for (std::size_t k = 0; k < rank; ++k)
        for (std::size_t j = 0; j < q; ++j)
            half(k,j) = eigen.eigenvectors(j,k) / std::sqrt(eigen.eigenvalues[k]);
    return {std::move(half)};
}

/// @brief Inner product of vectors in working coordinates.
long double dot(const Vector& a, const Vector& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0L);
}

/// @brief Fits a weighted moment system and aggregates the resulting residual scores.
StepFit fit_step(const Panel& panel, const Weight& weight, const Matrix& moment_design, const Vector& moment_y,
    long double tolerance) {
    const auto p = panel.design_scale.size(), q = panel.instrument_columns;
    if (weight.half.rows() < p)
        throw std::domain_error("Arellano-Bond: insufficient independent instruments");
    const auto factor = linalg::svd(weight.half * moment_design);
    if (!factor.converged)
        throw std::runtime_error("Arellano-Bond: weighted design factorization failed");
    if (factor.rank(tolerance) != p)
        throw std::domain_error("Arellano-Bond: retained moments do not identify all coefficients");
    const auto inverse = factor.pseudo_inverse(tolerance);
    StepFit fit{factor.solve(weight.half * moment_y, tolerance), inverse * weight.half,
        inverse * inverse.transpose(), Vector(panel.equations.size()), {}};
    fit.scores.reserve(panel.groups.size());
    for (const auto& [begin, end] : panel.groups) {
        auto& score = fit.scores.emplace_back(q);
        for (std::size_t i = begin; i < end; ++i) {
            const auto& row = panel.equations[i];
            fit.residuals[i] = row.response - dot(row.design, fit.beta);
            for (const auto& [j, value] : row.instruments)
                score[j] += value * fit.residuals[i];
        }
    }
    return fit;
}

/// @brief Forms the uncentered sample cross-product of unit scores.
Matrix robust_meat(const std::vector<Vector>& scores) {
    const auto q = scores.front().size();
    Matrix covariance(q,q);
    for (const auto& score : scores)
        for (std::size_t i = 0; i < q; ++i)
            for (std::size_t j = 0; j <= i; ++j)
                covariance(i,j) += score[i] * score[j];
    for (std::size_t i = 0; i < q; ++i)
        for (std::size_t j = 0; j < i; ++j)
            covariance(j,i) = covariance(i,j);
    return covariance;
}

/// @brief Sums unit scores into the full fitted moment vector.
Vector total_score(const std::vector<Vector>& scores) {
    Vector total(scores.front().size());
    for (const auto& score : scores)
        for (std::size_t j = 0; j < total.size(); ++j)
            total[j] += score[j];
    return total;
}

/// @brief Corrects second-step covariance for the estimated first-step weighting matrix.
Matrix windmeijer(const Panel& panel, const Weight& weight, const StepFit& first, const StepFit& second,
    const Matrix& first_covariance, const Vector& second_total) {
    const auto p = panel.design_scale.size(), q = panel.instrument_columns;
    const auto weighted = weight.half.transpose() * (weight.half * second_total);
    Matrix change(q,p);
    for (std::size_t g = 0; g < panel.groups.size(); ++g) {
        const auto& [begin, end] = panel.groups[g];
        const auto& score = first.scores[g];
        const auto score_weight = dot(score, weighted);
        Vector design_weight(p);
        for (std::size_t i = begin; i < end; ++i) {
            const auto& row = panel.equations[i];
            long double instrument_weight = 0;
            for (const auto& [j, value] : row.instruments)
                instrument_weight += value * weighted[j];
            for (std::size_t k = 0; k < p; ++k)
                design_weight[k] += row.design[k] * instrument_weight;
            for (const auto& [j, value] : row.instruments)
                for (std::size_t k = 0; k < p; ++k)
                    change(j,k) += value * row.design[k] * score_weight;
        }
        for (std::size_t j = 0; j < q; ++j)
            for (std::size_t k = 0; k < p; ++k)
                change(j,k) += score[j] * design_weight[k];
    }
    const auto derivative = second.gain * change;
    const auto adjustment = derivative * second.bread;
    return second.bread + adjustment + adjustment.transpose() +
        (derivative * first_covariance) * derivative.transpose();
}

/// @brief Finite conversion for estimates in their original units.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("Arellano-Bond: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

/// @brief Unit residual products and derivative of the serial moment at one lag.
struct SerialMoment {
    Vector products;        ///< Sum of lagged error products per unit.
    Vector ex;              ///< Sum of lagged errors times current regressors.
    std::size_t pairs{};    ///< Count of observed pairs at this lag.
};

/// @brief Tests serial correlation using period-matched first-step residuals and robust influences.
std::vector<ArellanoBondSerialTest> serial_tests(const Panel& panel, const StepFit& first, std::size_t max_lag) {
    if (max_lag == 0)
        return {};
    std::map<std::size_t, SerialMoment> moments;
    const auto p = panel.design_scale.size(), groups = panel.groups.size();
    for (std::size_t g = 0; g < groups; ++g) {
        const auto& [begin, end] = panel.groups[g];
        for (std::size_t i = begin; i < end; ++i) {
            const auto& row = panel.equations[i];
            for (std::size_t j = i; j-- > begin;) {
                const auto lag = distance(row.time, panel.equations[j].time);
                if (lag > max_lag)
                    break;
                auto& moment = moments.try_emplace(lag).first->second;
                if (moment.pairs == 0) {
                    moment.products.resize(groups);
                    moment.ex.resize(p);
                }
                ++moment.pairs;
                moment.products[g] += first.residuals[i] * first.residuals[j];
                for (std::size_t k = 0; k < p; ++k)
                    moment.ex[k] += row.design[k] * first.residuals[j];
            }
        }
    }
    std::vector<ArellanoBondSerialTest> result;
    for (const auto& [lag, moment] : moments) {
        const auto sensitivity = first.gain.transpose() * moment.ex;
        long double variance = 0;
        for (std::size_t g = 0; g < groups; ++g) {
            const auto contribution = moment.products[g] - dot(sensitivity, first.scores[g]);
            variance += contribution * contribution;
        }
        if (variance > 0) {
            const auto total = std::accumulate(moment.products.begin(), moment.products.end(), 0.0L);
            const auto statistic = output(total / std::sqrt(variance));
            result.push_back({lag, moment.pairs, statistic, detail::normal_test_p(statistic, Alternative::TwoSided)});
        }
    }
    return result;
}

} // namespace

ArellanoBondFit fit_arellano_bond(const linalg::DenseMatrix<double>& predictors, const std::vector<double>& response,
    const std::vector<std::int64_t>& unit, const std::vector<std::int64_t>& time, const ArellanoBondOptions& options) {
    const auto n = response.size();
    const auto finite = [](double x) { return std::isfinite(x); };
    const auto valid_type = [](auto type) {
        return type == ArellanoBondRegressor::StrictlyExogenous || type == ArellanoBondRegressor::Predetermined ||
            type == ArellanoBondRegressor::Endogenous;
    };
    if (n == 0 || n != unit.size() || n != time.size() || n != predictors.rows() ||
        !std::ranges::all_of(response, finite) || !std::ranges::all_of(predictors.data(), finite) ||
        (!options.regressor_types.empty() && options.regressor_types.size() != predictors.cols()) ||
        !std::ranges::all_of(options.regressor_types, valid_type) ||
        (options.step != ArellanoBondStep::One && options.step != ArellanoBondStep::Two) ||
        options.minimum_outcome_lag < 2 || (options.maximum_lag && options.maximum_lag < options.minimum_outcome_lag) ||
        !(options.relative_tolerance > 0 && options.relative_tolerance < 1))
        throw std::invalid_argument("Arellano-Bond: invalid inputs or controls");

    const auto panel = build_panel(predictors, response, unit, time, options);
    const auto p = panel.design_scale.size(), q = panel.instrument_columns;
    if (q < p)
        throw std::domain_error("Arellano-Bond: too few instrument columns");
    Matrix moment_design(q,p);
    Vector moment_y(q);
    for (const auto& row : panel.equations)
        for (const auto& [j, value] : row.instruments) {
            moment_y[j] += value * row.response;
            for (std::size_t k = 0; k < p; ++k)
                moment_design(j,k) += value * row.design[k];
        }

    const auto tolerance = std::max(static_cast<long double>(options.relative_tolerance),
        64.0L * q * std::numeric_limits<long double>::epsilon());
    const auto initial = weight(initial_meat(panel), tolerance);
    const auto first = fit_step(panel, initial, moment_design, moment_y, tolerance);
    const auto meat = robust_meat(first.scores);
    const auto first_covariance = (first.gain * meat) * first.gain.transpose();
    auto covariance = first_covariance;
    const StepFit* fit = &first;
    std::optional<StepFit> efficient;
    ArellanoBondFit result{};
    result.instrument_rank = initial.half.rows();
    if (options.step == ArellanoBondStep::Two) {
        // Keep the first-step instrument space fixed when differentiating the new weight.
        auto updated = weight((initial.half * meat) * initial.half.transpose(), tolerance);
        if (updated.half.rows() < initial.half.rows())
            throw std::domain_error("Arellano-Bond: singular second-step weight, reduce the instrument count");
        updated.half = updated.half * initial.half;
        efficient = fit_step(panel, updated, moment_design, moment_y, tolerance);
        fit = &*efficient;
        result.instrument_rank = updated.half.rows();
        const auto total = total_score(fit->scores);
        covariance = windmeijer(panel, updated, first, *fit, first_covariance, total);
        if (result.instrument_rank > p) {
            const auto weighted = updated.half * total;
            const auto statistic = output(dot(weighted, weighted));
            const auto df = result.instrument_rank - p;
            result.hansen = ArellanoBondOveridentification{statistic, df,
                detail::chi_squared_test_survival(statistic, df)};
        }
    }

    result.units = panel.groups.size();
    result.time_levels = panel.time_levels;
    result.instrument_columns = q;
    result.serial_correlation = serial_tests(panel, first, options.max_serial_order);
    result.coefficients.resize(p);
    result.covariance = linalg::DenseMatrix<double>(p,p);
    Vector scales(p);
    for (std::size_t j = 0; j < p; ++j) {
        scales[j] = panel.response_scale / panel.design_scale[j];
        result.coefficients[j] = output(scales[j] * fit->beta[j]);
        for (std::size_t k = 0; k <= j; ++k)
            result.covariance(j,k) = result.covariance(k,j) =
                output(scales[j] * scales[k] * std::midpoint(covariance(j,k), covariance(k,j)));
    }
    const auto equations = panel.equations.size();
    std::vector<std::size_t> order(equations);
    std::iota(order.begin(), order.end(), 0uz);
    std::ranges::sort(order, {}, [&](auto i) { return panel.equations[i].original; });
    for (const auto i : order) {
        const auto& row = panel.equations[i];
        result.equation_rows.push_back(row.original);
        result.residuals.push_back(output(panel.response_scale * fit->residuals[i]));
        result.fitted.push_back(output(panel.response_scale * (row.response - fit->residuals[i])));
    }
    return result;
}

ArellanoBondFit fit_arellano_bond(const std::vector<double>& response, const std::vector<std::int64_t>& unit,
    const std::vector<std::int64_t>& time, const ArellanoBondOptions& options) {
    return fit_arellano_bond(linalg::DenseMatrix<double>(response.size(),0), response, unit, time, options);
}

} // namespace datamunge::stats
