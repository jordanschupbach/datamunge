/// @file panel_fixed_effects.cpp
/// @brief Matrix-free absorption and clustered inference for additive panel models.

#include <datamunge/stats/panel_fixed_effects.hpp>

#include "cluster_covariance.hpp"
#include <datamunge/linalg/svd.hpp>
#include <datamunge/stats/detail/test_common.hpp>
#include <datamunge/stats/lm.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Sparse incidence, fixed-effect projection, and numerical helpers.
namespace {

/// @brief Working vector for the scaled panel columns.
using Vector = std::vector<long double>;
/// @brief Extended-precision within-transformed design.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Dense indices for observed unit and period labels.
using Indices = std::vector<std::size_t>;
/// @brief Original signed panel labels.
using Labels = std::vector<std::int64_t>;

/// @brief Unit-by-period incidence and constraints for fixed-effect absorption.
struct Incidence {
    Indices unit;                 ///< Unit index per row.
    Indices time;                 ///< Period index per row.
    Indices count;                ///< Number of observations per unit.
    Indices first;                ///< First observation index per unit.
    Vector inverse_diagonal;      ///< Schur-complement preconditioner, zero at anchored periods.
    std::size_t components{};     ///< Number of connected bipartite components.
};

/// @brief Compressed representative of an observed incidence component.
std::size_t root(Indices& parents, std::size_t i) {
    while (parents[i] != i) {
        parents[i] = parents[parents[i]];
        i = parents[i];
    }
    return i;
}

/// @brief Sorted labels and a map from each observation to its dense level index.
std::pair<Labels, Indices> encode(const Labels& labels) {
    auto levels = labels;
    std::ranges::sort(levels);
    levels.erase(std::unique(levels.begin(), levels.end()), levels.end());
    Indices index;
    index.reserve(labels.size());
    for (const auto label : labels)
        index.push_back(std::ranges::lower_bound(levels, label) - levels.begin());
    return {std::move(levels), std::move(index)};
}

/// @brief Builds incidence counts, graph components, and the diagonal preconditioner.
Incidence incidence(Indices unit, Indices time, std::size_t units, std::size_t periods) {
    if (units > Indices{}.max_size() - periods)
        throw std::length_error("panel fixed effects: incidence exceeds storage capacity");
    Incidence result{std::move(unit), std::move(time), Indices(units), Indices(units), Vector(periods), 0};
    Indices parents(units + periods), sizes(units + periods, 1);
    std::iota(parents.begin(), parents.end(), 0uz);
    std::map<std::pair<std::size_t, std::size_t>, std::size_t> cells;
    for (std::size_t i = 0; i < result.unit.size(); ++i) {
        const auto u = result.unit[i], t = result.time[i];
        if (result.count[u]++ == 0)
            result.first[u] = i;
        ++cells[{u, t}];
        auto a = root(parents, u), b = root(parents, units + t);
        if (a == b)
            continue;
        if (sizes[a] < sizes[b])
            std::swap(a, b);
        parents[b] = a;
        sizes[a] += sizes[b];
    }

    for (const auto& [cell, count] : cells) {
        const auto [u, t] = cell;
        result.inverse_diagonal[t] += static_cast<long double>(count) * (result.count[u] - count) / result.count[u];
    }
    std::vector<bool> anchored(parents.size());
    for (std::size_t t = 0; t < periods; ++t) {
        const auto component = root(parents, units + t);
        if (anchored[component]) {
            result.inverse_diagonal[t] = 1 / result.inverse_diagonal[t];
        } else {
            anchored[component] = true;
            result.inverse_diagonal[t] = 0;
            ++result.components;
        }
    }
    return result;
}

/// @brief Centers a column within units using a unit-local reference value.
Vector center_unit(const Incidence& panel, const Vector& column) {
    Vector mean(panel.count.size()), centered(column.size());
    for (std::size_t i = 0; i < column.size(); ++i) {
        const auto u = panel.unit[i];
        centered[i] = column[i] - column[panel.first[u]];
        mean[u] += centered[i];
    }
    for (std::size_t u = 0; u < mean.size(); ++u)
        mean[u] /= panel.count[u];
    for (std::size_t i = 0; i < column.size(); ++i)
        centered[i] -= mean[panel.unit[i]];
    return centered;
}

/// @brief Sums a unit-centered column into the free period coordinates.
Vector period_sums(const Incidence& panel, const Vector& column) {
    Vector sums(panel.inverse_diagonal.size());
    for (std::size_t i = 0; i < column.size(); ++i)
        sums[panel.time[i]] += column[i];
    for (std::size_t t = 0; t < sums.size(); ++t)
        if (panel.inverse_diagonal[t] == 0)
            sums[t] = 0;
    return sums;
}

/// @brief Expands period coefficients into observations and projects out unit effects.
Vector centered_periods(const Incidence& panel, const Vector& effects) {
    Vector expanded(panel.time.size());
    for (std::size_t i = 0; i < expanded.size(); ++i)
        expanded[i] = effects[panel.time[i]];
    return center_unit(panel, expanded);
}

/// @brief Inner product in the free period coordinates.
long double dot(const Vector& left, const Vector& right) {
    return std::inner_product(left.begin(), left.end(), right.begin(), 0.0L);
}

/// @brief Wide-precision infinity norm.
long double norm(const Vector& values) {
    long double maximum = 0;
    for (const auto value : values)
        maximum = std::max(maximum, std::abs(value));
    return maximum;
}

/// @brief Solves the anchored period Schur complement with preconditioned conjugate gradients.
Vector solve_time(const Incidence& panel, const Vector& right, long double tolerance, std::size_t limit) {
    const auto threshold = tolerance * norm(right), dimension = static_cast<long double>(right.size());
    Vector effects(right.size()), residual = right, direction(right.size());
    long double previous = 0;
    bool restart = true;
    for (std::size_t iteration = 0;; ++iteration) {
        if (norm(residual) <= threshold) {
            const auto applied = period_sums(panel, centered_periods(panel, effects));
            for (std::size_t j = 0; j < right.size(); ++j)
                residual[j] = right[j] - applied[j];
            // A wide-precision roundoff bound permits long panels whose effects accumulate over time.
            const auto roundoff = 8 * std::numeric_limits<long double>::epsilon() * std::sqrt(dimension) *
                (norm(applied) + norm(right));
            if (norm(residual) <= threshold + roundoff)
                return effects;
            restart = true;
        }
        if (iteration == limit)
            break;
        Vector scaled(right.size());
        for (std::size_t j = 0; j < right.size(); ++j)
            scaled[j] = panel.inverse_diagonal[j] * residual[j];
        const auto energy = dot(residual, scaled);
        if (!(energy > 0) || !std::isfinite(energy))
            break;
        const auto mixing = restart ? 0 : energy / previous;
        for (std::size_t j = 0; j < right.size(); ++j)
            direction[j] = scaled[j] + mixing * direction[j];
        const auto applied = period_sums(panel, centered_periods(panel, direction));
        const auto curvature = dot(direction, applied);
        if (!(curvature > 0) || !std::isfinite(curvature))
            break;
        const auto step = energy / curvature;
        for (std::size_t j = 0; j < right.size(); ++j) {
            effects[j] += step * direction[j];
            residual[j] -= step * applied[j];
        }
        previous = energy;
        restart = false;
    }
    throw std::runtime_error("panel fixed effects: period absorption did not converge");
}

/// @brief Residualized column and its associated anchored period coefficients.
struct Projection {
    Vector within;       ///< Column after absorbing unit and period effects.
    Vector time_effects; ///< Estimated effects at the period levels.
};

/// @brief Removes period effects from a unit-centered column without constructing dummy columns.
Projection absorb_periods(const Incidence& panel, Vector centered, const PanelFixedEffectsOptions& options) {
    const auto limit = options.max_iterations ? options.max_iterations : std::max(1000uz, panel.inverse_diagonal.size());
    auto effects = solve_time(panel, period_sums(panel, centered), options.tolerance, limit);
    const auto projected = centered_periods(panel, effects);
    for (std::size_t i = 0; i < centered.size(); ++i)
        centered[i] -= projected[i];
    return {std::move(centered), std::move(effects)};
}

/// @brief Checks finite output when converting from working precision.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("panel fixed effects: result exceeds the finite reporting range");
    return static_cast<double>(value);
}

} // namespace

PanelContrastResult PanelFixedEffectsFit::test_contrast(const std::vector<double>& weights, double null_value) const {
    const auto p = coefficients.size();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (p == 0 || weights.size() != p || cluster_df == 0 || covariance.rows() != p || covariance.cols() != p ||
        !std::isfinite(null_value) || !std::ranges::all_of(weights, finite) ||
        !std::ranges::all_of(coefficients, finite) || !std::ranges::all_of(covariance.data(), finite))
        throw std::invalid_argument("panel fixed effects: invalid contrast or fitted model");
    long double estimate = 0, variance = 0;
    for (std::size_t j = 0; j < p; ++j) {
        estimate += static_cast<long double>(weights[j]) * coefficients[j];
        for (std::size_t k = 0; k < p; ++k)
            variance += static_cast<long double>(weights[j]) * covariance(j,k) * weights[k];
    }
    if (!(variance > 0))
        throw std::domain_error("panel fixed effects: contrast variance must be positive");
    const auto se = std::sqrt(variance);
    const auto t = static_cast<double>((estimate - null_value) / se);
    return {output(estimate), output(se), t, cluster_df, detail::student_test_p(t, cluster_df, Alternative::TwoSided)};
}

PanelFixedEffectsFit fit_two_way_fixed_effects(const linalg::DenseMatrix<double>& predictors,
    const std::vector<double>& response, const std::vector<std::int64_t>& unit, const std::vector<std::int64_t>& time,
    const PanelFixedEffectsOptions& options) {
    const auto n = predictors.rows(), p = predictors.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    if (p == 0 || n <= p || response.size() != n || unit.size() != n || time.size() != n ||
        !(options.tolerance > 0 && options.tolerance < 1) ||
        !std::ranges::all_of(predictors.data(), finite) || !std::ranges::all_of(response, finite))
        throw std::invalid_argument("panel fixed effects: invalid data or solver controls");
    const auto capacity = Vector{}.max_size();
    if (n > capacity / p)
        throw std::length_error("panel fixed effects: slope design exceeds storage capacity");
    auto [unit_levels, unit_index] = encode(unit);
    auto [time_levels, time_index] = encode(time);
    const auto units = unit_levels.size(), periods = time_levels.size();

    std::vector<Labels> labels;
    std::size_t groups = 0;
    switch (options.clustering) {
        case PanelClustering::Unit: labels = {unit}; groups = units; break;
        case PanelClustering::Time: labels = {time}; groups = periods; break;
        case PanelClustering::UnitAndTime: labels = {unit, time}; groups = std::min(units, periods); break;
        default: throw std::invalid_argument("panel fixed effects: unknown clustering option");
    }
    if (groups < 2)
        throw std::invalid_argument("panel fixed effects: at least two clusters per dimension are required");
    const auto panel = incidence(std::move(unit_index), std::move(time_index), units, periods);
    const auto fixed_rank = units + periods - panel.components;
    if (fixed_rank >= n - p)
        throw std::invalid_argument("panel fixed effects: insufficient residual degrees of freedom");
    const auto residual_df = n - p - fixed_rank;

    Vector raw(n), x_scale(p);
    Matrix x(n,p), time_design(periods,p);
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            raw[i] = predictors(i,j);
        auto centered = center_unit(panel, raw);
        x_scale[j] = norm(centered);
        if (x_scale[j] == 0)
            throw std::invalid_argument("panel fixed effects: constant slope predictor within units");
        for (auto& value : centered)
            value /= x_scale[j];
        const auto transformed = absorb_periods(panel, std::move(centered), options);
        for (std::size_t i = 0; i < n; ++i)
            x(i,j) = transformed.within[i];
        for (std::size_t t = 0; t < periods; ++t)
            time_design(t,j) = transformed.time_effects[t];
    }
    const auto solution = linalg::svd(x);
    const auto cutoff = std::max(16.0L * options.tolerance, 32.0L * n * std::numeric_limits<long double>::epsilon()) *
        std::sqrt(static_cast<long double>(n));
    if (!solution.converged)
        throw std::runtime_error("panel fixed effects: slope factorization did not converge");
    if (solution.s.back() <= cutoff)
        throw std::invalid_argument("panel fixed effects: aliased slopes after absorption");

    std::ranges::copy(response, raw.begin());
    auto centered = center_unit(panel, raw);
    auto y_scale = norm(centered);
    if (y_scale == 0)
        y_scale = 1;
    for (auto& value : centered)
        value /= y_scale;
    auto y = absorb_periods(panel, std::move(centered), options);
    const auto beta = solution.solve(y.within, 0);
    const auto fitted_within = x * beta;

    PanelFixedEffectsFit result;
    result.unit_levels = std::move(unit_levels);
    result.time_levels = std::move(time_levels);
    result.connected_components = panel.components;
    result.fixed_effect_rank = fixed_rank;
    result.residual_df = residual_df;
    result.cluster_df = groups - 1;
    result.coefficients.resize(p);
    result.residuals.resize(n);
    result.fitted.resize(n);
    Matrix influence(p,p);
    for (std::size_t j = 0; j < p; ++j) {
        const auto multiplier = y_scale / x_scale[j];
        result.coefficients[j] = output(beta[j] * multiplier);
        for (std::size_t k = 0; k < p; ++k)
            influence(j,k) = solution.V(j,k) * multiplier / solution.s[k];
    }
    std::vector<Vector> scores(n, Vector(p));
    for (std::size_t i = 0; i < n; ++i) {
        const auto residual = y.within[i] - fitted_within[i];
        result.residuals[i] = output(y_scale * residual);
        result.fitted[i] = output(response[i] - y_scale * residual);
        for (std::size_t j = 0; j < p; ++j) {
            long double value = 0;
            for (std::size_t k = 0; k < p; ++k)
                value += solution.U(i,k) * influence(j,k);
            scores[i][j] = value * residual;
        }
    }
    result.covariance = detail::cluster_covariance_from_scores(scores, labels, residual_df,
        options.finite_sample_adjustment).covariance;

    // Combine the period solutions for the outcome and all slope columns.
    const auto period_adjustment = time_design * beta;
    result.time_effects.resize(periods);
    for (std::size_t t = 0; t < periods; ++t) {
        y.time_effects[t] -= period_adjustment[t];
        result.time_effects[t] = output(y.time_effects[t] * y_scale);
    }
    Vector unit_total(units);
    for (std::size_t i = 0; i < n; ++i) {
        long double adjusted = raw[i] / y_scale - y.time_effects[panel.time[i]];
        for (std::size_t j = 0; j < p; ++j)
            adjusted -= beta[j] * (predictors(i,j) / x_scale[j]);
        unit_total[panel.unit[i]] += adjusted;
    }
    result.unit_effects.resize(units);
    for (std::size_t u = 0; u < units; ++u)
        result.unit_effects[u] = output(y_scale * (unit_total[u] / panel.count[u]));
    return result;
}

} // namespace datamunge::stats
