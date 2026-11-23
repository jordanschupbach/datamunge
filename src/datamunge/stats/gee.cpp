/// @file gee.cpp
/// @brief Working-correlation whitening, marginal scoring, and cluster sandwich inference.

#include <datamunge/stats/gee.hpp>

#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/linalg/svd.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Working data and blockwise estimating equations.
namespace {
/// @brief Extended-precision working vector.
using Vector = std::vector<long double>;
/// @brief Extended-precision working matrix.
using Matrix = linalg::DenseMatrix<long double>;

/// @brief Centered link quantities and correlation-ready Pearson residuals.
struct State {
    Vector eta;       ///< Current linear predictor including known offsets.
    Vector pearson;   ///< Residual divided by its unit-dispersion standard deviation.
    Vector slope;     ///< Mean derivative divided by the same standard deviation.
    long double phi;  ///< Pearson dispersion, or one for unit-dispersion families.
};

/// @brief Marginal mean problem after validation and fixed-design scaling.
struct Problem {
    Matrix x;                                    ///< Fixed design with scaled columns.
    const std::vector<double>& y;                ///< Response, or successes for binomial.
    Vector offset;                               ///< Link-scale offsets.
    Vector trials;                               ///< Trial counts as working values.
    std::vector<std::vector<std::size_t>> groups; ///< Row indices in each cluster.
    std::vector<std::size_t> visit;               ///< Time indices, possibly implicit.
    GEEFamily family;                             ///< Mean and variance family.
    GEECorrelation correlation;                  ///< Working correlation structure.
    std::size_t largest;                          ///< Maximum observed cluster size.
};

/// @brief Cluster score sum, its working information, and optional sandwich meat.
struct Equation {
    Vector score; ///< Total estimating score.
    Matrix bread; ///< Model-based information before multiplying by dispersion.
    Matrix meat;  ///< Sum of cluster score outer products.
};

/// @brief Converts a working statistic to a finite public number.
double output(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max())
        throw std::overflow_error("GEE: result exceeds finite reporting range");
    return static_cast<double>(value);
}

/// @brief Stable logistic probability and its independently evaluated complement.
std::pair<long double,long double> probability(long double eta) {
    const auto e = std::exp(-std::abs(eta));
    return eta >= 0 ? std::pair{1 / (1 + e), e / (1 + e)} : std::pair{e / (1 + e), 1 / (1 + e)};
}

/// @brief Checks whether the family enumeration has a defined marginal model.
bool valid_family(GEEFamily family) {
    return family == GEEFamily::Gaussian || family == GEEFamily::Binomial ||
           family == GEEFamily::Poisson || family == GEEFamily::Gamma;
}

/// @brief Returns the inner product in working coordinates.
long double dot(const Vector& a, const Vector& b) {
    return std::inner_product(a.begin(), a.end(), b.begin(), 0.0L);
}

/// @brief Computes standardized residuals and mean derivatives at fixed coefficients.
std::optional<State> evaluate(const Problem& problem, const Vector& beta) {
    const auto n = problem.x.rows();
    State state{problem.x * beta, Vector(n), Vector(n), 1};
    for (std::size_t i = 0; i < n; ++i) {
        const auto eta = state.eta[i] += problem.offset[i];
        if (!std::isfinite(eta))
            return std::nullopt;
        const long double y = problem.y[i];
        switch (problem.family) {
            case GEEFamily::Gaussian:
                state.pearson[i] = y - eta;
                state.slope[i] = 1;
                break;
            case GEEFamily::Binomial: {
                const auto [p, q] = probability(eta);
                state.slope[i] = std::sqrt(problem.trials[i] * p * q);
                state.pearson[i] = (y*q - (problem.trials[i] - y)*p) / state.slope[i];
                break;
            }
            case GEEFamily::Poisson: {
                const auto mu = std::exp(eta);
                state.slope[i] = std::sqrt(mu);
                state.pearson[i] = (y - mu) / state.slope[i];
                break;
            }
            case GEEFamily::Gamma: {
                const auto mu = std::exp(eta);
                if (!(mu > 0) || !std::isfinite(mu))
                    return std::nullopt;
                state.pearson[i] = y / mu - 1;
                state.slope[i] = 1;
                break;
            }
        }
        if (!std::isfinite(state.pearson[i]) || !(state.slope[i] > 0) || !std::isfinite(state.slope[i]))
            return std::nullopt;
    }
    if (problem.family == GEEFamily::Gaussian || problem.family == GEEFamily::Gamma) {
        state.phi = dot(state.pearson, state.pearson) / (n - problem.x.cols());
        if (!std::isfinite(state.phi))
            return std::nullopt;
    }
    return state;
}

/// @brief Computes a signed integer-lag power without rounding away the parity of the lag.
long double lag_power(long double rho, std::size_t gap) {
    const auto magnitude = std::pow(std::abs(rho), static_cast<long double>(gap));
    return rho < 0 && gap % 2 != 0 ? -magnitude : magnitude;
}

/// @brief Estimates the working correlation by matching standardized pair products.
long double estimate_correlation(const Problem& problem, const State& state, const GEEOptions& options) {
    if (options.fixed_correlation)
        return *options.fixed_correlation;
    if (problem.correlation == GEECorrelation::Independence || problem.largest < 2 || state.phi == 0)
        return 0;
    if (problem.correlation == GEECorrelation::Exchangeable) {
        long double total = 0, pairs = 0;
        for (const auto& group : problem.groups) {
            long double prefix = 0;
            for (const auto i : group) {
                total += prefix * state.pearson[i];
                prefix += state.pearson[i];
            }
            pairs += static_cast<long double>(group.size()) * (group.size() - 1) / 2;
        }
        return std::clamp(total / pairs / state.phi, -.99L / (problem.largest - 1), .99L);
    }

    /// @brief Counts and cross-product sums for one integer time lag.
    struct Moments {
        long double sum{0};   ///< Sum of standardized Pearson cross-products.
        long double count{0}; ///< Number of within-cluster pairs at the lag.
    };
    std::map<std::size_t, Moments> lags;
    for (const auto& group : problem.groups)
        for (std::size_t j = 1; j < group.size(); ++j)
            for (std::size_t i = 0; i < j; ++i) {
                const auto a = group[i], b = group[j];
                auto& moment = lags[problem.visit[b] - problem.visit[a]];
                moment.sum += state.pearson[a] * state.pearson[b];
                ++moment.count;
            }
    if (lags.size() == 1 && lags.begin()->first == 1) {
        const auto& m = lags.begin()->second;
        return std::clamp(m.sum / m.count / state.phi, -.99L, .99L);
    }
    const auto can_be_negative = std::ranges::any_of(lags, [](const auto& lag) { return lag.first % 2 != 0; });
    long double best = 0;
    long double rho = 0;
    /// @brief Scores the least-squares moment fit, omitting terms constant in rho.
    const auto consider = [&](long double parameter) {
        long double result = 0;
        for (const auto& [lag, moment] : lags) {
            const auto target = lag_power(parameter, lag);
            result += target * (2 * moment.sum / state.phi - moment.count * target);
        }
        if (result > best || (result == best && std::abs(parameter) < std::abs(rho))) {
            best = result;
            rho = parameter;
        }
    };
    /// @brief Derivative of the moment criterion divided by two.
    const auto slope = [&](long double parameter) {
        long double result = 0;
        for (const auto& [lag, moment] : lags)
            result += lag * lag_power(parameter, lag - 1) *
                (moment.sum / state.phi - moment.count * lag_power(parameter, lag));
        return result;
    };
    constexpr std::size_t intervals = 64;
    long double previous = can_be_negative ? -.99L : 0;
    auto previous_slope = slope(previous);
    consider(previous);
    for (std::size_t i = 1; i <= intervals; ++i) {
        const auto next = std::lerp(can_be_negative ? -.99L : 0, .99L,
                                    static_cast<long double>(i) / intervals);
        const auto next_slope = slope(next);
        consider(next);
        if (previous_slope > 0 && next_slope < 0) {
            auto lo = previous, hi = next;
            for (int j = 0; j < 60; ++j) {
                const auto mid = std::midpoint(lo, hi);
                if (slope(mid) > 0)
                    lo = mid;
                else
                    hi = mid;
            }
            consider(std::midpoint(lo, hi));
        }
        previous = next;
        previous_slope = next_slope;
    }
    return rho;
}

/// @brief Whitens one cluster's augmented design using its working correlation.
void whiten(Matrix& block, const Problem& problem, const std::vector<std::size_t>& group, long double rho) {
    if (rho == 0 || group.size() == 1)
        return;
    if (problem.correlation == GEECorrelation::Exchangeable) {
        const auto within = 1 / std::sqrt(1 - rho);
        const auto shared = 1 / std::sqrt(1 + (group.size() - 1) * rho);
        for (std::size_t j = 0; j < block.cols(); ++j) {
            long double average = 0;
            for (std::size_t i = 0; i < group.size(); ++i)
                average += block(i, j) / group.size();
            for (std::size_t i = 0; i < group.size(); ++i)
                block(i, j) = (block(i, j) - average) * within + average * shared;
        }
    } else if (problem.correlation == GEECorrelation::AR1) {
        // Work backwards so the preceding row is still untransformed.
        for (std::size_t i = group.size(); --i > 0;) {
            const auto lag = problem.visit[group[i]] - problem.visit[group[i-1]];
            const auto r = lag_power(rho, lag);
            const auto sd = std::sqrt((1-r) * (1+r));
            for (std::size_t j = 0; j < block.cols(); ++j)
                block(i, j) = (block(i, j) - r * block(i-1, j)) / sd;
        }
    }
}

/// @brief Assembles estimating equations from whitened rows, accumulating independent cluster scores.
Equation equations(const Problem& problem, const State& state, long double rho, bool robust) {
    const auto p = problem.x.cols();
    Equation result{Vector(p), Matrix(p,p), robust ? Matrix(p,p) : Matrix{}};
    for (const auto& group : problem.groups) {
        Matrix block(group.size(), p + 1);
        for (std::size_t i = 0; i < group.size(); ++i) {
            for (std::size_t j = 0; j < p; ++j)
                block(i, j) = state.slope[group[i]] * problem.x(group[i], j);
            block(i, p) = state.pearson[group[i]];
        }
        whiten(block, problem, group, rho);
        Vector score(p);
        for (std::size_t i = 0; i < group.size(); ++i)
            for (std::size_t j = 0; j < p; ++j) {
                score[j] += block(i,j) * block(i,p);
                for (std::size_t k = 0; k <= j; ++k)
                    result.bread(j,k) += block(i,j) * block(i,k);
            }
        for (std::size_t j = 0; j < p; ++j) {
            result.score[j] += score[j];
            if (robust)
                for (std::size_t k = 0; k <= j; ++k)
                    result.meat(j,k) += score[j] * score[k];
        }
    }
    if (robust)
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k < j; ++k)
                result.meat(k,j) = result.meat(j,k);
    return result;
}

/// @brief Inverts the working information after scaling its diagonal to one.
Matrix inverse_bread(const Matrix& information) {
    const auto p = information.rows();
    Matrix correlation(p,p);
    Vector diagonal(p);
    for (std::size_t j = 0; j < p; ++j) {
        if (!(information(j,j) > 0) || !std::isfinite(information(j,j)))
            return {};
        diagonal[j] = std::sqrt(information(j,j));
        correlation(j,j) = 1;
        for (std::size_t k = 0; k < j; ++k) {
            correlation(j,k) = information(j,k) / diagonal[j] / diagonal[k];
            if (!std::isfinite(correlation(j,k)))
                return {};
        }
    }
    const auto cholesky = linalg::cholesky(correlation);
    if (!cholesky.ok)
        return {};
    for (std::size_t j = 0; j < p; ++j)
        if (!(cholesky.L(j,j) > 8 * std::sqrt(std::numeric_limits<long double>::epsilon())))
            return {};
    auto result = cholesky.solve(Matrix::identity(p));
    for (std::size_t j = 0; j < p; ++j)
        for (std::size_t k = 0; k <= j; ++k)
            result(j,k) = result(k,j) = result(j,k) / diagonal[j] / diagonal[k];
    return result;
}

} // namespace

std::vector<double> GEEFit::predict(const linalg::DenseMatrix<double>& design, const std::vector<double>& offset) const {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (design.cols() != coefficients.size() || (!offset.empty() && offset.size() != design.rows()) ||
        !valid_family(family) || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(coefficients, finite) || !std::ranges::all_of(offset, finite))
        throw std::invalid_argument("GEE: invalid prediction shape or values");
    std::vector<double> result(design.rows());
    for (std::size_t i = 0; i < design.rows(); ++i) {
        long double eta = offset.empty() ? 0 : offset[i];
        for (std::size_t j = 0; j < design.cols(); ++j)
            eta = std::fma(static_cast<long double>(design(i,j)), coefficients[j], eta);
        const auto mean = family == GEEFamily::Gaussian ? eta :
            (family == GEEFamily::Binomial ? probability(eta).first : std::exp(eta));
        result[i] = output(mean);
    }
    return result;
}

GEEFit fit_gee(const linalg::DenseMatrix<double>& design, const std::vector<double>& response,
    const std::vector<std::size_t>& cluster, const GEEOptions& options) {
    const auto n = design.rows(), p = design.cols();
    const auto finite = [](double value) { return std::isfinite(value); };
    const bool binomial = options.family == GEEFamily::Binomial;
    const bool ar = options.correlation == GEECorrelation::AR1;
    const bool exchangeable = options.correlation == GEECorrelation::Exchangeable;
    if (p == 0 || n <= p || response.size() != n || cluster.size() != n || !valid_family(options.family) ||
        (!ar && !exchangeable && options.correlation != GEECorrelation::Independence) ||
        (!options.visit.empty() && (!ar || options.visit.size() != n)) ||
        (!options.trials.empty() && (!binomial || options.trials.size() != n)) ||
        (!options.offset.empty() && options.offset.size() != n) || options.max_iterations == 0 ||
        !(options.tolerance > 0 && options.tolerance < 1) || !std::ranges::all_of(design.data(), finite) ||
        !std::ranges::all_of(response, finite) || !std::ranges::all_of(options.offset, finite))
        throw std::invalid_argument("GEE: invalid family, correlation, dimensions, or controls");
    const auto capacity = Vector{}.max_size();
    if (p > capacity / p || p >= capacity || n > capacity / (p + 1))
        throw std::length_error("GEE: working matrix exceeds storage capacity");
    Problem problem{Matrix(n,p), response, Vector(n), Vector(n,1), {}, options.visit,
                    options.family, options.correlation, 0};
    std::map<std::size_t, std::vector<std::size_t>> groups;
    for (std::size_t i = 0; i < n; ++i)
        groups[cluster[i]].push_back(i);
    if (groups.size() < 2)
        throw std::invalid_argument("GEE: need two or more clusters");
    if (ar && problem.visit.empty())
        problem.visit.resize(n);
    for (auto& [label, group] : groups) {
        problem.largest = std::max(problem.largest, group.size());
        if (!options.visit.empty()) {
            std::ranges::sort(group, {}, [&](std::size_t i) { return options.visit[i]; });
            for (std::size_t j = 1; j < group.size(); ++j)
                if (options.visit[group[j]] == options.visit[group[j-1]])
                    throw std::invalid_argument("GEE: AR(1) visits must be distinct within each cluster");
        } else if (ar) {
            for (std::size_t j = 0; j < group.size(); ++j)
                problem.visit[group[j]] = j;
        }
        problem.groups.push_back(std::move(group));
    }
    if (options.fixed_correlation) {
        const long double rho = *options.fixed_correlation;
        const auto lower = exchangeable && problem.largest > 1 ? -1.0L / (problem.largest - 1) : -1.0L;
        if (!(rho > lower && rho < 1) || (!ar && !exchangeable && rho != 0))
            throw std::invalid_argument("GEE: fixed correlation is outside the admissible interval");
    }

    Vector rough(n), scales(p);
    for (std::size_t i = 0; i < n; ++i) {
        const double y = response[i];
        if (!options.offset.empty())
            problem.offset[i] = options.offset[i];
        const auto trials = options.trials.empty() ? std::size_t{1} : options.trials[i];
        if (binomial || options.family == GEEFamily::Poisson) {
            if (y < 0 || std::trunc(y) != y || (binomial && (trials == 0 || y > static_cast<long double>(trials))))
                throw std::invalid_argument("GEE: counts outside their admissible domain");
            problem.trials[i] = trials;
            rough[i] = binomial ? std::log(y + .5L) - std::log(problem.trials[i] - y + .5L) : std::log(y + .5L);
        } else if (options.family == GEEFamily::Gamma) {
            if (!(y > 0))
                throw std::invalid_argument("GEE: Gamma responses must be positive");
            rough[i] = std::log(static_cast<long double>(y));
        } else {
            rough[i] = y;
        }
        rough[i] -= problem.offset[i];
    }
    for (std::size_t j = 0; j < p; ++j) {
        for (std::size_t i = 0; i < n; ++i)
            scales[j] = std::max(scales[j], std::abs(static_cast<long double>(design(i,j))));
        if (scales[j] == 0)
            throw std::invalid_argument("GEE: zero fixed design column");
        for (std::size_t i = 0; i < n; ++i)
            problem.x(i,j) = design(i,j) / scales[j];
    }
    const auto factor = linalg::svd(problem.x);
    if (!factor.converged || factor.rank() != p)
        throw std::invalid_argument("GEE: rank-deficient fixed design");

    auto beta = factor.solve(rough);
    auto state = evaluate(problem, beta);
    if (!state) {
        beta = Vector(p);
        state = evaluate(problem, beta);
    }
    if (!state)
        throw std::runtime_error("GEE: no usable initial mean");
    auto status = GEEStatus::IterationLimit;
    std::size_t iterations = 0;
    long double rho = 0;
    for (;;) {
        rho = estimate_correlation(problem, *state, options);
        const auto current = equations(problem, *state, rho, false);
        const auto inverse = inverse_bread(current.bread);
        if (inverse.rows() == 0) {
            status = GEEStatus::NumericalFailure;
            break;
        }
        const auto direction = inverse * current.score;
        const auto effect = problem.x * direction;
        long double change = 0;
        for (std::size_t i = 0; i < n; ++i)
            change = std::max(change, std::abs(effect[i]) / (1 + std::abs(state->eta[i] - problem.offset[i])));
        if (change <= options.tolerance && std::ranges::all_of(effect, [](auto v) { return std::isfinite(v); })) {
            status = GEEStatus::Converged;
            break;
        }
        if (iterations == options.max_iterations)
            break;

        const auto previous = dot(current.score, direction);
        bool accepted = false;
        for (long double step = 1; step >= std::ldexp(1.L,-50); step /= 2) {
            auto trial = beta;
            for (std::size_t j = 0; j < p; ++j)
                trial[j] += step * direction[j];
            if (trial == beta)
                break;
            auto next = evaluate(problem, trial);
            if (!next)
                continue;
            // Hold the current working correlation and metric fixed while
            // testing whether this step brings the estimating score toward zero.
            const auto score = equations(problem, *next, rho, false).score;
            if (dot(score, inverse * score) < previous) {
                beta = std::move(trial);
                state = std::move(next);
                ++iterations;
                accepted = true;
                break;
            }
        }
        if (!accepted) {
            status = GEEStatus::NumericalFailure;
            break;
        }
    }

    GEEFit result{options.family, {}, {}, output(rho), output(state->phi), {}, {}, status, iterations};
    for (std::size_t j = 0; j < p; ++j)
        result.coefficients.push_back(output(beta[j] / scales[j]));
    result.fitted = result.predict(design, options.offset);
    const auto final = equations(problem, *state, rho, true);
    const auto covariance = inverse_bread(final.bread);
    if (covariance.rows() != 0) {
        const auto robust = covariance * final.meat * covariance;
        result.model_covariance = linalg::DenseMatrix<double>(p,p);
        result.sandwich_covariance = linalg::DenseMatrix<double>(p,p);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t k = 0; k <= j; ++k) {
                result.model_covariance(j,k) = result.model_covariance(k,j) =
                    output(state->phi * covariance(j,k) / scales[j] / scales[k]);
                result.sandwich_covariance(j,k) = result.sandwich_covariance(k,j) =
                    output(.5L * (robust(j,k) + robust(k,j)) / scales[j] / scales[k]);
            }
    }
    return result;
}

} // namespace datamunge::stats
