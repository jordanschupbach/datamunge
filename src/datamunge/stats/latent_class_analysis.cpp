/// @file latent_class_analysis.cpp
/// @brief Grouped observed-data EM for categorical latent class models.

#include <datamunge/stats/latent_class_analysis.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

namespace datamunge::stats {
/// @brief Observation grouping and latent class EM workspaces.
namespace {

/// @brief Working probability matrix with extended precision for log likelihood evaluations.
using Matrix = linalg::DenseMatrix<long double>;
/// @brief Sentinel for an absent response or an entirely missing response pattern.
constexpr auto missing = std::numeric_limits<std::size_t>::max();

/// @brief Unique observed response patterns and their counts, in first-seen order.
struct ResponsePatterns {
    std::vector<std::size_t> offsets;              ///< Start and end of each item's flattened category segment.
    std::vector<std::vector<std::size_t>> values;  ///< Pattern rows of flattened category indices or missing.
    std::vector<std::size_t> counts;               ///< Number of input rows matching each pattern.
    std::vector<std::size_t> row_pattern;          ///< Pattern index per input row, or missing for all-missing rows.
    std::size_t informative{};                    ///< Total number of rows with at least one response.
};

/// @brief Current parameters, posterior and likelihood for one EM start.
struct State {
    std::vector<long double> proportions; ///< Mixing probabilities.
    Matrix items;                        ///< Class by flattened category probabilities.
    Matrix posterior;                    ///< Pattern by class responsibilities given these parameters.
    long double log_likelihood{};        ///< Observed log likelihood of the grouped responses.
    std::size_t iterations{};            ///< Completed updates at this state.
    bool converged{};                    ///< Whether the likelihood change met tolerance.
};

/// @brief Checks that matrix dimensions fit DenseMatrix storage before multiplying them.
template<typename T>
void check_storage(std::size_t rows, std::size_t cols) {
    if (cols != 0 && rows > std::vector<T>{}.max_size() / cols)
        throw std::length_error("latent class analysis: dense workspace exceeds storage capacity");
}

/// @brief Validates category codes and indexes repeated informative response patterns.
ResponsePatterns group_responses(const linalg::DenseMatrix<std::int32_t>& responses,
    const std::vector<std::size_t>& levels) {
    ResponsePatterns result;
    result.offsets.push_back(0);
    for (const auto count : levels) {
        if (count < 2 || count - 1 > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
            throw std::invalid_argument("latent class analysis: invalid number of response levels");
        if (count > missing - result.offsets.back())
            throw std::length_error("latent class analysis: too many response levels");
        result.offsets.push_back(result.offsets.back() + count);
    }

    std::vector<bool> observed(levels.size(), false);
    std::map<std::vector<std::size_t>,std::size_t> index;
    for (std::size_t i = 0; i < responses.rows(); ++i) {
        std::vector<std::size_t> row(levels.size(), missing);
        bool informative = false;
        for (std::size_t j = 0; j < levels.size(); ++j) {
            const auto value = responses(i,j);
            if (value == -1)
                continue;
            if (value < 0 || static_cast<std::size_t>(value) >= levels[j])
                throw std::invalid_argument("latent class analysis: response is outside its declared levels");
            observed[j] = informative = true;
            row[j] = result.offsets[j] + static_cast<std::size_t>(value);
        }
        if (!informative) {
            result.row_pattern.push_back(missing);
            continue;
        }
        const auto [entry, inserted] = index.try_emplace(row, result.values.size());
        if (inserted) {
            result.values.push_back(std::move(row));
            result.counts.push_back(0);
        }
        ++result.counts[entry->second];
        ++result.informative;
        result.row_pattern.push_back(entry->second);
    }
    if (!std::ranges::all_of(observed, [](bool present) { return present; }))
        throw std::invalid_argument("latent class analysis: every item requires observed responses");
    return result;
}

/// @brief One starting distribution with positive random scores normalized within each item.
State initialize(const ResponsePatterns& data, std::size_t classes, std::mt19937_64& random) {
    State result{std::vector<long double>(classes, 1.0L / classes),
        Matrix(classes, data.offsets.back()), Matrix(data.values.size(), classes)};
    std::exponential_distribution<long double> rate(1);
    for (std::size_t k = 0; k < classes; ++k)
        for (std::size_t j = 1; j < data.offsets.size(); ++j) {
            const auto first = data.offsets[j-1], last = data.offsets[j];
            long double total = 0;
            for (std::size_t c = first; c < last; ++c) {
                result.items(k,c) = rate(random) + std::numeric_limits<long double>::epsilon();
                total += result.items(k,c);
            }
            for (std::size_t c = first; c < last; ++c)
                result.items(k,c) /= total;
        }
    return result;
}

/// @brief Log of a probability, with an exact zero mapped to minus infinity.
long double log_probability(long double p) {
    return p > 0 ? std::log(p) : -std::numeric_limits<long double>::infinity();
}

/// @brief Calculates responsibilities and likelihood with a log-sum-exp normalization.
void expectation(const ResponsePatterns& data, State& state) {
    const auto classes = state.proportions.size(), categories = state.items.cols();
    std::vector<long double> priors(classes);
    Matrix log_items(classes, categories);
    std::ranges::transform(state.proportions, priors.begin(), log_probability);
    std::ranges::transform(state.items.data(), log_items.data().begin(), log_probability);
    state.log_likelihood = 0;
    for (std::size_t i = 0; i < data.values.size(); ++i) {
        long double peak = -std::numeric_limits<long double>::infinity();
        for (std::size_t k = 0; k < classes; ++k) {
            long double score = priors[k];
            for (const auto response : data.values[i])
                if (response != missing)
                    score += log_items(k,response);
            state.posterior(i,k) = score;
            peak = std::max(peak, score);
        }
        if (!std::isfinite(peak)) {
            state.log_likelihood = peak;
            return;
        }
        long double sum = 0;
        for (std::size_t k = 0; k < classes; ++k) {
            auto& probability = state.posterior(i,k);
            probability = std::exp(probability - peak);
            sum += probability;
        }
        for (std::size_t k = 0; k < classes; ++k)
            state.posterior(i,k) /= sum;
        state.log_likelihood += data.counts[i] * (peak + std::log(sum));
    }
}

/// @brief Updates proportions and item probabilities using weighted observed counts.
State maximize(const ResponsePatterns& data, const State& current) {
    const auto classes = current.proportions.size(), categories = current.items.cols();
    State next{std::vector<long double>(classes), Matrix(classes, categories),
        Matrix(data.values.size(), classes)};
    for (std::size_t i = 0; i < data.values.size(); ++i)
        for (std::size_t k = 0; k < classes; ++k) {
            const auto weight = data.counts[i] * current.posterior(i,k);
            next.proportions[k] += weight;
            for (const auto response : data.values[i])
                if (response != missing)
                    next.items(k,response) += weight;
        }
    for (std::size_t k = 0; k < classes; ++k) {
        next.proportions[k] /= data.informative;
        for (std::size_t j = 1; j < data.offsets.size(); ++j) {
            const auto first = data.offsets[j-1], last = data.offsets[j];
            long double total = 0;
            for (std::size_t c = first; c < last; ++c)
                total += next.items(k,c);
            for (std::size_t c = first; c < last; ++c)
                next.items(k,c) = total > 0 ? next.items(k,c) / total : current.items(k,c);
        }
    }
    return next;
}

/// @brief Runs one EM start while retaining the last finite, monotonically improving state.
State fit_start(const ResponsePatterns& data, const LatentClassOptions& options, std::mt19937_64& random) {
    auto current = initialize(data, options.classes, random);
    expectation(data, current);
    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        auto next = maximize(data,current);
        expectation(data,next);
        const auto improvement = next.log_likelihood - current.log_likelihood;
        const auto scale = 1 + std::abs(current.log_likelihood);
        if (!std::isfinite(improvement))
            break;
        if (improvement < 0) {
            current.converged = -improvement <= std::min(static_cast<long double>(options.relative_tolerance),
                64 * std::numeric_limits<long double>::epsilon()) * scale;
            break;
        }
        next.iterations = iteration + 1;
        next.converged = improvement <= options.relative_tolerance * scale;
        current = std::move(next);
        if (current.converged)
            break;
    }
    return current;
}

/// @brief Restores original row order and permutes every result by decreasing class proportion.
LatentClassFit format_fit(const ResponsePatterns& data, const State& state) {
    const auto classes = state.proportions.size();
    std::vector<std::size_t> order(classes);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return state.proportions[a] > state.proportions[b];
    });
    LatentClassFit result{std::vector<double>(classes), {},
        linalg::DenseMatrix<double>(data.row_pattern.size(), classes),
        static_cast<double>(state.log_likelihood), state.iterations, state.converged};
    for (std::size_t k = 0; k < classes; ++k)
        result.class_probabilities[k] = static_cast<double>(state.proportions[order[k]]);
    for (std::size_t j = 1; j < data.offsets.size(); ++j) {
        const auto first = data.offsets[j-1], count = data.offsets[j] - first;
        auto& item = result.item_probabilities.emplace_back(classes, count);
        for (std::size_t k = 0; k < classes; ++k)
            for (std::size_t c = 0; c < count; ++c)
                item(k,c) = static_cast<double>(state.items(order[k], first+c));
    }
    for (std::size_t i = 0; i < data.row_pattern.size(); ++i)
        for (std::size_t k = 0; k < classes; ++k) {
            const auto pattern = data.row_pattern[i];
            result.posterior(i,k) = pattern == missing ? result.class_probabilities[k]
                : static_cast<double>(state.posterior(pattern,order[k]));
        }
    return result;
}

} // namespace

LatentClassFit fit_latent_class_analysis(const linalg::DenseMatrix<std::int32_t>& responses,
    const std::vector<std::size_t>& levels, const LatentClassOptions& options) {
    if (responses.rows() == 0 || responses.cols() == 0 || responses.cols() != levels.size() ||
        options.classes == 0 || options.starts == 0 || options.max_iterations == 0 ||
        !(options.relative_tolerance > 0 && options.relative_tolerance < 1))
        throw std::invalid_argument("latent class analysis: invalid dimensions or EM controls");
    const auto data = group_responses(responses, levels);
    if (options.classes > data.informative)
        throw std::invalid_argument("latent class analysis: classes exceed informative rows");
    check_storage<double>(responses.rows(), options.classes);
    check_storage<long double>(data.values.size(), options.classes);
    check_storage<long double>(options.classes, data.offsets.back());

    std::mt19937_64 random(options.seed);
    State best;
    best.log_likelihood = -std::numeric_limits<long double>::infinity();
    for (std::size_t start = 0; start < options.starts; ++start) {
        auto candidate = fit_start(data, options, random);
        if (candidate.log_likelihood > best.log_likelihood ||
            (candidate.log_likelihood == best.log_likelihood && candidate.converged))
            best = std::move(candidate);
        if (options.classes == 1)
            break;
    }
    if (!std::isfinite(best.log_likelihood))
        throw std::runtime_error("latent class analysis: no finite observed likelihood");
    return format_fit(data,best);
}

} // namespace datamunge::stats
