#include <datamunge/optim/nelder_mead.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::optim {
namespace {

void validate(const NelderMeadOptions& options) {
    if (options.initial_simplex_scale <= 0.0)
        throw std::invalid_argument("NelderMead: initial_simplex_scale must be positive");
    if (options.reflection <= 0.0) throw std::invalid_argument("NelderMead: reflection must be positive");
    if (options.expansion <= 1.0) throw std::invalid_argument("NelderMead: expansion must be greater than 1");
    if (options.contraction <= 0.0 || options.contraction >= 1.0)
        throw std::invalid_argument("NelderMead: contraction must be in (0, 1)");
    if (options.shrink <= 0.0 || options.shrink >= 1.0)
        throw std::invalid_argument("NelderMead: shrink must be in (0, 1)");
    if (options.tolerance < 0.0) throw std::invalid_argument("NelderMead: tolerance must be non-negative");
}

} // namespace

NelderMead::NelderMead(NelderMeadOptions options) : options_(options) {}

double NelderMead::optimize(ArbitraryFunction& function, std::vector<double>& coordinates) const {
    validate(options_);
    const std::size_t dimension = coordinates.size();
    if (dimension == 0) return function.evaluate(coordinates);

    std::vector<std::vector<double>> simplex(dimension + 1, coordinates);
    for (std::size_t j = 0; j < dimension; ++j) {
        const double scale = options_.initial_simplex_scale * std::max(1.0, std::abs(coordinates[j]));
        simplex[j + 1][j] += scale;
    }
    std::vector<double> values(dimension + 1);
    for (std::size_t i = 0; i < simplex.size(); ++i) values[i] = function.evaluate(simplex[i]);

    std::vector<std::size_t> order(dimension + 1);
    for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&values](const std::size_t a, const std::size_t b) {
            return values[a] < values[b];
        });
        const std::size_t best = order.front();
        const std::size_t worst = order.back();
        const std::size_t second_worst = order[dimension - 1];

        double max_distance = 0.0;
        for (const auto& point : simplex) {
            double distance_sq = 0.0;
            for (std::size_t j = 0; j < dimension; ++j) {
                const double delta = point[j] - simplex[best][j];
                distance_sq += delta * delta;
            }
            max_distance = std::max(max_distance, std::sqrt(distance_sq));
        }
        if (max_distance <= options_.tolerance && values[worst] - values[best] <= options_.tolerance) break;

        std::vector<double> centroid(dimension, 0.0);
        for (std::size_t rank = 0; rank < dimension; ++rank) {
            for (std::size_t j = 0; j < dimension; ++j) centroid[j] += simplex[order[rank]][j];
        }
        for (double& component : centroid) component /= static_cast<double>(dimension);

        auto point_from_centroid = [&](const double factor) {
            std::vector<double> point(dimension);
            for (std::size_t j = 0; j < dimension; ++j)
                point[j] = centroid[j] + factor * (centroid[j] - simplex[worst][j]);
            return point;
        };
        auto reflected = point_from_centroid(options_.reflection);
        const double reflected_value = function.evaluate(reflected);
        if (reflected_value < values[best]) {
            auto expanded = point_from_centroid(options_.expansion);
            const double expanded_value = function.evaluate(expanded);
            if (expanded_value < reflected_value) {
                simplex[worst] = std::move(expanded);
                values[worst] = expanded_value;
            } else {
                simplex[worst] = std::move(reflected);
                values[worst] = reflected_value;
            }
            continue;
        }
        if (reflected_value < values[second_worst]) {
            simplex[worst] = std::move(reflected);
            values[worst] = reflected_value;
            continue;
        }

        const bool outside = reflected_value < values[worst];
        std::vector<double> contracted(dimension);
        for (std::size_t j = 0; j < dimension; ++j) {
            const double direction = outside ? reflected[j] - centroid[j] : simplex[worst][j] - centroid[j];
            contracted[j] = centroid[j] + options_.contraction * direction;
        }
        const double contracted_value = function.evaluate(contracted);
        if (contracted_value < (outside ? reflected_value : values[worst])) {
            simplex[worst] = std::move(contracted);
            values[worst] = contracted_value;
            continue;
        }

        for (std::size_t rank = 1; rank <= dimension; ++rank) {
            const std::size_t i = order[rank];
            for (std::size_t j = 0; j < dimension; ++j)
                simplex[i][j] = simplex[best][j] + options_.shrink * (simplex[i][j] - simplex[best][j]);
            values[i] = function.evaluate(simplex[i]);
        }
    }

    const auto best = static_cast<std::size_t>(std::min_element(values.begin(), values.end()) - values.begin());
    coordinates = simplex[best];
    return values[best];
}

} // namespace datamunge::optim
