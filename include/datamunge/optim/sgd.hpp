#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct SGDOptions {
    double step_size{0.01};
    std::size_t max_epochs{100};
    std::size_t batch_size{1};
    /// @brief Stops when the full-objective change between epochs drops below this.
    double tolerance{1e-8};
    bool shuffle{true};
    std::uint64_t seed{42};
};

/// @brief Stochastic (mini-batch) gradient descent on a DifferentiableSeparableFunction: each
///        step updates coordinates from the average gradient of one shuffled mini-batch of
///        terms, rather than the full objective's gradient.
class SGD {
public:
    explicit SGD(SGDOptions options = {});

    double optimize(DifferentiableSeparableFunction& function, std::vector<double>& coordinates) const;

private:
    SGDOptions options_;
};

} // namespace datamunge::optim
