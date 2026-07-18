#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::bayes {

struct MAPOptions {
    std::size_t max_iterations{500};
    double tolerance{1e-8};
    std::size_t history_size{10};
};

/// @brief Maximum a posteriori estimation: finds the mode of a log-posterior density via
///        L-BFGS (datamunge::optim::LBFGS), analogous to Stan's `optimizing()`.
class MAP {
public:
    explicit MAP(MAPOptions options = {});

    /// @brief Maximizes @p log_posterior starting from @p coordinates, updating it in place
    ///        to the mode found, and returns the log-posterior density there.
    double optimize(optim::DifferentiableFunction& log_posterior, std::vector<double>& coordinates) const;

private:
    MAPOptions options_;
};

} // namespace datamunge::bayes
