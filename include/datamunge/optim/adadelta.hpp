#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>

namespace datamunge::optim {

struct AdaDeltaOptions {
    double decay_rate{0.95};
    double epsilon{1e-6};
    std::size_t max_iterations{10000};
    double tolerance{1e-8};
};

/// @brief Full-batch AdaDelta (Zeiler 2012) on a DifferentiableFunction.  AdaDelta rescales
///        gradients by the ratio of running RMS parameter updates to running RMS gradients,
///        avoiding a separate global learning-rate parameter.
class AdaDelta {
public:
    explicit AdaDelta(AdaDeltaOptions options = {});
    double optimize(DifferentiableFunction& function, std::vector<double>& coordinates) const;
private:
    AdaDeltaOptions options_;
};

} // namespace datamunge::optim
