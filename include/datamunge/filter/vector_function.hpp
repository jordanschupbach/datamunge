#pragma once

#include <vector>

namespace datamunge::filter {

/// @brief A vector-valued function R^n -> R^m: the state-transition or observation model
///        supplied to this module's nonlinear filters (ExtendedKalmanFilter,
///        UnscentedKalmanFilter, EnsembleKalmanFilter, ParticleFilter). Director-enabled
///        (subclass directly in C++, or via SWIG directors in any bound language) so this
///        module never needs to know the concrete form of the caller's dynamics.
class VectorFunction {
  public:
    virtual ~VectorFunction() = default;

    /// @brief Evaluates the function at @p x.
    virtual std::vector<double> evaluate(const std::vector<double>& x) = 0;
};

} // namespace datamunge::filter
