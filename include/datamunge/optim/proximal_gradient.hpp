#pragma once
#include <datamunge/optim/function_types.hpp>
#include <cstddef>
namespace datamunge::optim {
struct ProximalGradientOptions { double step_size{0.01}; std::size_t max_iterations{10000}; double tolerance{1e-8}; };
/// @brief Proximal gradient descent for composite objectives such as lasso.
class ProximalGradient { public: explicit ProximalGradient(ProximalGradientOptions options = {}); double optimize(ProximalFunction& function, std::vector<double>& coordinates) const; private: ProximalGradientOptions options_; };
} // namespace datamunge::optim
