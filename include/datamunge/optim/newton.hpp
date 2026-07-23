#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim { struct NewtonOptions { std::size_t max_iterations{100}; double tolerance{1e-8}; double damping{1e-10}; }; class Newton { public: explicit Newton(NewtonOptions options = {}); double optimize(HessianFunction& function, std::vector<double>& coordinates) const; private: NewtonOptions options_; }; }
