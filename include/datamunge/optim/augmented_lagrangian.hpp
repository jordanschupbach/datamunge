#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim { struct AugmentedLagrangianOptions { double step_size{0.01}; double initial_penalty{1.0}; std::size_t max_outer_iterations{50}; std::size_t inner_iterations{200}; double tolerance{1e-8}; }; class AugmentedLagrangian { public: explicit AugmentedLagrangian(AugmentedLagrangianOptions options = {}); double optimize(EqualityConstrainedFunction& function,std::vector<double>& coordinates) const; private: AugmentedLagrangianOptions options_; }; }
