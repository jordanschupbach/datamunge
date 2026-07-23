#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim { struct InteriorPointOptions { double step_size{0.01}; double initial_barrier{1.0}; double barrier_decay{0.1}; std::size_t max_outer_iterations{20}; std::size_t inner_iterations{200}; }; class InteriorPoint { public: explicit InteriorPoint(InteriorPointOptions options = {}); double optimize(InequalityConstrainedFunction& function,std::vector<double>& coordinates) const; private: InteriorPointOptions options_; }; }
