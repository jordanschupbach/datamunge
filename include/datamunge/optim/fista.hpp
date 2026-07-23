#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim {
struct FISTAOptions { double step_size{0.01}; std::size_t max_iterations{10000}; double tolerance{1e-8}; };
class FISTA { public: explicit FISTA(FISTAOptions options = {}); double optimize(ProximalFunction& function, std::vector<double>& coordinates) const; private: FISTAOptions options_; };
}
