#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim { struct TrustRegionNewtonOptions { double initial_radius{1.0}; double max_radius{1000.0}; std::size_t max_iterations{200}; double tolerance{1e-8}; }; class TrustRegionNewton { public: explicit TrustRegionNewton(TrustRegionNewtonOptions options = {}); double optimize(HessianFunction& function,std::vector<double>& coordinates) const; private: TrustRegionNewtonOptions options_; }; }
