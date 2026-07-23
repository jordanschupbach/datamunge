#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim { struct LevenbergMarquardtOptions { double initial_damping{1e-3}; std::size_t max_iterations{200}; double tolerance{1e-8}; }; class LevenbergMarquardt { public: explicit LevenbergMarquardt(LevenbergMarquardtOptions options = {}); double optimize(ResidualFunction& function,std::vector<double>& coordinates) const; private: LevenbergMarquardtOptions options_; }; }
