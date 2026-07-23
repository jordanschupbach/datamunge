#pragma once
#include <datamunge/optim/function_types.hpp>
#include <cstdint>
namespace datamunge::optim {
/// @brief Surrogate/acquisition extension point for bounded Bayesian optimization.
class BayesianSurrogate { public: virtual ~BayesianSurrogate()=default; virtual void fit(const std::vector<std::vector<double>>& points,const std::vector<double>& values)=0; virtual double acquisition(const std::vector<double>& point,double incumbent)=0; };
struct BayesianOptimizationOptions { std::size_t initial_samples{8}; std::size_t max_iterations{50}; std::uint64_t seed{42}; };
class BayesianOptimization { public: explicit BayesianOptimization(BayesianOptimizationOptions options = {}); double optimize(ArbitraryFunction& function,std::vector<double>& coordinates,const std::vector<double>& lower,const std::vector<double>& upper,BayesianSurrogate& surrogate) const; private: BayesianOptimizationOptions options_; };
}
