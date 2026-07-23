#pragma once
#include <datamunge/optim/function_types.hpp>
namespace datamunge::optim { struct SQPOptions { double step_size{1.0}; double regularization{1e-6}; std::size_t max_iterations{200}; double tolerance{1e-8}; }; class SQP { public: explicit SQP(SQPOptions options = {}); double optimize(EqualityConstrainedFunction& function,std::vector<double>& coordinates) const; private: SQPOptions options_; }; }
