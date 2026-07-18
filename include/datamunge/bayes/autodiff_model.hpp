#pragma once

#include <datamunge/autodiff/tape.hpp>
#include <datamunge/optim/function_types.hpp>

#include <functional>
#include <vector>

namespace datamunge::bayes {

/// @brief C++-only convenience: wraps a log-density function written declaratively in terms
///        of autodiff::Var (typically composed from the distributions.hpp log-density
///        helpers, exactly the way a Stan model block is written in terms of distribution
///        statements) as an optim::DifferentiableFunction, ready to hand to MAP/HMC/NUTS.
///
///        Every call builds a fresh Tape, wraps @p params as leaf Vars, evaluates the
///        supplied function, and (for gradient()) reads it off via Tape::backward() -- the
///        same pattern as optim::gradient_reverse(). Because it holds a std::function over
///        autodiff::Var (a type that, unlike its SWIG-bound Var/Tape facade counterparts,
///        cannot cross the language boundary), this class is C++-only: cross-language
///        callers instead subclass optim::DifferentiableFunction directly, supplying a
///        hand-written gradient, exactly as they would for GradientDescent or LBFGS.
class AutodiffModel : public optim::DifferentiableFunction {
public:
    using LogDensityFn = std::function<autodiff::Var(autodiff::Tape&, const std::vector<autodiff::Var>&)>;

    explicit AutodiffModel(LogDensityFn log_density_fn);

    double evaluate(const std::vector<double>& params) override;
    std::vector<double> gradient(const std::vector<double>& params) override;

private:
    LogDensityFn log_density_fn_;
};

} // namespace datamunge::bayes
