#pragma once

/// @file autodiff_model.hpp
/// @brief Adapters that turn C++ lambdas into the function objects the samplers take.

#include <datamunge/autodiff/tape.hpp>
#include <datamunge/optim/function_types.hpp>

#include <functional>
#include <vector>

/// @brief Bayesian inference: models, samplers and diagnostics.
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
    /// @brief Signature of the wrapped log density: tape and parameter leaves to a Var.
    using LogDensityFn = std::function<autodiff::Var(autodiff::Tape&, const std::vector<autodiff::Var>&)>;

    /// @brief Wraps a log density written in terms of autodiff::Var.
    /// @param log_density_fn Function of the tape and the parameter leaves.
    explicit AutodiffModel(LogDensityFn log_density_fn);

    /// @brief Evaluates the log density.
    /// @param params Parameter values.
    /// @return The log density.
    double evaluate(const std::vector<double>& params) override;
    /// @brief Gradient of the log density by reverse-mode automatic differentiation.
    /// @param params Parameter values.
    /// @return The gradient.
    std::vector<double> gradient(const std::vector<double>& params) override;

private:
    LogDensityFn log_density_fn_; ///< The wrapped log density.
};

/// @brief C++-only adapter: a plain `double(const std::vector<double>&)` log density as an
///        optim::ArbitraryFunction, for the gradient-free samplers (RandomWalkMetropolis,
///        GibbsSampler, ImportanceSampling) and grid_posterior(). Use it for log densities
///        that are not differentiable, such as a Laplace likelihood.
class LogDensity : public optim::ArbitraryFunction {
public:
    /// @brief Signature of the wrapped log density.
    using Fn = std::function<double(const std::vector<double>&)>;

    /// @brief Wraps @p fn.
    /// @param fn Log density up to an additive constant.
    explicit LogDensity(Fn fn);

    /// @brief Evaluates the wrapped function.
    /// @param params Parameter values.
    /// @return The log density.
    double evaluate(const std::vector<double>& params) override;

private:
    Fn fn_; ///< The wrapped log density.
};

} // namespace datamunge::bayes
