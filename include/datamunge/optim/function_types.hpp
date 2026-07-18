#pragma once

#include <cstddef>
#include <vector>

namespace datamunge::optim {

/// @brief The most general objective function: only supports evaluation. Optimizers that
///        consume this type (e.g. SimulatedAnnealing) are derivative-free.
///
///        This is a director-enabled extension point: subclass it directly (in C++, or, via
///        SWIG directors, in any supported language) to define a custom objective.
class ArbitraryFunction {
public:
    virtual ~ArbitraryFunction() = default;

    /// @brief Evaluates the objective at @p coordinates.
    virtual double evaluate(const std::vector<double>& coordinates) = 0;
};

/// @brief An objective function that additionally supports exact gradient evaluation.
///        Optimizers that consume this type (GradientDescent, Adam, LBFGS) use the full
///        gradient at every step.
class DifferentiableFunction : public ArbitraryFunction {
public:
    /// @brief Returns the gradient of the objective at @p coordinates.
    virtual std::vector<double> gradient(const std::vector<double>& coordinates) = 0;
};

/// @brief An objective that decomposes as a sum over num_functions() independent terms
///        (e.g. one term per training example). evaluate() defaults to summing every term,
///        but remains overridable for a more efficient full-objective computation.
class SeparableFunction : public ArbitraryFunction {
public:
    /// @brief The number of additive terms the objective decomposes into.
    [[nodiscard]] virtual std::size_t num_functions() const = 0;

    /// @brief Evaluates just term @p i of the objective at @p coordinates.
    virtual double evaluate_term(const std::vector<double>& coordinates, std::size_t i) = 0;

    double evaluate(const std::vector<double>& coordinates) override;
};

/// @brief A function that is both differentiable and separable -- the common case for
///        machine learning loss functions (a sum of per-example losses). Optimizers that
///        consume this type (SGD) can update coordinates from a single term's gradient
///        without evaluating the full objective.
class DifferentiableSeparableFunction : public DifferentiableFunction {
public:
    /// @brief The number of additive terms the objective decomposes into.
    [[nodiscard]] virtual std::size_t num_functions() const = 0;

    /// @brief Evaluates just term @p i of the objective at @p coordinates.
    virtual double evaluate_term(const std::vector<double>& coordinates, std::size_t i) = 0;

    /// @brief Returns the gradient of just term @p i of the objective at @p coordinates.
    virtual std::vector<double> gradient_term(const std::vector<double>& coordinates, std::size_t i) = 0;

    double evaluate(const std::vector<double>& coordinates) override;
    std::vector<double> gradient(const std::vector<double>& coordinates) override;
};

} // namespace datamunge::optim
