/// @file optimizer_analysis.cpp
/// @brief State traces of tensor optimizers and analytic scalar-mode stability.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/nn/optimizer_analysis.hpp>
#include <stdexcept>
namespace datamunge::nn {
/// @brief Private validation and state extraction for deterministic optimizer experiments.
namespace {
/// @brief Restricts experiments to the three velocity-based first-order rules.
/// @param config Configuration to validate.
void require_rule(const OptimizerConfig& config) {
  config.validate();
  if (config.kind != OptimizerKind::Sgd && config.kind != OptimizerKind::Momentum &&
      config.kind != OptimizerKind::Nesterov)
    throw std::invalid_argument("optimizer analysis: use SGD, Momentum or Nesterov");
}
/// @brief Checks finite vector entries without changing the vector.
/// @param x Vector to check.
void require_finite(const std::vector<double>& x) {
  for (double value : x)
    if (!std::isfinite(value))
      throw std::invalid_argument("optimizer analysis: finite input required");
}
/// @brief Copies the current optimizer's first velocity slot, or zeros before it exists.
/// @param optimizer Read-only optimizer over one vector leaf.
/// @param size Coordinate count.
/// @return Owned state vector.
std::vector<double> velocity(const Optimizer& optimizer, std::size_t size) {
  for (const auto& [name, value] : optimizer.state())
    if (name == "velocity/0")
      return value.to_vector<double>();
  return std::vector<double>(size, 0);
}
/// @brief Rejects unstable trajectories that have overflowed a finite representation.
/// @param values Current tensor values.
void require_state(const std::vector<double>& values) {
  for (double value : values)
    if (!std::isfinite(value))
      throw std::overflow_error("optimizer analysis: nonfinite trajectory");
}
} // namespace
dstruct::DataFrame optimizer_gradient_trace(
    const std::vector<double>&         initial,
    const linalg::DenseMatrix<double>& gradients,
    const std::vector<double>&         rates,
    OptimizerConfig                    config) {
  require_rule(config);
  require_finite(initial);
  require_finite(rates);
  if (initial.empty() || gradients.cols() != initial.size() ||
      (rates.size() != 1 && rates.size() != gradients.rows()) || rates.empty())
    throw std::invalid_argument("optimizer_gradient_trace: invalid shape/rates");
  for (double value : rates)
    if (value < 0)
      throw std::invalid_argument("optimizer_gradient_trace: negative rate");
  auto                w = from_vector(initial).requires_grad_();
  Optimizer           opt({w}, rates.front(), config);
  std::vector<double> step, coordinate, rate, gradient, before, after, vel;
  for (std::size_t i = 0; i < gradients.rows(); ++i) {
    const auto g = gradients.row(i);
    require_finite(g);
    const auto   old = w.to_vector<double>();
    const double lr  = rates.size() == 1 ? rates.front() : rates[i];
    opt.set_learning_rate(lr);
    opt.step({from_vector(g)});
    const auto now = w.to_vector<double>(), v = velocity(opt, initial.size());
    require_state(now);
    require_state(v);
    for (std::size_t j = 0; j < initial.size(); ++j) {
      step.push_back(i + 1);
      coordinate.push_back(j);
      rate.push_back(lr);
      gradient.push_back(g[j]);
      before.push_back(old[j]);
      after.push_back(now[j]);
      vel.push_back(v[j]);
    }
  }
  dstruct::DataFrame out;
  out.add_column("step", step);
  out.add_column("coordinate", coordinate);
  out.add_column("rate", rate);
  out.add_column("gradient", gradient);
  out.add_column("before", before);
  out.add_column("after", after);
  out.add_column("velocity", vel);
  return out;
}
dstruct::DataFrame optimizer_quadratic_trace(
    const std::vector<double>& curvature,
    const std::vector<double>& initial,
    std::size_t                steps,
    double                     rate,
    OptimizerConfig            config) {
  require_rule(config);
  require_finite(curvature);
  require_finite(initial);
  if (curvature.empty() || curvature.size() != initial.size() || !std::isfinite(rate) || rate < 0)
    throw std::invalid_argument("optimizer_quadratic_trace: invalid vectors/rate");
  for (double value : curvature)
    if (value <= 0)
      throw std::invalid_argument("optimizer_quadratic_trace: positive curvature required");
  auto                w = from_vector(initial).requires_grad_();
  const auto          h = from_vector(curvature);
  Optimizer           opt({w}, rate, config);
  std::vector<double> step, coordinate, weight, gradient, vel, loss;
  for (std::size_t i = 0;; ++i) {
    const auto   objective = sum(h * square(w)) * .5;
    const double value     = objective.item<double>();
    if (!std::isfinite(value))
      throw std::overflow_error("optimizer_quadratic_trace: nonfinite objective");
    const auto now = w.to_vector<double>(), v = velocity(opt, initial.size());
    require_state(now);
    require_state(v);
    for (std::size_t j = 0; j < initial.size(); ++j) {
      step.push_back(i);
      coordinate.push_back(j);
      weight.push_back(now[j]);
      gradient.push_back(curvature[j] * now[j]);
      vel.push_back(v[j]);
      loss.push_back(value);
    }
    if (i == steps)
      break;
    opt.minimize(objective);
  }
  dstruct::DataFrame out;
  out.add_column("step", step);
  out.add_column("coordinate", coordinate);
  out.add_column("weight", weight);
  out.add_column("gradient", gradient);
  out.add_column("velocity", vel);
  out.add_column("loss", loss);
  return out;
}
double optimizer_mode_radius(OptimizerKind kind, double a, double momentum) {
  OptimizerConfig config;
  config.kind     = kind;
  config.momentum = momentum;
  require_rule(config);
  if (!std::isfinite(a) || a < 0)
    throw std::invalid_argument(
        "optimizer_mode_radius: finite nonnegative rate*curvature required");
  if (kind == OptimizerKind::Sgd)
    return std::abs(1 - a);
  const long double mu = momentum, alpha = a;
  const long double trace =
      kind == OptimizerKind::Momentum ? 1 + mu - alpha : (1 + mu) * (1 - alpha);
  const long double determinant = kind == OptimizerKind::Momentum ? mu : mu * (1 - alpha),
                    disc        = trace * trace - 4 * determinant;
  if (disc < 0)
    return static_cast<double>(std::sqrt(determinant));
  return static_cast<double>((std::abs(trace) + std::sqrt(disc)) / 2);
}
} // namespace datamunge::nn
