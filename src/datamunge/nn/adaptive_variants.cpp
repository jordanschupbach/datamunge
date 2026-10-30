/// @file adaptive_variants.cpp
/// @brief RAdam, AdaBelief and Yogi, RAdam's rectification and the Adam step bound.

#include "datamunge/nn/adaptive_variants.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

std::string adaptive_rule_name(AdaptiveRule rule) {
  switch (rule) {
    case AdaptiveRule::RAdam: return "RAdam";
    case AdaptiveRule::AdaBelief: return "AdaBelief";
    case AdaptiveRule::Yogi: return "Yogi";
  }
  throw std::invalid_argument("adaptive_rule_name: unknown rule");
}

void AdaptiveConfig::validate() const {
  (void)adaptive_rule_name(rule);
  if (!(beta1 >= 0 && beta1 < 1)) throw std::invalid_argument("AdaptiveConfig: beta1 must be in [0, 1)");
  if (!(beta2 > 0 && beta2 < 1)) throw std::invalid_argument("AdaptiveConfig: beta2 must be in (0, 1)");
  if (!(epsilon > 0) || !std::isfinite(epsilon)) throw std::invalid_argument("AdaptiveConfig: epsilon must be positive");
  if (!(weight_decay >= 0) || !std::isfinite(weight_decay)) throw std::invalid_argument("AdaptiveConfig: weight_decay must be >= 0");
}

AdaptiveOptimizer::AdaptiveOptimizer(std::vector<Tensor> parameters, double learning_rate, AdaptiveConfig config)
    : params_(std::move(parameters)), lr_(learning_rate), config_(config) {
  config_.validate();
  if (params_.empty()) throw std::invalid_argument("AdaptiveOptimizer: no parameters");
  for (const auto& p : params_)
    if (!p.defined() || !p.requires_grad() || !p.is_leaf())
      throw std::invalid_argument("AdaptiveOptimizer: parameters must be leaf tensors that require grad");
  set_learning_rate(learning_rate);
  m_.resize(params_.size()), v_.resize(params_.size());
}

AdaptiveOptimizer::AdaptiveOptimizer(const ParameterStore& store, double learning_rate, AdaptiveConfig config)
    : AdaptiveOptimizer(store.parameters(), learning_rate, config) {}

void AdaptiveOptimizer::set_learning_rate(double learning_rate) {
  if (!std::isfinite(learning_rate) || learning_rate < 0)
    throw std::invalid_argument("AdaptiveOptimizer: learning rate must be finite and >= 0");
  lr_ = learning_rate;
}

void AdaptiveOptimizer::step(const std::vector<Tensor>& gradients) {
  if (gradients.size() != params_.size()) throw std::invalid_argument("AdaptiveOptimizer::step: one gradient per parameter");
  NoGradGuard guard;
  ++step_;
  const double t = static_cast<double>(step_);
  const double b1 = config_.beta1, b2 = config_.beta2, eps = config_.epsilon;
  const double c1 = 1 - std::pow(b1, t), c2 = 1 - std::pow(b2, t);
  const double r = config_.rule == AdaptiveRule::RAdam ? radam_rectification(step_, b2) : 1.0;
  for (std::size_t i = 0; i < params_.size(); ++i) {
    const Tensor& g0 = gradients[i];
    if (!g0.defined()) continue;
    Tensor& w = params_[i];
    if (g0.shape() != w.shape()) throw std::invalid_argument("AdaptiveOptimizer::step: gradient shape mismatch");
    const Tensor g = cast(g0.detach(), w.dtype());
    if (!m_[i].defined()) m_[i] = zeros(w.shape(), w.dtype()), v_[i] = zeros(w.shape(), w.dtype());
    Tensor m = b1 * m_[i] + (1 - b1) * g, v;
    switch (config_.rule) {
      case AdaptiveRule::RAdam: v = b2 * v_[i] + (1 - b2) * square(g); break;
      case AdaptiveRule::AdaBelief: v = b2 * v_[i] + (1 - b2) * square(g - m) + eps; break;
      case AdaptiveRule::Yogi: {
        const Tensor g2 = square(g);
        v = v_[i] - (1 - b2) * sign(v_[i] - g2) * g2;
        break;
      }
    }
    Tensor direction;
    if (config_.rule == AdaptiveRule::RAdam && r == 0.0) direction = m / c1;
    else direction = r * (m / c1) / (sqrt(v / c2) + eps);
    Tensor next = w.detach();
    if (config_.weight_decay > 0) next = next - lr_ * config_.weight_decay * next;
    next = next - lr_ * direction;
    w.copy_(next);
    m_[i] = m, v_[i] = v;
  }
}

void AdaptiveOptimizer::minimize(const Tensor& loss) {
  GradOptions options;
  options.allow_unused = true;
  step(grad({loss}, params_, {}, options));
}

std::vector<std::pair<std::string, Tensor>> AdaptiveOptimizer::state() const {
  std::vector<std::pair<std::string, Tensor>> out;
  out.emplace_back("step", scalar(static_cast<double>(step_), DType::Int64));
  for (std::size_t i = 0; i < params_.size(); ++i) {
    if (m_[i].defined()) out.emplace_back("m/" + std::to_string(i), m_[i].clone());
    if (v_[i].defined()) out.emplace_back("v/" + std::to_string(i), v_[i].clone());
  }
  return out;
}

double radam_rho(std::int64_t t, double beta2) {
  if (t < 1) throw std::invalid_argument("radam_rho: t must be at least 1");
  if (!(beta2 > 0 && beta2 < 1)) throw std::invalid_argument("radam_rho: beta2 must be in (0, 1)");
  const double rho_inf = 2 / (1 - beta2) - 1, bt = std::pow(beta2, static_cast<double>(t));
  return rho_inf - 2 * static_cast<double>(t) * bt / (1 - bt);
}

double radam_rectification(std::int64_t t, double beta2) {
  const double rho = radam_rho(t, beta2), rho_inf = 2 / (1 - beta2) - 1;
  if (rho <= 5) return 0.0;
  return std::sqrt((rho - 4) * (rho - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho));
}

double adam_step_bound(double beta1, double beta2, std::int64_t t) {
  if (!(beta1 >= 0 && beta1 < 1) || !(beta2 > 0 && beta2 < 1) || t < 1)
    throw std::invalid_argument("adam_step_bound: need beta1 in [0, 1), beta2 in (0, 1) and t >= 1");
  const double q = beta1 * beta1 / beta2, n = static_cast<double>(t);
  const double sum = std::abs(q - 1) < 1e-15 ? n : (1 - std::pow(q, n)) / (1 - q);
  return (1 - beta1) * std::sqrt(1 - std::pow(beta2, n)) / ((1 - std::pow(beta1, n)) * std::sqrt(1 - beta2)) * std::sqrt(sum);
}

}  // namespace datamunge::nn
