#include "datamunge/nn/optim.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

namespace datamunge::nn {

std::string optimizer_name(OptimizerKind kind) {
  switch (kind) {
    case OptimizerKind::Sgd: return "SGD";
    case OptimizerKind::Momentum: return "Momentum";
    case OptimizerKind::Nesterov: return "Nesterov";
    case OptimizerKind::Adagrad: return "Adagrad";
    case OptimizerKind::RMSProp: return "RMSProp";
    case OptimizerKind::Adam: return "Adam";
    case OptimizerKind::AdamW: return "AdamW";
    case OptimizerKind::Nadam: return "Nadam";
    case OptimizerKind::AMSGrad: return "AMSGrad";
    case OptimizerKind::Adamax: return "Adamax";
    case OptimizerKind::Lamb: return "LAMB";
    case OptimizerKind::Lars: return "LARS";
    case OptimizerKind::Lion: return "Lion";
    case OptimizerKind::Adafactor: return "Adafactor";
  }
  throw std::invalid_argument("unknown optimizer");
}

void OptimizerConfig::validate() const {
  (void)optimizer_name(kind);
  for (double b : {momentum, beta1, beta2, rho})
    if (!std::isfinite(b) || b < 0 || b >= 1) throw std::invalid_argument("optimizer decay factors must be in [0, 1)");
  if (!std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(initial_accumulator) || initial_accumulator < 0 ||
      !std::isfinite(weight_decay) || weight_decay < 0)
    throw std::invalid_argument("invalid optimizer epsilon, accumulator or weight decay");
  for (double x : {trust_coefficient, factor_epsilon, parameter_epsilon, update_clip, decay_exponent})
    if (!std::isfinite(x) || x <= 0) throw std::invalid_argument("optimizer scales must be positive");
  if (decay_exponent > 1) throw std::invalid_argument("Adafactor decay exponent must be <= 1");
  if (weight_decay != 0 && kind != OptimizerKind::AdamW && kind != OptimizerKind::Lamb &&
      kind != OptimizerKind::Lars && kind != OptimizerKind::Lion)
    throw std::invalid_argument("weight decay is only used by AdamW, LAMB, LARS and Lion; add an L2 term to the loss for " +
                                optimizer_name(kind));
}

Optimizer::Optimizer(std::vector<Tensor> parameters, double learning_rate, OptimizerConfig config)
    : params_(std::move(parameters)), lr_(learning_rate), config_(config) {
  config_.validate();
  if (params_.empty()) throw std::invalid_argument("Optimizer: no parameters");
  for (const auto& p : params_)
    if (!p.defined() || !p.requires_grad() || !p.is_leaf())
      throw std::invalid_argument("Optimizer: parameters must be leaf tensors that require grad");
  set_learning_rate(learning_rate);
}

Optimizer::Optimizer(const ParameterStore& store, double learning_rate, OptimizerConfig config)
    : Optimizer(store.parameters(), learning_rate, config) {}

void Optimizer::set_learning_rate(double learning_rate) {
  if (!std::isfinite(learning_rate) || learning_rate < 0)
    throw std::invalid_argument("Optimizer: learning rate must be finite and >= 0");
  lr_ = learning_rate;
}

Tensor& Optimizer::slot(const std::string& name, std::size_t index, const Shape& shape, double initial) {
  const std::string key = name + "/" + std::to_string(index);
  auto              it  = slot_index_.find(key);
  if (it != slot_index_.end()) return slots_[it->second].second;
  slot_index_[key] = slots_.size();
  slots_.emplace_back(key, full(shape, initial, params_[index].dtype()));
  return slots_.back().second;
}

namespace {

Tensor l2_norm(const Tensor& x) { return sqrt(sum(square(x))); }
Tensor rms(const Tensor& x) { return sqrt(mean(square(x))); }

/// numerator / denominator where both are positive, else 1 (LAMB/LARS trust ratio).
double trust(double numerator, double denominator) {
  return numerator > 0 && denominator > 0 ? numerator / denominator : 1.0;
}

}  // namespace

void Optimizer::step(const std::vector<Tensor>& gradients) {
  if (gradients.size() != params_.size())
    throw std::invalid_argument("Optimizer::step: got " + std::to_string(gradients.size()) + " gradients for " +
                                std::to_string(params_.size()) + " parameters");
  NoGradGuard  guard;
  const double t  = static_cast<double>(step_ + 1);
  const double c1 = 1 - std::pow(config_.beta1, t);
  const double c2 = 1 - std::pow(config_.beta2, t);
  double       mu = 0, mu_next = 0, product = 0, product_next = 0;
  if (config_.kind == OptimizerKind::Nadam) {
    mu           = config_.beta1 * (1 - 0.5 * std::pow(0.96, t));
    mu_next      = config_.beta1 * (1 - 0.5 * std::pow(0.96, t + 1));
    product      = nadam_product_ * mu;
    product_next = product * mu_next;
  }

  // Compute every update before writing any, so a parameter shared between
  // two entries sees consistent values (synchronous semantics, as in TF).
  std::vector<Tensor> updates(params_.size());
  for (std::size_t i = 0; i < params_.size(); ++i) {
    if (!gradients[i].defined()) continue;
    const Tensor w = params_[i].detach();
    const Tensor g = cast(gradients[i].detach(), w.dtype());
    if (g.shape() != w.shape())
      throw std::invalid_argument("Optimizer::step: gradient " + std::to_string(i) + " has shape " +
                                  shape_str(g.shape()) + ", parameter has " + shape_str(w.shape()));
    Tensor direction = g;
    double rate      = lr_;
    switch (config_.kind) {
      case OptimizerKind::Sgd: break;
      case OptimizerKind::Momentum:
      case OptimizerKind::Nesterov: {
        Tensor& velocity = slot("velocity", i, w.shape());
        velocity         = add(mul(velocity, config_.momentum), g);
        direction = config_.kind == OptimizerKind::Nesterov ? add(g, mul(velocity, config_.momentum)) : velocity;
        break;
      }
      case OptimizerKind::Adagrad: {
        Tensor& accumulator = slot("accumulator", i, w.shape(), config_.initial_accumulator);
        accumulator         = add(accumulator, square(g));
        direction           = div(g, add(sqrt(accumulator), config_.epsilon));
        break;
      }
      case OptimizerKind::RMSProp: {
        Tensor& variance = slot("variance", i, w.shape());
        variance         = add(mul(variance, config_.rho), mul(square(g), 1 - config_.rho));
        direction        = div(g, sqrt(add(variance, config_.epsilon)));
        break;
      }
      case OptimizerKind::Lars: {
        const double wn    = l2_norm(w).item();
        const double gn    = l2_norm(g).item();
        const double local = config_.trust_coefficient * trust(wn, gn + config_.weight_decay * wn);
        Tensor&      velocity = slot("velocity", i, w.shape());
        // Original LARS keeps the learning rate inside the velocity.
        velocity  = add(mul(velocity, config_.momentum), mul(add(g, mul(w, config_.weight_decay)), lr_ * local));
        direction = velocity;
        rate      = 1.0;
        break;
      }
      case OptimizerKind::Lion: {
        Tensor&      m   = slot("m", i, w.shape());
        const Tensor old = m;
        direction = add(sign(add(mul(old, config_.beta1), mul(g, 1 - config_.beta1))), mul(w, config_.weight_decay));
        m         = add(mul(old, config_.beta2), mul(g, 1 - config_.beta2));
        break;
      }
      case OptimizerKind::Adafactor: {
        const double decay   = 1 - std::pow(t, -config_.decay_exponent);
        const Tensor squared = add(square(g), config_.factor_epsilon);
        Tensor       variance;
        if (w.dim() >= 2) {
          // Rank > 2 tensors are treated as a matrix over the last axis.
          const std::int64_t columns = w.shape().back();
          if (columns == 0 || w.numel() == 0) throw std::invalid_argument("Adafactor requires nonempty parameters");
          const std::int64_t rows   = w.numel() / columns;
          const Tensor       matrix = reshape(squared, {rows, columns});
          Tensor&            row    = slot("row", i, {rows});
          Tensor&            col    = slot("col", i, {columns});
          row = add(mul(row, decay), mul(mean(matrix, {1}), 1 - decay));
          col = add(mul(col, decay), mul(mean(matrix, {0}), 1 - decay));
          const Tensor normalized_row = div(row, maximum(mean(row), scalar_like(config_.factor_epsilon, row)));
          variance                    = reshape(outer(normalized_row, col), w.shape());
        } else {
          Tensor& v = slot("v", i, w.shape());
          v         = add(mul(v, decay), mul(squared, 1 - decay));
          variance  = v;
        }
        direction = div(g, sqrt(maximum(variance, scalar_like(config_.factor_epsilon, variance))));
        direction = div(direction, std::max(1.0, rms(direction).item() / config_.update_clip));
        if (config_.relative_step) rate = std::min(rate, 1.0 / std::sqrt(t));
        if (config_.scale_parameter) rate *= std::max(rms(w).item(), config_.parameter_epsilon);
        break;
      }
      default: {  // the Adam family
        Tensor& m      = slot("m", i, w.shape());
        m              = add(mul(m, config_.beta1), mul(g, 1 - config_.beta1));
        const Tensor m_hat = div(m, c1);
        if (config_.kind == OptimizerKind::Adamax) {
          Tensor& u = slot("u", i, w.shape());
          u         = maximum(mul(u, config_.beta2), abs(g));
          direction = div(m_hat, add(u, config_.epsilon));
          break;
        }
        Tensor& v = slot("v", i, w.shape());
        v         = add(mul(v, config_.beta2), mul(square(g), 1 - config_.beta2));
        Tensor denom_v = v;
        if (config_.kind == OptimizerKind::AMSGrad) {
          Tensor& vmax = slot("vmax", i, w.shape());
          vmax         = maximum(vmax, v);
          denom_v      = vmax;
        }
        Tensor numerator = m_hat;
        if (config_.kind == OptimizerKind::Nadam)
          numerator = add(div(mul(m, mu_next), 1 - product_next), div(mul(g, 1 - mu), 1 - product));
        direction = div(numerator, add(sqrt(div(denom_v, c2)), config_.epsilon));
        if (config_.kind == OptimizerKind::AdamW || config_.kind == OptimizerKind::Lamb)
          direction = add(direction, mul(w, config_.weight_decay));
        if (config_.kind == OptimizerKind::Lamb) direction = mul(direction, trust(l2_norm(w).item(), l2_norm(direction).item()));
        break;
      }
    }
    updates[i] = mul(direction, rate);
  }
  for (std::size_t i = 0; i < params_.size(); ++i)
    if (updates[i].defined()) params_[i].sub_(updates[i]);
  if (config_.kind == OptimizerKind::Nadam) nadam_product_ = product;
  ++step_;
}

void Optimizer::step() {
  std::vector<Tensor> grads;
  for (const auto& p : params_) grads.push_back(p.grad());
  step(grads);
}

void Optimizer::minimize(const Tensor& loss) {
  GradOptions options;
  options.allow_unused = true;
  step(grad({loss}, params_, {}, options));
}

void Optimizer::zero_grad() {
  for (auto& p : params_) p.zero_grad();
}

std::vector<std::pair<std::string, Tensor>> Optimizer::state() const {
  std::vector<std::pair<std::string, Tensor>> out;
  out.emplace_back("step", scalar(static_cast<double>(step_), DType::Int64));
  out.emplace_back("nadam_product", scalar(nadam_product_, DType::Float64));
  for (const auto& [name, value] : slots_) out.emplace_back(name, value.clone());
  return out;
}

void Optimizer::load_state(const std::vector<std::pair<std::string, Tensor>>& state) {
  slots_.clear();
  slot_index_.clear();
  for (const auto& [name, value] : state) {
    if (name == "step") {
      step_ = value.item<std::int64_t>();
    } else if (name == "nadam_product") {
      nadam_product_ = value.item();
    } else {
      const auto slash = name.rfind('/');
      if (slash == std::string::npos) throw std::invalid_argument("Optimizer::load_state: bad slot name '" + name + "'");
      const auto index = static_cast<std::size_t>(std::stoul(name.substr(slash + 1)));
      if (index >= params_.size()) throw std::invalid_argument("Optimizer::load_state: slot '" + name + "' out of range");
      slot_index_[name] = slots_.size();
      slots_.emplace_back(name, cast(value, params_[index].dtype()).clone());
    }
  }
}

// ---- gradient utilities -------------------------------------------------------------------

double global_norm(const std::vector<Tensor>& tensors) {
  NoGradGuard guard;
  double      total = 0;
  for (const auto& t : tensors)
    if (t.defined()) total += sum(square(cast(t.detach(), DType::Float64))).item();
  return std::sqrt(total);
}

std::vector<Tensor> clip_by_value(const std::vector<Tensor>& gradients, double low, double high) {
  std::vector<Tensor> out;
  for (const auto& g : gradients) out.push_back(g.defined() ? clip(g, low, high) : g);
  return out;
}

Tensor clip_by_norm(const Tensor& gradient, double max_norm) {
  if (!(max_norm > 0)) throw std::invalid_argument("clip_by_norm: max_norm must be positive");
  const double n = global_norm({gradient});
  return n > max_norm ? mul(gradient, max_norm / n) : gradient;
}

std::pair<std::vector<Tensor>, double> clip_by_global_norm(const std::vector<Tensor>& gradients, double max_norm) {
  if (!(max_norm > 0)) throw std::invalid_argument("clip_by_global_norm: max_norm must be positive");
  const double        norm  = global_norm(gradients);
  const double        scale = max_norm / std::max(norm, max_norm);
  std::vector<Tensor> out;
  for (const auto& g : gradients) out.push_back(g.defined() && scale != 1.0 ? mul(g, scale) : g);
  return {out, norm};
}

void GradientAccumulator::add(const std::vector<Tensor>& gradients, double weight) {
  if (!std::isfinite(weight) || weight <= 0)
    throw std::invalid_argument("GradientAccumulator: weight must be positive and finite");
  if (count_ > 0 && gradients.size() != sums_.size())
    throw std::invalid_argument("GradientAccumulator: gradient count changed between add() calls");
  NoGradGuard guard;
  auto        scaled = [&](const Tensor& g) { return weight == 1.0 ? g.detach().clone() : mul(g.detach(), weight); };
  if (count_ == 0) {
    sums_.clear();
    for (const auto& g : gradients) sums_.push_back(g.defined() ? scaled(g) : Tensor());
  } else {
    for (std::size_t i = 0; i < gradients.size(); ++i) {
      if (!gradients[i].defined()) continue;
      sums_[i] = sums_[i].defined() ? nn::add(sums_[i], scaled(gradients[i])) : scaled(gradients[i]);
    }
  }
  ++count_;
  weight_ += weight;
}

std::vector<Tensor> GradientAccumulator::sum() const { return sums_; }

std::vector<Tensor> GradientAccumulator::mean() const {
  if (count_ == 0) throw std::logic_error("GradientAccumulator: nothing accumulated");
  NoGradGuard         guard;
  std::vector<Tensor> out;
  for (const auto& s : sums_) out.push_back(s.defined() ? div(s, weight_) : s);
  return out;
}

void GradientAccumulator::reset() {
  sums_.clear();
  count_  = 0;
  weight_ = 0.0;
}

}  // namespace datamunge::nn
