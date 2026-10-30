/// @file optim.cpp
/// @brief Stateful tensor optimizer rules, fused update kernels, and gradient aggregation/clipping.

#include "datamunge/nn/optim.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"
#include "detail.hpp"

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
    case OptimizerKind::Adadelta: return "Adadelta";
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

const char* Optimizer::slot_name(SlotKind kind) {
  static_assert(static_cast<std::size_t>(SlotKind::Count) == 11);
  /// @brief Checkpoint slot-name table indexed by real SlotKind values, excluding Count.
  static constexpr const char* names[] = {
      "velocity", "accumulator", "variance", "accum", "accum_update", "m", "u", "v", "vmax", "row", "col"};
  return names[static_cast<std::size_t>(kind)];
}

Tensor& Optimizer::slot(SlotKind kind, std::size_t index, const Shape& shape, double initial) {
  if (slot_index_.empty()) slot_index_.resize(params_.size());
  auto& position = slot_index_[index][static_cast<std::size_t>(kind)];
  if (position != 0) return slots_[position - 1].second;
  slots_.emplace_back(std::string(slot_name(kind)) + "/" + std::to_string(index),
                      full(shape, initial, params_[index].dtype()));
  position = slots_.size();
  return slots_.back().second;
}

/// @brief Norms and packed update kernels used without recording optimizer operations.
namespace {

/// @brief Returns the scalar L2 norm as a tensor, using the currently active grad mode.
Tensor l2_norm(const Tensor& x) { return sqrt(sum(square(x))); }
/// @brief Returns the scalar root-mean-square value as a tensor.
Tensor rms(const Tensor& x) { return sqrt(mean(square(x))); }

/// numerator / denominator where both are positive, else 1 (LAMB/LARS trust ratio).
double trust(double numerator, double denominator) {
  return numerator > 0 && denominator > 0 ? numerator / denominator : 1.0;
}

/// @brief Produces a packed SGD/Adam/AdamW decrement and updates supplied moment slots in place.
/// Preserves staged scalar rounding while avoiding intermediate tensors. Parameter
/// writes occur in a later pass to preserve synchronous semantics for aliased inputs.
/// @tparam T Matching scalar dtype of contiguous weights, gradients, and non-null slots.
/// @tparam Kind SGD, Adam, or AdamW; Adam variants require both first/second moment slots.
template<typename T, OptimizerKind Kind>
Tensor fused_basic_update(
    const Tensor&          w,
    const Tensor&          g,
    Tensor*                first,
    Tensor*                second,
    double                 learning_rate,
    const OptimizerConfig& config,
    double                 c1,
    double                 c2) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  Tensor   update   = empty(w.shape(), w.dtype());
  T*       out      = update.data_ptr<T>();
  const T* gradient = g.data_ptr<T>();
  const T* weight   = w.data_ptr<T>();
  T*       m        = first ? first->data_ptr<T>() : nullptr;
  T*       v        = second ? second->data_ptr<T>() : nullptr;
  const T  rate     = static_cast<T>(learning_rate);
  const T  beta1 = static_cast<T>(config.beta1), beta2 = static_cast<T>(config.beta2);
  const T  one_minus_beta1 = static_cast<T>(1 - config.beta1),
          one_minus_beta2  = static_cast<T>(1 - config.beta2);
  const T correction1 = static_cast<T>(c1), correction2 = static_cast<T>(c2);
  const T epsilon = static_cast<T>(config.epsilon), decay = static_cast<T>(config.weight_decay);
  detail::parallel_elements(w.numel(), [&](std::int64_t i) {
    if constexpr (Kind == OptimizerKind::Sgd) {
      out[i] = gradient[i] * rate;
    } else {
      const T moment_old    = m[i] * beta1;
      const T moment_grad   = gradient[i] * one_minus_beta1;
      const T moment        = moment_old + moment_grad;
      const T grad_squared  = gradient[i] * gradient[i];
      const T variance_old  = v[i] * beta2;
      const T variance_grad = grad_squared * one_minus_beta2;
      const T variance      = variance_old + variance_grad;
      m[i]                  = moment;
      v[i]                  = variance;
      const T mean          = moment / correction1;
      const T variance_hat  = variance / correction2;
      const T root          = std::sqrt(variance_hat);
      const T denom         = root + epsilon;
      T       direction     = mean / denom;
      if constexpr (Kind == OptimizerKind::AdamW) {
        const T decayed = weight[i] * decay;
        direction       = direction + decayed;
      }
      out[i] = direction * rate;
    }
  });
  if (first)
    ++first->impl()->storage->version;
  if (second)
    ++second->impl()->storage->version;
  return update;
}

/// @brief Produces packed decrements for momentum, adaptive-history, Lion, and scheduled Adam variants.
/// Updates the non-null slots in place; callers provide the rule-specific slot layout,
/// matching packed shapes/dtypes, and moment/schedule correction factors.
template<typename T, OptimizerKind Kind>
Tensor fused_other_update(
    const Tensor&                 w,
    const Tensor&                 g,
    const std::array<Tensor*, 3>& slots,
    double                        learning_rate,
    const OptimizerConfig&        config,
    double                        c1,
    double                        c2,
    double                        mu,
    double                        mu_next,
    double                        product,
    double                        product_next) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  Tensor   update   = empty(w.shape(), w.dtype());
  const T* gradient = g.data_ptr<T>();
  const T* weight   = w.data_ptr<T>();
  T*       out      = update.data_ptr<T>();
  T*       a        = slots[0] ? slots[0]->data_ptr<T>() : nullptr;
  T*       b        = slots[1] ? slots[1]->data_ptr<T>() : nullptr;
  T*       c        = slots[2] ? slots[2]->data_ptr<T>() : nullptr;
  const T  rate = static_cast<T>(learning_rate), momentum = static_cast<T>(config.momentum);
  const T  beta1 = static_cast<T>(config.beta1), beta2 = static_cast<T>(config.beta2);
  const T  beta1c = static_cast<T>(1 - config.beta1), beta2c = static_cast<T>(1 - config.beta2);
  const T  rho = static_cast<T>(config.rho), rhoc = static_cast<T>(1 - config.rho);
  const T  eps = static_cast<T>(config.epsilon), decay = static_cast<T>(config.weight_decay);
  const T  correction1 = static_cast<T>(c1), correction2 = static_cast<T>(c2);
  const T  next_mu = static_cast<T>(mu_next), current_mu_c = static_cast<T>(1 - mu);
  const T  product_c     = static_cast<T>(1 - product),
          next_product_c = static_cast<T>(1 - product_next);
  detail::parallel_elements(w.numel(), [&](std::int64_t i) {
    const T value = gradient[i];
    T       direction;
    if constexpr (Kind == OptimizerKind::Momentum || Kind == OptimizerKind::Nesterov) {
      const T old      = a[i] * momentum;
      const T velocity = old + value;
      a[i]             = velocity;
      if constexpr (Kind == OptimizerKind::Nesterov) {
        const T ahead = velocity * momentum;
        direction     = value + ahead;
      } else
        direction = velocity;
    } else if constexpr (Kind == OptimizerKind::Adagrad) {
      const T squared     = value * value;
      const T accumulator = a[i] + squared;
      a[i]                = accumulator;
      const T root        = std::sqrt(accumulator);
      const T denom       = root + eps;
      direction           = value / denom;
    } else if constexpr (Kind == OptimizerKind::RMSProp || Kind == OptimizerKind::Adadelta) {
      const T squared     = value * value;
      const T history     = a[i] * rho;
      const T incoming    = squared * rhoc;
      const T accumulator = history + incoming;
      a[i]                = accumulator;
      const T regularized = accumulator + eps;
      const T root        = std::sqrt(regularized);
      if constexpr (Kind == OptimizerKind::RMSProp)
        direction = value / root;
      else {
        const T regularized_update = b[i] + eps;
        const T numerator          = std::sqrt(regularized_update);
        const T ratio              = numerator / root;
        direction                  = ratio * value;
        const T old_update         = b[i] * rho;
        const T direction_squared  = direction * direction;
        const T new_update         = direction_squared * rhoc;
        b[i]                       = old_update + new_update;
      }
    } else if constexpr (Kind == OptimizerKind::Lion) {
      const T old          = a[i];
      const T left         = old * beta1;
      const T right        = value * beta1c;
      const T interpolated = left + right;
      const T sign         = static_cast<T>((interpolated > 0) - (interpolated < 0));
      const T decayed      = weight[i] * decay;
      direction            = sign + decayed;
      const T history      = old * beta2;
      const T incoming     = value * beta2c;
      a[i]                 = history + incoming;
    } else {
      const T history        = a[i] * beta1;
      const T incoming       = value * beta1c;
      const T mean           = history + incoming;
      a[i]                   = mean;
      const T corrected_mean = mean / correction1;
      if constexpr (Kind == OptimizerKind::Adamax) {
        const T old       = b[i] * beta2;
        const T magnitude = value < 0 ? -value : value;
        const T maximum   = old >= magnitude ? old : magnitude;
        b[i]              = maximum;
        const T denom     = maximum + eps;
        direction         = corrected_mean / denom;
      } else {
        const T squared           = value * value;
        const T old               = b[i] * beta2;
        const T incoming_variance = squared * beta2c;
        const T variance          = old + incoming_variance;
        b[i]                      = variance;
        T denom_v                 = variance;
        if constexpr (Kind == OptimizerKind::AMSGrad) {
          denom_v = c[i] >= variance ? c[i] : variance;
          c[i]    = denom_v;
        }
        T numerator = corrected_mean;
        if constexpr (Kind == OptimizerKind::Nadam) {
          const T history_part       = mean * next_mu;
          const T history_corrected  = history_part / next_product_c;
          const T gradient_part      = value * current_mu_c;
          const T gradient_corrected = gradient_part / product_c;
          numerator                  = history_corrected + gradient_corrected;
        }
        const T corrected_variance = denom_v / correction2;
        const T root               = std::sqrt(corrected_variance);
        const T denom              = root + eps;
        direction                  = numerator / denom;
      }
    }
    out[i] = direction * rate;
  });
  for (auto* slot : slots)
    if (slot)
      ++slot->impl()->storage->version;
  return update;
}

/// @brief Updates LARS velocity from a packed gradient plus weight decay and returns its decrement.
/// The supplied rate already includes the layer's local trust coefficient and ratio.
template<typename T>
Tensor fused_lars_update(
    const Tensor&          w,
    const Tensor&          g,
    Tensor&                velocity,
    double                 rate,
    const OptimizerConfig& config) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  Tensor   update   = empty(w.shape(), w.dtype());
  T*       out      = update.data_ptr<T>();
  T*       v        = velocity.data_ptr<T>();
  const T* weight   = w.data_ptr<T>();
  const T* gradient = g.data_ptr<T>();
  const T  momentum = static_cast<T>(config.momentum), decay = static_cast<T>(config.weight_decay),
          learning_rate = static_cast<T>(rate);
  detail::parallel_elements(w.numel(), [&](std::int64_t i) {
    const T decayed   = weight[i] * decay;
    const T direction = gradient[i] + decayed;
    const T incoming  = direction * learning_rate;
    const T old       = v[i] * momentum;
    v[i]              = old + incoming;
    out[i]            = v[i];
  });
  ++velocity.impl()->storage->version;
  return update;
}

/// @brief Divides each packed update element by divisor, then multiplies by multiplier in dtype T.
/// The caller supplies nonzero divisor; storage version is incremented after mutation.
template<typename T>
void scale_fused_update(Tensor& update, double divisor, double multiplier) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  T*      values = update.data_ptr<T>();
  const T d = static_cast<T>(divisor), m = static_cast<T>(multiplier);
  detail::parallel_elements(update.numel(), [&](std::int64_t i) {
    const T divided = values[i] / d;
    values[i]       = divided * m;
  });
  ++update.impl()->storage->version;
}

/// @brief Forms an AdamW direction, scales by its LAMB trust ratio and rate, and updates moments.
template<typename T>
Tensor fused_lamb_update(
    const Tensor&          w,
    const Tensor&          g,
    Tensor&                m,
    Tensor&                v,
    double                 rate,
    const OptimizerConfig& config,
    double                 c1,
    double                 c2) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  Tensor  update = fused_basic_update<T, OptimizerKind::AdamW>(w, g, &m, &v, 1.0, config, c1, c2);
  const T ratio  = static_cast<T>(trust(l2_norm(w).item(), l2_norm(update).item()));
  const T learning_rate = static_cast<T>(rate);
  T*      values        = update.data_ptr<T>();
  detail::parallel_elements(update.numel(), [&](std::int64_t i) {
    const T scaled = values[i] * ratio;
    values[i]      = scaled * learning_rate;
  });
  ++update.impl()->storage->version;
  return update;
}

/// @brief Forms a clipped Adafactor decrement and updates factored or unfactored variance slots.
/// Non-null second selects row/column factoring; time is the positive one-based step.
template<typename T>
Tensor fused_adafactor_update(
    const Tensor&          w,
    const Tensor&          g,
    Tensor&                first,
    Tensor*                second,
    double                 rate,
    const OptimizerConfig& config,
    double                 time) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  Tensor       update      = empty(w.shape(), w.dtype());
  T*           out         = update.data_ptr<T>();
  const T*     gradient    = g.data_ptr<T>();
  const double decay_value = 1 - std::pow(time, -config.decay_exponent);
  const T      decay = static_cast<T>(decay_value), incoming = static_cast<T>(1 - decay_value),
          epsilon = static_cast<T>(config.factor_epsilon);
  T* row          = first.data_ptr<T>();
  if (second) {
    const auto columns = w.shape().back(), rows = w.numel() / columns;
    detail::parallel_elements(w.numel(), [&](std::int64_t i) {
      const T squared = gradient[i] * gradient[i];
      out[i]          = squared + epsilon;
    });
    const Tensor matrix = reshape(update, {rows, columns});
    {
      const Tensor means  = mean(matrix, {1});
      const T*     values = means.data_ptr<T>();
      detail::parallel_elements(rows, [&](std::int64_t i) {
        const T old  = row[i] * decay;
        const T next = values[i] * incoming;
        row[i]       = old + next;
      });
    }
    T* col = second->data_ptr<T>();
    {
      const Tensor means  = mean(matrix, {0});
      const T*     values = means.data_ptr<T>();
      detail::parallel_elements(columns, [&](std::int64_t i) {
        const T old  = col[i] * decay;
        const T next = values[i] * incoming;
        col[i]       = old + next;
      });
    }
    const T row_mean    = mean(first).item<T>();
    const T denominator = row_mean >= epsilon ? row_mean : epsilon;
    parallel_for(
        0,
        rows,
        std::max<std::int64_t>(1, detail::kParallelGrain / columns),
        [&](std::int64_t lo, std::int64_t hi) {
          for (std::int64_t r = lo; r < hi; ++r) {
            const T normalized = row[r] / denominator;
            for (std::int64_t column = 0; column < columns; ++column) {
              const auto i        = r * columns + column;
              const T    variance = normalized * col[column];
              const T    bounded  = variance >= epsilon ? variance : epsilon;
              const T    root     = std::sqrt(bounded);
              out[i]              = gradient[i] / root;
            }
          }
        });
    ++second->impl()->storage->version;
  } else {
    detail::parallel_elements(w.numel(), [&](std::int64_t i) {
      const T squared     = gradient[i] * gradient[i];
      const T regularized = squared + epsilon;
      const T old         = row[i] * decay;
      const T next        = regularized * incoming;
      const T variance    = old + next;
      row[i]              = variance;
      const T bounded     = variance >= epsilon ? variance : epsilon;
      const T root        = std::sqrt(bounded);
      out[i]              = gradient[i] / root;
    });
  }
  ++first.impl()->storage->version;
  const double clip = std::max(1.0, rms(update).item() / config.update_clip);
  if (config.relative_step)
    rate = std::min(rate, 1.0 / std::sqrt(time));
  if (config.scale_parameter)
    rate *= std::max(rms(w).item(), config.parameter_epsilon);
  scale_fused_update<T>(update, clip, rate);
  return update;
}

/// @brief Subtracts a matching packed decrement from a floating parameter and bumps its version.
void apply_packed_update(Tensor& parameter, const Tensor& update) {
  detail::dispatch_float(parameter.dtype(), "Optimizer::step", [&](auto tag) {
    using T       = typename decltype(tag)::type;
    T*       to   = parameter.data_ptr<T>();
    const T* from = update.data_ptr<T>();
    detail::parallel_elements(parameter.numel(), [&](std::int64_t i) { to[i] = to[i] - from[i]; });
  });
  ++parameter.impl()->storage->version;
}

}  // namespace

void Optimizer::step(const std::vector<Tensor>& gradients) {
  if (gradients.size() != params_.size())
    throw std::invalid_argument("Optimizer::step: got " + std::to_string(gradients.size()) + " gradients for " +
                                std::to_string(params_.size()) + " parameters");
  NoGradGuard  guard;
  const double t  = static_cast<double>(step_ + 1);
  const bool   correct_moments =
      config_.kind == OptimizerKind::Adam || config_.kind == OptimizerKind::AdamW ||
      config_.kind == OptimizerKind::Nadam || config_.kind == OptimizerKind::AMSGrad ||
      config_.kind == OptimizerKind::Adamax || config_.kind == OptimizerKind::Lamb;
  const double c1 = correct_moments ? 1 - std::pow(config_.beta1, t) : 1.0;
  const double c2 = correct_moments ? 1 - std::pow(config_.beta2, t) : 1.0;
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
    // No optimizer creates more than three slots for one parameter. Reserve
    // before acquiring any Tensor&/Tensor* so vector growth cannot dangle them.
    if (config_.kind != OptimizerKind::Sgd && slots_.capacity() - slots_.size() < 3)
      slots_.reserve(std::max(slots_.size() + 3, slots_.capacity() * 2));
    const Tensor w = params_[i].detach();
    const Tensor g = cast(gradients[i].detach(), w.dtype());
    if (g.shape() != w.shape())
      throw std::invalid_argument("Optimizer::step: gradient " + std::to_string(i) + " has shape " +
                                  shape_str(g.shape()) + ", parameter has " + shape_str(w.shape()));
    const bool basic = config_.kind == OptimizerKind::Sgd || config_.kind == OptimizerKind::Adam ||
                       config_.kind == OptimizerKind::AdamW;
    if (basic && w.is_contiguous() && g.is_contiguous()) {
      Tensor* first  = nullptr;
      Tensor* second = nullptr;
      if (config_.kind != OptimizerKind::Sgd) {
        first  = &slot(SlotKind::M, i, w.shape());
        second = &slot(SlotKind::V, i, w.shape());
      }
      // Restored broadcast-shaped or differentiable slots retain the
      // general path's out-of-place update and history behavior.
      if (!first || (first->shape() == w.shape() && second->shape() == w.shape() &&
                     !first->requires_grad() && first->is_contiguous() && !second->requires_grad() && second->is_contiguous())) {
        updates[i] = detail::dispatch_float(w.dtype(), "Optimizer::step", [&](auto tag) {
          using T = typename decltype(tag)::type;
          if (config_.kind == OptimizerKind::Sgd)
            return fused_basic_update<T, OptimizerKind::Sgd>(
                w, g, first, second, lr_, config_, c1, c2);
          if (config_.kind == OptimizerKind::AdamW)
            return fused_basic_update<T, OptimizerKind::AdamW>(
                w, g, first, second, lr_, config_, c1, c2);
          return fused_basic_update<T, OptimizerKind::Adam>(
              w, g, first, second, lr_, config_, c1, c2);
        });
        continue;
      }
    }
    const bool elementwise =
        config_.kind == OptimizerKind::Momentum || config_.kind == OptimizerKind::Nesterov ||
        config_.kind == OptimizerKind::Adagrad || config_.kind == OptimizerKind::RMSProp ||
        config_.kind == OptimizerKind::Adadelta || config_.kind == OptimizerKind::Lion ||
        config_.kind == OptimizerKind::Adamax || config_.kind == OptimizerKind::AMSGrad ||
        config_.kind == OptimizerKind::Nadam;
    if (elementwise && w.is_contiguous() && g.is_contiguous()) {
      std::array<Tensor*, 3> state{nullptr, nullptr, nullptr};
      switch (config_.kind) {
        case OptimizerKind::Momentum:
        case OptimizerKind::Nesterov:
          state[0] = &slot(SlotKind::Velocity, i, w.shape());
          break;
        case OptimizerKind::Adagrad:
          state[0] = &slot(SlotKind::Accumulator, i, w.shape(), config_.initial_accumulator);
          break;
        case OptimizerKind::RMSProp:
          state[0] = &slot(SlotKind::Variance, i, w.shape());
          break;
        case OptimizerKind::Adadelta:
          state[0] = &slot(SlotKind::Accum, i, w.shape());
          state[1] = &slot(SlotKind::AccumUpdate, i, w.shape());
          break;
        case OptimizerKind::Lion:
          state[0] = &slot(SlotKind::M, i, w.shape());
          break;
        case OptimizerKind::Adamax:
          state[0] = &slot(SlotKind::M, i, w.shape());
          state[1] = &slot(SlotKind::U, i, w.shape());
          break;
        default:
          state[0] = &slot(SlotKind::M, i, w.shape());
          state[1] = &slot(SlotKind::V, i, w.shape());
          if (config_.kind == OptimizerKind::AMSGrad)
            state[2] = &slot(SlotKind::VMax, i, w.shape());
          break;
      }
      if (std::all_of(state.begin(), state.end(), [&](const Tensor* value) {
            return !value || (value->shape() == w.shape() && value->dtype() == w.dtype() &&
                              !value->requires_grad() && value->is_contiguous());
          })) {
        updates[i] = detail::dispatch_float(w.dtype(), "Optimizer::step", [&](auto tag) {
          using T = typename decltype(tag)::type;
/// @def FUSED_CASE(kind)
/// @brief Dispatches one optimizer enum arm to the matching fused update template.
#define FUSED_CASE(kind)                               \
  case OptimizerKind::kind:                            \
    return fused_other_update<T, OptimizerKind::kind>( \
        w, g, state, lr_, config_, c1, c2, mu, mu_next, product, product_next)
          switch (config_.kind) {
            FUSED_CASE(Momentum);
            FUSED_CASE(Nesterov);
            FUSED_CASE(Adagrad);
            FUSED_CASE(RMSProp);
            FUSED_CASE(Adadelta);
            FUSED_CASE(Lion);
            FUSED_CASE(Adamax);
            FUSED_CASE(AMSGrad);
            FUSED_CASE(Nadam);
            default:
              throw std::logic_error("Optimizer::step: unexpected fused optimizer");
          }
#undef FUSED_CASE
        });
        continue;
      }
    }
    const bool norm_kind = config_.kind == OptimizerKind::Lars ||
                           config_.kind == OptimizerKind::Lamb ||
                           config_.kind == OptimizerKind::Adafactor;
    if (norm_kind && w.is_contiguous() && g.is_contiguous() && w.numel() > 0) {
      Tensor* first       = nullptr;
      Tensor* second      = nullptr;
      Shape   first_shape = w.shape(), second_shape = w.shape();
      if (config_.kind == OptimizerKind::Lars)
        first = &slot(SlotKind::Velocity, i, w.shape());
      else if (config_.kind == OptimizerKind::Lamb) {
        first  = &slot(SlotKind::M, i, w.shape());
        second = &slot(SlotKind::V, i, w.shape());
      } else if (w.dim() >= 2) {
        first_shape  = {w.numel() / w.shape().back()};
        second_shape = {w.shape().back()};
        first        = &slot(SlotKind::Row, i, first_shape);
        second       = &slot(SlotKind::Col, i, second_shape);
      } else
        first = &slot(SlotKind::V, i, w.shape());
      const bool valid = !first->requires_grad() && first->is_contiguous() && first->shape() == first_shape &&
                         (!second || (!second->requires_grad() && second->is_contiguous() && second->shape() == second_shape));
      if (valid) {
        if (config_.kind == OptimizerKind::Lars) {
          const double wn = l2_norm(w).item(), gn = l2_norm(g).item();
          const double local =
              config_.trust_coefficient * trust(wn, gn + config_.weight_decay * wn);
          updates[i] = detail::dispatch_float(w.dtype(), "Optimizer::step", [&](auto tag) {
            return fused_lars_update<typename decltype(tag)::type>(
                w, g, *first, lr_ * local, config_);
          });
        } else if (config_.kind == OptimizerKind::Lamb) {
          updates[i] = detail::dispatch_float(w.dtype(), "Optimizer::step", [&](auto tag) {
            return fused_lamb_update<typename decltype(tag)::type>(
                w, g, *first, *second, lr_, config_, c1, c2);
          });
        } else {
          updates[i] = detail::dispatch_float(w.dtype(), "Optimizer::step", [&](auto tag) {
            return fused_adafactor_update<typename decltype(tag)::type>(
                w, g, *first, second, lr_, config_, t);
          });
        }
        continue;
      }
    }
    Tensor direction = g;
    double rate      = lr_;
    switch (config_.kind) {
      case OptimizerKind::Sgd: break;
      case OptimizerKind::Momentum:
      case OptimizerKind::Nesterov: {
        Tensor& velocity = slot(SlotKind::Velocity, i, w.shape());
        velocity         = add(mul(velocity, config_.momentum), g);
        direction = config_.kind == OptimizerKind::Nesterov ? add(g, mul(velocity, config_.momentum)) : velocity;
        break;
      }
      case OptimizerKind::Adagrad: {
        Tensor& accumulator = slot(SlotKind::Accumulator, i, w.shape(), config_.initial_accumulator);
        accumulator         = add(accumulator, square(g));
        direction           = div(g, add(sqrt(accumulator), config_.epsilon));
        break;
      }
      case OptimizerKind::RMSProp: {
        Tensor& variance = slot(SlotKind::Variance, i, w.shape());
        variance         = add(mul(variance, config_.rho), mul(square(g), 1 - config_.rho));
        direction        = div(g, sqrt(add(variance, config_.epsilon)));
        break;
      }
      case OptimizerKind::Adadelta: {
        Tensor& accum        = slot(SlotKind::Accum, i, w.shape());
        Tensor& accum_update = slot(SlotKind::AccumUpdate, i, w.shape());
        accum                = add(mul(accum, config_.rho), mul(square(g), 1 - config_.rho));
        direction = mul(div(sqrt(add(accum_update, config_.epsilon)), sqrt(add(accum, config_.epsilon))), g);
        accum_update = add(mul(accum_update, config_.rho), mul(square(direction), 1 - config_.rho));
        break;
      }
      case OptimizerKind::Lars: {
        const double wn    = l2_norm(w).item();
        const double gn    = l2_norm(g).item();
        const double local = config_.trust_coefficient * trust(wn, gn + config_.weight_decay * wn);
        Tensor&      velocity = slot(SlotKind::Velocity, i, w.shape());
        // Original LARS keeps the learning rate inside the velocity.
        velocity  = add(mul(velocity, config_.momentum), mul(add(g, mul(w, config_.weight_decay)), lr_ * local));
        direction = velocity;
        rate      = 1.0;
        break;
      }
      case OptimizerKind::Lion: {
        Tensor&      m   = slot(SlotKind::M, i, w.shape());
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
          Tensor&            row    = slot(SlotKind::Row, i, {rows});
          Tensor&            col    = slot(SlotKind::Col, i, {columns});
          row = add(mul(row, decay), mul(mean(matrix, {1}), 1 - decay));
          col = add(mul(col, decay), mul(mean(matrix, {0}), 1 - decay));
          const Tensor normalized_row = div(row, maximum(mean(row), scalar_like(config_.factor_epsilon, row)));
          variance                    = reshape(outer(normalized_row, col), w.shape());
        } else {
          Tensor& v = slot(SlotKind::V, i, w.shape());
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
        Tensor& m      = slot(SlotKind::M, i, w.shape());
        m              = add(mul(m, config_.beta1), mul(g, 1 - config_.beta1));
        const Tensor m_hat = div(m, c1);
        if (config_.kind == OptimizerKind::Adamax) {
          Tensor& u = slot(SlotKind::U, i, w.shape());
          u         = maximum(mul(u, config_.beta2), abs(g));
          direction = div(m_hat, add(u, config_.epsilon));
          break;
        }
        Tensor& v = slot(SlotKind::V, i, w.shape());
        v         = add(mul(v, config_.beta2), mul(square(g), 1 - config_.beta2));
        Tensor denom_v = v;
        if (config_.kind == OptimizerKind::AMSGrad) {
          Tensor& vmax = slot(SlotKind::VMax, i, w.shape());
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
  for (std::size_t i = 0; i < params_.size(); ++i) {
    if (!updates[i].defined())
      continue;
    if (params_[i].is_contiguous() && updates[i].is_contiguous() &&
        updates[i].shape() == params_[i].shape() && updates[i].dtype() == params_[i].dtype())
      apply_packed_update(params_[i], updates[i]);
    else
      params_[i].sub_(updates[i]);
  }
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
      slots_.emplace_back(name, cast(value, params_[index].dtype()).clone());
      // Keep arbitrary checkpoint entries and creation order. Only exact
      // canonical names participate in direct lookup, just as before; e.g.
      // "m/01" must not become an alias of "m/1". Last duplicates win.
      if (name.substr(slash + 1) == std::to_string(index)) {
        for (std::size_t kind = 0; kind < static_cast<std::size_t>(SlotKind::Count); ++kind) {
          if (name.compare(0, slash, slot_name(static_cast<SlotKind>(kind))) != 0) continue;
          if (slot_index_.empty()) slot_index_.resize(params_.size());
          slot_index_[index][kind] = slots_.size();
          break;
        }
      }
    }
  }
}

// ---- gradient utilities -------------------------------------------------------------------

double global_norm(const std::vector<Tensor>& tensors) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
  NoGradGuard guard;
  double      total = 0;
  for (const auto& tensor : tensors) {
    if (!tensor.defined())
      continue;
    const Tensor packed   = tensor.contiguous();
    const auto   count    = packed.numel();
    const double subtotal = detail::dispatch(packed.dtype(), [&](auto tag) {
      using T         = typename decltype(tag)::type;
      const T* values = packed.data_ptr<T>();
      double   sum    = 0;
      for (std::int64_t i = 0; i < count; ++i) {
        const double value   = static_cast<double>(values[i]);
        const double squared = value * value;
        sum += squared;
      }
      return sum;
    });
    total += subtotal;
  }
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

Tensor max_norm_constraint(const Tensor& w, double max_norm, const std::vector<std::int64_t>& axes) {
  if (!(max_norm > 0) || !std::isfinite(max_norm))
    throw std::invalid_argument("max_norm_constraint: max_norm must be positive and finite");
  if (!is_floating(w.dtype())) throw std::invalid_argument("max_norm_constraint: w must be floating point");
  const Tensor norms = sqrt(sum(square(w), axes, true));
  return div(w, maximum(scalar_like(1.0, norms), div(norms, max_norm)));
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
      if (sums_[i].defined())
        sums_[i] = nn::add(
            sums_[i], weight == 1.0 ? gradients[i].detach() : mul(gradients[i].detach(), weight));
      else
        sums_[i] = scaled(gradients[i]);
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
