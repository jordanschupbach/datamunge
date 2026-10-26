#pragma once

/// @file optim.hpp
/// @brief Tensor optimizers, gradient clipping and accumulation.
///
/// One `Optimizer` class covers the usual first-order methods; the update
/// rules match tflow's `nn::Optimizer` so ported examples follow the same
/// trajectories. Updates are in place and not recorded by autograd.
///
/// Typical loop:
///   nn::Optimizer opt(store, 1e-3, {nn::OptimizerKind::Adam});
///   for (...) { auto g = nn::grad(loss(), store.parameters()); opt.step(g); }

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "datamunge/nn/params.hpp"
#include "datamunge/nn/tensor.hpp"

namespace datamunge::nn {

enum class OptimizerKind {
  Sgd,        // w -= lr g
  Momentum,   // v = m v + g;           w -= lr v
  Nesterov,   // v = m v + g;           w -= lr (g + m v)
  Adagrad,    // a += g^2;              w -= lr g / (sqrt(a) + eps)
  RMSProp,    // v = rho v + (1-rho) g^2; w -= lr g / sqrt(v + eps)
  Adam,       // bias-corrected moments; w -= lr m_hat / (sqrt(v_hat) + eps)
  AdamW,      // Adam + decoupled weight decay (lr * wd * w)
  Nadam,      // Adam with Nesterov momentum (Dozat 2016, TF schedule)
  AMSGrad,    // Adam with the running max of v
  Adamax,     // infinity-norm Adam
  Lamb,       // AdamW direction scaled by the trust ratio ||w|| / ||d||
  Lars,       // layer-wise trust ratio with momentum
  Lion,       // sign of an interpolated momentum (Chen et al. 2023)
  Adafactor,  // factored second moments (Shazeer & Stern 2018)
  Adadelta,   // a = rho a + (1-rho) g^2; d = sqrt(u + eps) / sqrt(a + eps) g;
              // u = rho u + (1-rho) d^2; w -= lr d   (TF ApplyAdadelta; TF uses
              // rho 0.95, eps 1e-7 (Keras) / 1e-8 (tf.train), lr 1.0 (tf.train) - set them in the config)
};

std::string optimizer_name(OptimizerKind kind);

struct OptimizerConfig {
  OptimizerKind kind{OptimizerKind::Sgd};
  double        momentum{0.9};
  double        beta1{0.9};
  double        beta2{0.999};
  double        rho{0.9};
  double        epsilon{1e-8};
  double        initial_accumulator{0.1};
  double        weight_decay{0.0};
  double        trust_coefficient{0.001};                    // LARS
  double        factor_epsilon{1e-30};
  double        parameter_epsilon{1e-3};  // Adafactor
  double        update_clip{1.0};
  double        decay_exponent{0.8};           // Adafactor
  bool          relative_step{true};
  bool          scale_parameter{true};      // Adafactor
  /// Throws std::invalid_argument for out-of-range settings.
  void validate() const;
};

class Optimizer {
 public:
  Optimizer(std::vector<Tensor> parameters, double learning_rate, OptimizerConfig config = {});
  /// Optimizes every parameter currently in `store`.
  Optimizer(const ParameterStore& store, double learning_rate, OptimizerConfig config = {});

  /// Applies one update with gradients aligned to parameters(). An
  /// undefined gradient skips that parameter.
  void step(const std::vector<Tensor>& gradients);
  /// Applies one update from each parameter's accumulated grad().
  void step();
  /// Computes the gradients of `loss` and applies one update.
  void minimize(const Tensor& loss);
  /// Drops every parameter's accumulated grad().
  void zero_grad();

  [[nodiscard]] double                     learning_rate() const noexcept { return lr_; }
  void                                     set_learning_rate(double learning_rate);
  [[nodiscard]] std::int64_t               iterations() const noexcept { return step_; }
  [[nodiscard]] const std::vector<Tensor>& parameters() const noexcept { return params_; }
  [[nodiscard]] const OptimizerConfig&     config() const noexcept { return config_; }

  /// Slot tensors and the step count, named "<slot>/<index>" (for checkpoints).
  [[nodiscard]] std::vector<std::pair<std::string, Tensor>> state() const;
  /// Restores what state() returned.
  void load_state(const std::vector<std::pair<std::string, Tensor>>& state);

 private:
  enum class SlotKind { Velocity, Accumulator, Variance, Accum, AccumUpdate, M, U, V, VMax, Row, Col, Count };
  static const char* slot_name(SlotKind kind);
  Tensor& slot(SlotKind kind, std::size_t index, const Shape& shape, double initial = 0.0);

  std::vector<Tensor>                                  params_;
  double                                               lr_;
  OptimizerConfig                                      config_;
  std::int64_t                                         step_{0};
  double                                               nadam_product_{1.0};
  // step() reserves room for all three possible slots before taking references.
  std::vector<std::pair<std::string, Tensor>>          slots_;
  // Direct [parameter][slot kind] offsets; zero means absent, otherwise index+1.
  // Allocated lazily so SGD pays no slot-index cost.
  std::vector<std::array<std::size_t, 11>>             slot_index_;
};

// ---- gradient utilities -----------------------------------------------------------------

/// sqrt(sum of squared entries over all tensors); undefined tensors are skipped.
double global_norm(const std::vector<Tensor>& tensors);
/// Clamps every entry to [low, high].
std::vector<Tensor> clip_by_value(const std::vector<Tensor>& gradients, double low, double high);
/// Rescales one tensor so its norm is at most max_norm.
Tensor clip_by_norm(const Tensor& gradient, double max_norm);
/// Max-norm weight constraint (Keras MaxNorm, Kim 2014's l2 constraint s):
/// each slice's L2 norm over `axes` (default {0}: every column of a [in, out]
/// kernel) is rescaled to at most max_norm: w / max(1, norm / max_norm).
/// Apply it to the parameter after an optimizer step, e.g.
/// `w.copy_(max_norm_constraint(w.detach(), 3.0))` under a NoGradGuard.
Tensor max_norm_constraint(const Tensor& w, double max_norm, const std::vector<std::int64_t>& axes = {0});
/// Rescales all tensors by max_norm / max(global_norm, max_norm) (TF
/// clip_by_global_norm). Returns the clipped tensors and the norm before clipping.
std::pair<std::vector<Tensor>, double> clip_by_global_norm(const std::vector<Tensor>& gradients, double max_norm);

/// Sums gradients over several micro-batches.
class GradientAccumulator {
 public:
  /// Adds weight * gradients. Pass a micro-batch's sample count as `weight`
  /// to make mean() the full-batch gradient of a mean loss over unequal
  /// micro-batches. A nonpositive or non-finite weight throws and leaves the
  /// accumulator unchanged.
  void add(const std::vector<Tensor>& gradients, double weight = 1.0);
  /// Weighted sum of the added gradients.
  [[nodiscard]] std::vector<Tensor> sum() const;
  /// Weighted sum divided by the total weight (the number of add() calls
  /// when every weight is 1). Throws std::logic_error when empty.
  [[nodiscard]] std::vector<Tensor> mean() const;
  [[nodiscard]] int                 count() const noexcept { return count_; }
  [[nodiscard]] double              total_weight() const noexcept { return weight_; }
  void                              reset();

 private:
  std::vector<Tensor> sums_;
  int                 count_{0};
  double              weight_{0.0};
};

}  // namespace datamunge::nn
