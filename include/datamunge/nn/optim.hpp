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

/// @brief First-order parameter update rule selected by OptimizerConfig.
enum class OptimizerKind {
  Sgd,        ///< Subtract learning_rate times the gradient, without momentum state.
  Momentum,   ///< Accumulate velocity as momentum times old velocity plus gradient.
  Nesterov,   ///< Use gradient plus momentum times the newly updated velocity.
  Adagrad,    ///< Divide by the square root of accumulated squared gradients plus epsilon.
  RMSProp,    ///< Divide by the square root of exponentially averaged squared gradients plus epsilon.
  Adam,       ///< Bias-correct first/second moments and divide by sqrt(second moment) plus epsilon.
  AdamW,      ///< Adam direction with decoupled weight decay.
  Nadam,      ///< Adam with scheduled Nesterov momentum, following the TensorFlow-style schedule.
  AMSGrad,    ///< Adam with a nondecreasing maximum second-moment estimate.
  Adamax,     ///< Adam variant with an exponentially decayed infinity-norm denominator.
  Lamb,       ///< Scale an AdamW direction by the weight-to-direction L2 norm ratio.
  Lars,       ///< Layer-wise norm-based learning-rate scaling with momentum and weight decay.
  Lion,       ///< Sign of interpolated momentum with a separate momentum-history decay.
  Adafactor,  ///< Factored second moments for matrices, with optional relative rate/parameter scaling.
  Adadelta,   ///< Scale gradients by RMS update history divided by RMS gradient history.
};

/// @brief Returns the display name of an optimizer rule.
/// @throws std::invalid_argument If kind is not a recognized enum value.
std::string optimizer_name(OptimizerKind kind);

/// @brief Validated update-rule settings; unused fields retain their defaults for other rules.
struct OptimizerConfig {
  OptimizerKind kind{OptimizerKind::Sgd}; ///< Rule used for every parameter in this optimizer.
  double        momentum{0.9}; ///< Velocity decay for Momentum, Nesterov, and LARS; in [0, 1).
  double        beta1{0.9}; ///< First-moment decay for Adam-family methods and Lion; in [0, 1).
  double        beta2{0.999}; ///< Second-moment decay, or Lion's history decay; in [0, 1).
  double        rho{0.9}; ///< RMSProp/Adadelta squared-history decay; in [0, 1).
  double        epsilon{1e-8}; ///< Positive numerical stabilizer; placement depends on the update rule.
  double        initial_accumulator{0.1}; ///< Nonnegative starting squared-gradient sum for Adagrad.
  double        weight_decay{0.0}; ///< Nonnegative decay used only by AdamW, LAMB, LARS, and Lion.
  double        trust_coefficient{0.001}; ///< Positive multiplier for the LARS local trust ratio.
  double        factor_epsilon{1e-30}; ///< Positive Adafactor squared-gradient regularizer/variance floor.
  double        parameter_epsilon{1e-3}; ///< Positive RMS weight floor for Adafactor parameter scaling.
  double        update_clip{1.0}; ///< Positive maximum RMS of the normalized Adafactor direction.
  double        decay_exponent{0.8}; ///< Adafactor second-moment decay exponent, in (0, 1].
  bool          relative_step{true}; ///< Cap Adafactor's rate at 1/sqrt(step) when enabled.
  bool          scale_parameter{true}; ///< Multiply Adafactor's rate by max(RMS(weight), parameter_epsilon).

  /// Throws std::invalid_argument for out-of-range settings.
  void validate() const;
};

/// @brief In-place parameter updater retaining per-parameter state across calls.
/// Parameter handles alias the caller's leaves. Construction captures the current
/// parameter list; later additions to a ParameterStore are not included automatically.
class Optimizer {
 public:
  /// @brief Retains nonempty learnable leaf handles and validates the rate/configuration.
  /// @throws std::invalid_argument For invalid settings or undefined/non-leaf/untracked parameters.
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

  /// @brief Returns the current nonnegative base learning rate.
  [[nodiscard]] double                     learning_rate() const noexcept { return lr_; }
  /// @brief Replaces the base learning rate; zero is allowed.
  /// @throws std::invalid_argument If learning_rate is negative or non-finite.
  void                                     set_learning_rate(double learning_rate);
  /// @brief Returns completed step() calls, including calls where all gradients were undefined.
  [[nodiscard]] std::int64_t               iterations() const noexcept { return step_; }
  /// @brief Borrows retained parameter handles in their gradient-alignment order.
  [[nodiscard]] const std::vector<Tensor>& parameters() const noexcept { return params_; }
  /// @brief Borrows the validated update settings supplied at construction.
  [[nodiscard]] const OptimizerConfig&     config() const noexcept { return config_; }

  /// Independent copies of slot tensors named "<slot>/<index>", plus step and Nadam schedule state.
  /// Parameter values, optimizer kind, and learning rate are not included.
  [[nodiscard]] std::vector<std::pair<std::string, Tensor>> state() const;
  /// Restores what state() returned.
  void load_state(const std::vector<std::pair<std::string, Tensor>>& state);

 private:
  /// @brief Stable categories used to address lazily allocated optimizer history tensors.
  enum class SlotKind {
    Velocity, ///< Momentum/Nesterov/LARS velocity history.
    Accumulator, ///< Adagrad cumulative squared gradients.
    Variance, ///< RMSProp exponentially averaged squared gradients.
    Accum, ///< Adadelta squared-gradient history.
    AccumUpdate, ///< Adadelta squared-update history.
    M, ///< Adam-family first moment, or Lion momentum history.
    U, ///< Adamax decayed infinity-norm estimate.
    V, ///< Adam-family second moment or unfactored Adafactor variance.
    VMax, ///< AMSGrad maximum second moment.
    Row, ///< Adafactor per-row second moment after flattening leading axes.
    Col, ///< Adafactor per-column second moment along the last axis.
    Count ///< Number of real slot categories; not a valid slot to allocate.
  };
  /// @brief Returns the serialized name for a real SlotKind; Count must not be supplied.
  static const char* slot_name(SlotKind kind);
  /// @brief Borrows an existing slot or creates it with the parameter dtype and requested shape/fill.
  Tensor& slot(SlotKind kind, std::size_t index, const Shape& shape, double initial = 0.0);

  /// @brief Learnable leaf handles shared with the caller, in gradient input order.
  std::vector<Tensor>                                  params_;
  /// @brief Nonnegative finite base learning rate, possibly further scaled by the chosen rule.
  double                                               lr_;
  /// @brief Immutable update-rule settings validated at construction.
  OptimizerConfig                                      config_;
  /// @brief Count of completed optimization steps, used for moment bias correction.
  std::int64_t                                         step_{0};
  /// @brief Cumulative scheduled momentum product used by Nadam bias correction.
  double                                               nadam_product_{1.0};
  /// @brief Named history tensors; step() reserves capacity before borrowing up to three slots.
  std::vector<std::pair<std::string, Tensor>>          slots_;
  /// @brief Lazily allocated [parameter][slot kind] indices into slots_, with zero for absent and index+1 otherwise.
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
  /// Weighted sum of the added gradients, returned as handles aliasing internal sums.
  [[nodiscard]] std::vector<Tensor> sum() const;
  /// Weighted sum divided by the total weight (the number of add() calls
  /// when every weight is 1). Throws std::logic_error when empty.
  [[nodiscard]] std::vector<Tensor> mean() const;
  /// @brief Returns the number of successful add() calls since construction or reset().
  [[nodiscard]] int                 count() const noexcept { return count_; }
  /// @brief Returns the sum of weights supplied to add(), including all-undefined batches.
  [[nodiscard]] double              total_weight() const noexcept { return weight_; }
  /// @brief Drops accumulated tensor handles and clears the call count and total weight.
  void                              reset();

 private:
  /// @brief Per-position weighted gradient sums; undefined handles represent no contribution yet.
  std::vector<Tensor> sums_;
  /// @brief Number of accumulated micro-batches.
  int                 count_{0};
  /// @brief Sum of positive weights, used as mean()'s denominator.
  double              weight_{0.0};
};

}  // namespace datamunge::nn
