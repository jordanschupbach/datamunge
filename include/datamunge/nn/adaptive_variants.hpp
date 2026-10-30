#pragma once

/// @file adaptive_variants.hpp
/// @brief Three further Adam-family optimizers (RAdam, AdaBelief, Yogi), RAdam's variance rectification and the exact
///        worst-case size of an Adam step.
///
/// optim.hpp provides Adagrad, RMSProp, Adam, AdamW, Nadam, AMSGrad and Adamax. The rules here change one ingredient
/// each. RAdam (Liu et al. 2020) multiplies the adaptive step by a rectification term that switches the adaptive
/// denominator off while its variance is still undefined, replacing a learning-rate warm-up. AdaBelief (Zhuang et al.
/// 2020) divides by the spread of the gradient around its running mean instead of its raw second moment, so steady
/// gradients give large steps. Yogi (Zaheer et al. 2018) changes the second-moment update to an additive one whose
/// size is limited by the current squared gradient, so the denominator cannot shrink as fast as in Adam.

#include "datamunge/nn/params.hpp"
#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::nn {

/// @brief Update rules of AdaptiveOptimizer.
enum class AdaptiveRule {
  RAdam,      ///< Adam with the variance rectification of Liu et al. (2020); plain momentum SGD while \f$\rho_t\le5\f$.
  AdaBelief,  ///< Adam with \f$s_t=\beta_2s_{t-1}+(1-\beta_2)(g_t-m_t)^2+\epsilon\f$ in place of the second moment (Zhuang et al. 2020).
  Yogi,       ///< Adam with \f$v_t=v_{t-1}-(1-\beta_2)\,\mathrm{sign}(v_{t-1}-g_t^2)\,g_t^2\f$ (Zaheer et al. 2018).
};

/// @brief Returns the display name of an adaptive rule ("RAdam", "AdaBelief" or "Yogi").
/// @param rule  The rule.
/// @return Its name.
[[nodiscard]] std::string adaptive_rule_name(AdaptiveRule rule);

/// @brief Settings of AdaptiveOptimizer.
struct AdaptiveConfig {
  AdaptiveRule rule{AdaptiveRule::RAdam};  ///< Update rule.
  double beta1{0.9};                       ///< First-moment decay, in [0, 1).
  double beta2{0.999};                     ///< Second-moment decay, in [0, 1).
  double epsilon{1e-8};                    ///< Positive stabiliser added to the square root of the second moment.
  double weight_decay{0.0};                ///< Nonnegative decoupled weight decay: \f$\theta\leftarrow\theta-\eta\lambda\theta\f$ before the step.

  /// @brief Throws std::invalid_argument for out-of-range settings.
  void validate() const;
};

/// @brief An in-place optimizer for the rules of AdaptiveRule, with the same calling conventions as Optimizer.
///
/// Every rule keeps a first moment \f$m\f$ and a second statistic \f$v\f$ per parameter, both bias-corrected by \f$1-\beta^t\f$:
/// - RAdam: \f$\theta\leftarrow\theta-\eta\,r_t\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)\f$ when \f$\rho_t>5\f$, otherwise \f$\theta\leftarrow\theta-\eta\hat m_t\f$; see radam_rectification().
/// - AdaBelief: \f$\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat s_t}+\epsilon)\f$ with \f$s_t\f$ the running mean of \f$(g_t-m_t)^2\f$ plus \f$\epsilon\f$.
/// - Yogi: \f$\theta\leftarrow\theta-\eta\,\hat m_t/(\sqrt{\hat v_t}+\epsilon)\f$ with Yogi's additive \f$v_t\f$ (zero initial state, so the first step equals Adam's).
class AdaptiveOptimizer {
 public:
  /// @brief Retains the parameter handles and validates the settings.
  /// @param parameters  Learnable leaf tensors.
  /// @param learning_rate  Nonnegative base learning rate \f$\eta\f$.
  /// @param config  Rule and hyperparameters.
  /// @throws std::invalid_argument  For invalid settings, a negative rate or undefined parameters.
  AdaptiveOptimizer(std::vector<Tensor> parameters, double learning_rate, AdaptiveConfig config = {});
  /// @brief Optimizes every parameter currently in `store`.
  /// @param store  Parameter store.
  /// @param learning_rate  Nonnegative base learning rate.
  /// @param config  Rule and hyperparameters.
  AdaptiveOptimizer(const ParameterStore& store, double learning_rate, AdaptiveConfig config = {});

  /// @brief Applies one update with gradients aligned to parameters(); an undefined gradient skips that parameter.
  /// @param gradients  One gradient per parameter.
  /// @throws std::invalid_argument  If the number of gradients or a shape does not match.
  void step(const std::vector<Tensor>& gradients);
  /// @brief Computes the gradients of `loss` with respect to the parameters and applies one update.
  /// @param loss  Scalar loss.
  void minimize(const Tensor& loss);

  /// @brief Returns the base learning rate.
  [[nodiscard]] double learning_rate() const noexcept { return lr_; }
  /// @brief Replaces the base learning rate.
  /// @param learning_rate  Nonnegative finite rate.
  /// @throws std::invalid_argument  If it is negative or not finite.
  void set_learning_rate(double learning_rate);
  /// @brief Returns the number of completed step() calls.
  [[nodiscard]] std::int64_t iterations() const noexcept { return step_; }
  /// @brief Returns the settings.
  [[nodiscard]] const AdaptiveConfig& config() const noexcept { return config_; }
  /// @brief Copies of the state tensors, named "m/<i>" and "v/<i>" (for AdaBelief "v" holds s), plus "step".
  [[nodiscard]] std::vector<std::pair<std::string, Tensor>> state() const;

 private:
  std::vector<Tensor> params_;  ///< Parameter handles shared with the caller.
  std::vector<Tensor> m_;       ///< First moments, created on first use.
  std::vector<Tensor> v_;       ///< Second statistics, created on first use.
  double lr_;                   ///< Base learning rate.
  AdaptiveConfig config_;       ///< Validated settings.
  std::int64_t step_{0};        ///< Completed steps.
};

/// @brief RAdam's length of the approximated simple moving average, \f$\rho_t=\rho_\infty-2t\beta_2^t/(1-\beta_2^t)\f$ with \f$\rho_\infty=2/(1-\beta_2)-1\f$.
/// @param t  Step number, at least 1.
/// @param beta2  Second-moment decay in (0, 1).
/// @return \f$\rho_t\f$, which rises from 1 at t = 1 towards \f$\rho_\infty\f$.
/// @throws std::invalid_argument  If t < 1 or beta2 is outside (0, 1).
[[nodiscard]] double radam_rho(std::int64_t t, double beta2);

/// @brief RAdam's rectification term \f$r_t=\sqrt{\frac{(\rho_t-4)(\rho_t-2)\rho_\infty}{(\rho_\infty-4)(\rho_\infty-2)\rho_t}}\f$, or 0 while \f$\rho_t\le5\f$ (the adaptive step is then off).
///
/// Liu et al. derive it as \f$\sqrt{\mathrm{Var}[\psi_\infty]/\mathrm{Var}[\psi_t]}\f$ for the adaptive learning rate \f$\psi_t=\sqrt{(1-\beta_2^t)/v_t}\f$ under Gaussian gradients, using a
/// scaled-inverse-\f$\chi^2\f$ approximation with \f$\rho_t\f$ degrees of freedom. The threshold 5 follows the authors' code (the paper uses 4).
/// @param t  Step number, at least 1.
/// @param beta2  Second-moment decay in (0, 1).
/// @return \f$r_t\in[0,1)\f$.
[[nodiscard]] double radam_rectification(std::int64_t t, double beta2);

/// @brief The largest possible size of Adam's bias-corrected step at step t, in units of the learning rate (ignoring \f$\epsilon\f$).
///
/// With \f$m_t=(1-\beta_1)\sum_i\beta_1^{t-i}g_i\f$ and \f$v_t=(1-\beta_2)\sum_i\beta_2^{t-i}g_i^2\f$, the Cauchy–Schwarz inequality gives
/// \f$|\hat m_t|/\sqrt{\hat v_t}\le\frac{(1-\beta_1)\sqrt{1-\beta_2^t}}{(1-\beta_1^t)\sqrt{1-\beta_2}}\sqrt{\sum_{k=0}^{t-1}(\beta_1^2/\beta_2)^k}\f$, with equality for \f$g_i\propto(\beta_1/\beta_2)^{t-i}\f$. For the defaults
/// (0.9, 0.999) this tends to 7.27, so a single coordinate can move by more than seven learning rates.
/// @param beta1  First-moment decay in [0, 1).
/// @param beta2  Second-moment decay in (0, 1).
/// @param t  Step number, at least 1.
/// @return The supremum of \f$|\Delta\theta_t|/\eta\f$ over all gradient histories.
/// @throws std::invalid_argument  For parameters out of range.
[[nodiscard]] double adam_step_bound(double beta1, double beta2, std::int64_t t);

}  // namespace datamunge::nn
