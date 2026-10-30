#pragma once

/// @file advanced_variants.hpp
/// @brief Newton–Schulz orthogonalisation and the Muon optimizer, the Lookahead wrapper and sharpness-aware (SAM)
///        gradients.
///
/// optim.hpp provides LAMB, LARS, Lion and Adafactor. The additions here act on whole matrices or wrap another optimizer:
/// - newton_schulz_orthogonalize() pushes every singular value of a matrix towards 1 with a few matrix products, so it
///   approximates the orthogonal polar factor \f$UV^\top\f$ of \f$G=U\Sigma V^\top\f$ without an SVD.
/// - Muon (Jordan et al. 2024) applies it to the momentum of each weight matrix, so every update has all its singular
///   values near 1: the step size is the same in every direction of the matrix.
/// - Lookahead (Zhang et al. 2019) keeps slow weights and, every k steps of any inner optimizer, moves them a fraction
///   \f$\alpha\f$ towards the fast weights and resets the fast weights there.
/// - sam_gradients() (Foret et al. 2021) returns the gradient at the worst nearby point \f$w+\rho\,g/\lVert g\rVert\f$, which any optimizer
///   can then apply; it prefers minima whose neighbourhood is flat.

#include "datamunge/nn/tensor.hpp"

#include <cstdint>
#include <functional>
#include <vector>

namespace datamunge::nn {

/// @brief Coefficients of the odd polynomial \f$p(x)=ax+bx^3+cx^5\f$ applied to the singular values in each Newton–Schulz step.
struct NewtonSchulzCoefficients {
  double a{3.4445};   ///< Linear coefficient.
  double b{-4.7750};  ///< Cubic coefficient.
  double c{2.0315};   ///< Quintic coefficient.

  /// @brief The classical cubic iteration \f$p(x)=\tfrac32x-\tfrac12x^3\f$, which converges to exactly 1 for singular values in (0, \f$\sqrt3\f$).
  [[nodiscard]] static NewtonSchulzCoefficients cubic() { return {1.5, -0.5, 0.0}; }
  /// @brief Jordan's quintic (3.4445, −4.7750, 2.0315), the default: faster growth of small singular values, but the
  ///        values then oscillate around 1 (roughly within [0.7, 1.2]) instead of converging.
  [[nodiscard]] static NewtonSchulzCoefficients quintic() { return {}; }
};

/// @brief Approximates the orthogonal polar factor \f$UV^\top\f$ of a matrix by Newton–Schulz iteration.
///
/// The matrix is first divided by its Frobenius norm (plus 1e-7), so every singular value lies in (0, 1]. Each step
/// computes \f$X\leftarrow aX+b\,(XX^\top)X+c\,(XX^\top)^2X\f$, which maps each singular value \f$\sigma\f$ to \f$p(\sigma)\f$ and leaves the singular vectors unchanged.
/// After k steps the result is exactly \f$U\,\mathrm{diag}(p^{(k)}(\sigma_i/\lVert G\rVert_F))\,V^\top\f$. Wide and tall matrices are both accepted.
///
/// @param G  Matrix [m, n].
/// @param steps  Number of iterations, at least 1.
/// @param coefficients  Polynomial coefficients.
/// @return A matrix [m, n] with the singular vectors of G and singular values near 1.
/// @throws std::invalid_argument  If G is not rank 2 or steps < 1.
[[nodiscard]] Tensor newton_schulz_orthogonalize(const Tensor& G, int steps = 5,
                                                 NewtonSchulzCoefficients coefficients = NewtonSchulzCoefficients::quintic());

/// @brief Settings of Muon.
struct MuonConfig {
  double momentum{0.95};                ///< Momentum \f$\mu\f$ in [0, 1).
  bool nesterov{true};                  ///< Orthogonalise \f$g+\mu m\f$ (Nesterov) instead of \f$m\f$.
  int ns_steps{5};                      ///< Newton–Schulz iterations per update.
  NewtonSchulzCoefficients coefficients{};  ///< Polynomial of the iteration.
  double weight_decay{0.0};             ///< Decoupled weight decay \f$\lambda\ge0\f$.
};

/// @brief The Muon optimizer: momentum SGD whose matrix updates are orthogonalised by Newton–Schulz iteration.
///
/// For a weight matrix [m, n]: \f$m_t=\mu m_{t-1}+g_t\f$, \f$u=g_t+\mu m_t\f$ (Nesterov) or \f$u=m_t\f$, and
/// \f$W\leftarrow W-\eta\lambda W-\eta\sqrt{\max(1,m/n)}\;\mathrm{NS}(u)\f$. Parameters that are not matrices (biases, gains) get the same momentum
/// update without orthogonalisation, as plain momentum SGD; in practice they are often given to a separate optimizer.
class Muon {
 public:
  /// @brief Retains the parameter handles and validates the settings.
  /// @param parameters  Learnable leaf tensors.
  /// @param learning_rate  Nonnegative learning rate \f$\eta\f$.
  /// @param config  Settings.
  /// @throws std::invalid_argument  For invalid settings or parameters.
  Muon(std::vector<Tensor> parameters, double learning_rate, MuonConfig config = {});
  /// @brief Applies one update with gradients aligned to the parameters; undefined gradients are skipped.
  /// @param gradients  One gradient per parameter.
  void step(const std::vector<Tensor>& gradients);
  /// @brief Computes the gradients of `loss` and applies one update.
  /// @param loss  Scalar loss.
  void minimize(const Tensor& loss);
  /// @brief Returns the learning rate.
  [[nodiscard]] double learning_rate() const noexcept { return lr_; }
  /// @brief Replaces the learning rate.
  /// @param learning_rate  Nonnegative finite rate.
  void set_learning_rate(double learning_rate);
  /// @brief Returns the number of completed steps.
  [[nodiscard]] std::int64_t iterations() const noexcept { return step_; }

 private:
  std::vector<Tensor> params_;    ///< Parameter handles.
  std::vector<Tensor> momentum_;  ///< Momentum buffers, created on first use.
  double lr_;                     ///< Learning rate.
  MuonConfig config_;             ///< Settings.
  std::int64_t step_{0};          ///< Completed steps.
};

/// @brief Lookahead (Zhang et al. 2019): slow weights that follow the fast weights of any inner optimizer.
///
/// Call after_step() after every step of the inner optimizer. Every k-th call sets \f$\phi\leftarrow\phi+\alpha(\theta-\phi)\f$ for the slow weights \f$\phi\f$ and copies \f$\phi\f$
/// back into the parameters \f$\theta\f$. For a linear inner update \f$\theta\mapsto c\,\theta\f$ (gradient descent on a quadratic), one outer cycle
/// multiplies the error by exactly \f$1-\alpha+\alpha c^k\f$.
class Lookahead {
 public:
  /// @brief Records the current parameter values as the slow weights.
  /// @param parameters  The tensors the inner optimizer updates.
  /// @param k  Inner steps per synchronisation, at least 1.
  /// @param alpha  Slow-weight step size in (0, 1].
  /// @throws std::invalid_argument  For invalid k, alpha or parameters.
  Lookahead(std::vector<Tensor> parameters, int k = 5, double alpha = 0.5);
  /// @brief Counts one inner step and synchronises every k-th time.
  /// @return Whether this call synchronised.
  bool after_step();
  /// @brief Returns copies of the slow weights.
  [[nodiscard]] std::vector<Tensor> slow_weights() const;

 private:
  std::vector<Tensor> params_;  ///< Parameter handles.
  std::vector<Tensor> slow_;    ///< Slow weights.
  int k_;                       ///< Inner steps per synchronisation.
  double alpha_;                ///< Slow-weight step size.
  int count_{0};                ///< Inner steps since the last synchronisation.
};

/// @brief Loss and gradients returned by sam_gradients().
struct SamResult {
  Tensor loss;                     ///< Loss at the current (unperturbed) weights.
  Tensor sharp_loss;               ///< Loss at the perturbed weights \f$w+\epsilon\f$.
  std::vector<Tensor> gradients;   ///< Gradients at the perturbed weights, aligned to the parameters.
};

/// @brief Sharpness-aware gradients (Foret et al. 2021): \f$\nabla L(w+\epsilon)\f$ with \f$\epsilon=\rho\,g/\lVert g\rVert\f$, \f$g=\nabla L(w)\f$ over all parameters jointly.
///
/// The loss closure is evaluated twice; the parameters are perturbed in place and restored before returning. The
/// result can be passed to any optimizer's step(). With \f$\rho=0\f$ it is the ordinary gradient.
/// @param parameters  Learnable leaf tensors.
/// @param loss  Closure computing the scalar loss from the current parameter values.
/// @param rho  Neighbourhood radius \f$\rho\ge0\f$.
/// @return The two losses and the perturbed gradients.
/// @throws std::invalid_argument  If rho is negative or there are no parameters.
[[nodiscard]] SamResult sam_gradients(const std::vector<Tensor>& parameters, const std::function<Tensor()>& loss, double rho = 0.05);

}  // namespace datamunge::nn
