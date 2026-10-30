/// @file advanced_variants.cpp
/// @brief Newton–Schulz orthogonalisation, Muon, Lookahead and sharpness-aware gradients.

#include "datamunge/nn/advanced_variants.hpp"

#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/ops.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

Tensor newton_schulz_orthogonalize(const Tensor& G, int steps, NewtonSchulzCoefficients k) {
  if (G.dim() != 2) throw std::invalid_argument("newton_schulz_orthogonalize: G must be a matrix");
  if (steps < 1) throw std::invalid_argument("newton_schulz_orthogonalize: steps must be at least 1");
  // Work with the wide orientation so that X X^T is the smaller Gram matrix.
  const bool tall = G.size(0) > G.size(1);
  Tensor X = tall ? transpose(G) : G;
  X = X / (sqrt(sum(square(X))) + 1e-7);
  for (int i = 0; i < steps; ++i) {
    const Tensor A = matmul(X, X, false, true);
    const Tensor B = k.b * A + k.c * matmul(A, A);
    X = k.a * X + matmul(B, X);
  }
  return tall ? transpose(X) : X;
}

Muon::Muon(std::vector<Tensor> parameters, double learning_rate, MuonConfig config)
    : params_(std::move(parameters)), lr_(learning_rate), config_(config) {
  if (!(config_.momentum >= 0 && config_.momentum < 1)) throw std::invalid_argument("Muon: momentum must be in [0, 1)");
  if (config_.ns_steps < 1) throw std::invalid_argument("Muon: ns_steps must be at least 1");
  if (!(config_.weight_decay >= 0)) throw std::invalid_argument("Muon: weight_decay must be >= 0");
  if (params_.empty()) throw std::invalid_argument("Muon: no parameters");
  for (const auto& p : params_)
    if (!p.defined() || !p.requires_grad() || !p.is_leaf())
      throw std::invalid_argument("Muon: parameters must be leaf tensors that require grad");
  set_learning_rate(learning_rate);
  momentum_.resize(params_.size());
}

void Muon::set_learning_rate(double learning_rate) {
  if (!std::isfinite(learning_rate) || learning_rate < 0) throw std::invalid_argument("Muon: learning rate must be finite and >= 0");
  lr_ = learning_rate;
}

void Muon::step(const std::vector<Tensor>& gradients) {
  if (gradients.size() != params_.size()) throw std::invalid_argument("Muon::step: one gradient per parameter");
  NoGradGuard guard;
  ++step_;
  const double mu = config_.momentum;
  for (std::size_t i = 0; i < params_.size(); ++i) {
    if (!gradients[i].defined()) continue;
    Tensor& w = params_[i];
    if (gradients[i].shape() != w.shape()) throw std::invalid_argument("Muon::step: gradient shape mismatch");
    const Tensor g = cast(gradients[i].detach(), w.dtype());
    momentum_[i] = momentum_[i].defined() ? mu * momentum_[i] + g : g;
    const Tensor u = config_.nesterov ? g + mu * momentum_[i] : momentum_[i];
    Tensor next = w.detach();
    if (config_.weight_decay > 0) next = next - lr_ * config_.weight_decay * next;
    if (w.dim() == 2) {
      const double scale = std::sqrt(std::max(1.0, static_cast<double>(w.size(0)) / static_cast<double>(w.size(1))));
      next = next - lr_ * scale * newton_schulz_orthogonalize(u, config_.ns_steps, config_.coefficients);
    } else {
      next = next - lr_ * u;
    }
    w.copy_(next);
  }
}

void Muon::minimize(const Tensor& loss) {
  GradOptions options;
  options.allow_unused = true;
  step(grad({loss}, params_, {}, options));
}

Lookahead::Lookahead(std::vector<Tensor> parameters, int k, double alpha) : params_(std::move(parameters)), k_(k), alpha_(alpha) {
  if (k < 1) throw std::invalid_argument("Lookahead: k must be at least 1");
  if (!(alpha > 0 && alpha <= 1)) throw std::invalid_argument("Lookahead: alpha must be in (0, 1]");
  if (params_.empty()) throw std::invalid_argument("Lookahead: no parameters");
  for (const auto& p : params_) {
    if (!p.defined()) throw std::invalid_argument("Lookahead: undefined parameter");
    slow_.push_back(p.detach().clone());
  }
}

bool Lookahead::after_step() {
  if (++count_ < k_) return false;
  count_ = 0;
  NoGradGuard guard;
  for (std::size_t i = 0; i < params_.size(); ++i) {
    slow_[i] = slow_[i] + alpha_ * (params_[i].detach() - slow_[i]);
    params_[i].copy_(slow_[i]);
  }
  return true;
}

std::vector<Tensor> Lookahead::slow_weights() const {
  std::vector<Tensor> out;
  for (const auto& s : slow_) out.push_back(s.clone());
  return out;
}

SamResult sam_gradients(const std::vector<Tensor>& parameters, const std::function<Tensor()>& loss, double rho) {
  if (!(rho >= 0)) throw std::invalid_argument("sam_gradients: rho must be >= 0");
  if (parameters.empty()) throw std::invalid_argument("sam_gradients: no parameters");
  GradOptions options;
  options.allow_unused = true;
  SamResult r;
  r.loss = loss();
  const auto g = grad({r.loss}, parameters, {}, options);
  double norm2 = 0;
  for (const auto& gi : g)
    if (gi.defined()) norm2 += sum(square(cast(gi, DType::Float64))).item();
  const double scale = norm2 > 0 ? rho / std::sqrt(norm2) : 0.0;
  std::vector<Tensor> saved;
  {
    NoGradGuard guard;
    for (std::size_t i = 0; i < parameters.size(); ++i) {
      saved.push_back(parameters[i].detach().clone());
      if (g[i].defined()) Tensor(parameters[i]).copy_(parameters[i].detach() + scale * g[i]);
    }
  }
  r.sharp_loss = loss();
  r.gradients = grad({r.sharp_loss}, parameters, {}, options);
  {
    NoGradGuard guard;
    for (std::size_t i = 0; i < parameters.size(); ++i) Tensor(parameters[i]).copy_(saved[i]);
  }
  r.loss = r.loss.detach(), r.sharp_loss = r.sharp_loss.detach();
  return r;
}

}  // namespace datamunge::nn
