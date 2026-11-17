#pragma once
/// @file actor_critic.hpp
/// @brief Explicit native categorical actor/value graphs and optimizer-free K-FAC algebra.
#include <cmath>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Internal source-policy graph shape/value validation.
namespace policy_graph_detail {

/// @brief Validate the source policy graph batch and layer geometry.
///
/// @param x Nonempty Float32 CPU [B,D]; values are not scanned.
/// @param actions Action count in[1,4096].
/// @param hidden Hidden width in[1,4096].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
inline void Input(const Tensor& x, int actions, int hidden) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dtype() != DType::Float32 ||
      x.dim() != 2 || x.size(0) < 1 || x.size(1) < 1 || actions < 1 || hidden < 1 ||
      actions > 4096 || hidden > 4096)
    throw std::invalid_argument(
        "nonempty Float32 CPU [B,D] and bounded positive policy widths required");
}

/// @brief Validate finite nonempty Float32 CPU matrix data.
///
/// @param x Candidate matrix.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
inline void FiniteMatrix(const Tensor& x) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dtype() != DType::Float32 ||
      x.dim() != 2 || x.numel() < 1)
    throw std::invalid_argument("nonempty Float32 CPU matrix required");
  for (float v : x.to_vector<float>())
    if (!std::isfinite(v))
      throw std::invalid_argument("finite matrix required");
}
} // namespace policy_graph_detail
/// @brief Live per-row actor and optional independent-critic endpoints.
struct ActorCritic {
  Tensor logits;  ///< Unnormalized action scores[B,A].
  Tensor logp;    ///< Log-softmax values[B,A].
  Tensor probs;   ///< Probabilities[B,A].
  Tensor entropy; ///< Negative sum probs*logp[B].
  Tensor value;   ///< Critic values[B], undefined when with_value=false.
};

/// @brief Build source-named categorical policy and optional separate value trunk.
///
/// Actor names actor/dense0,actor/dense1,actor/logits; critic names
/// critic/dense0,critic/dense1,critic/value. Parameter initialization order follows the source.
/// Calls may advance store initialization RNG. No objective, sampling, optimizer, target
/// computation or collection loop is owned.
///
/// @param p Caller-owned ParameterStore; missing named leaves initialize lazily, existing leaves
/// are reused.
/// @param obs Nonempty Float32 CPU [B,D]; values are not scanned for finiteness.
/// @param n_actions Action count in[1,4096].
/// @param hidden Two hidden widths in[1,4096], source default64.
/// @param with_value Also construct independent critic trunk, defaulttrue.
/// @param act Native hidden activation, defaultTanh; higher derivatives follow its smoothness.
/// @return Live logits/logp/probs/entropy and optional values; all inputs and existing leaves
/// unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto path=n::models::BuildActorCritic(p,n::zeros({2,4}),2,64);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
/// @see StandaloneValue(), KfacActor()
inline ActorCritic BuildActorCritic(
    ParameterStore& p,
    const Tensor&   obs,
    int             n_actions,
    int             hidden,
    bool            with_value = true,
    Activation      act        = Activation::Tanh) {
  policy_graph_detail::Input(obs, n_actions, hidden);
  ActorCritic  ac;
  const Tensor h = mlp(p, "actor", obs, {hidden, hidden}, act);
  ac.logits      = dense(p, "actor/logits", h, n_actions);
  ac.logp        = log_softmax(ac.logits);
  ac.probs       = exp(ac.logp);
  ac.entropy     = -sum(ac.probs * ac.logp, {1});
  if (with_value) {
    const Tensor hv = mlp(p, "critic", obs, {hidden, hidden}, act);
    ac.value        = squeeze(dense(p, "critic/value", hv, 1), {1});
  }
  return ac;
}

/// @brief Build the source separate tanh value critic with mlp/v parameter names.
///
/// Source TRPO/ACKTR critic graph. Native first/higher derivatives remain live.
/// The caller owns fitting and optimizer state.
///
/// @param p Caller store; missing mlp/dense0,mlp/dense1,v leaves initialize lazily.
/// @param obs Nonempty Float32 CPU[B,D], values not scanned.
/// @param hidden Two hidden widths in[1,4096].
/// @return Live state values[B]; existing leaves and input unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(1);auto value=n::models::StandaloneValue(p,n::zeros({2,4}),64);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
/// @see BuildActorCritic()
inline Tensor StandaloneValue(ParameterStore& p, const Tensor& obs, int hidden) {
  policy_graph_detail::Input(obs, 1, hidden);
  auto h = mlp(p, "mlp", obs, {hidden, hidden}, Activation::Tanh);
  return squeeze(dense(p, "v", h, 1), {1});
}
/// @brief Live manual-actor endpoints needed for caller-computed K-FAC statistics.
struct KfacActorPath {
  std::vector<Tensor> inputs; ///< Three layer inputs[B,in_l], retained for factor statistics.
  std::vector<Tensor> pre; ///< Three pre-activations[B,out_l], valid autograd intermediate targets.
  Tensor              logits; ///< Final pre-activation[B,A].
  Tensor              logp;   ///< Log-softmax action scores[B,A].
  Tensor              probs;  ///< Action probabilities[B,A].
};

/// @brief Initialize or reuse the source three-layer manual ACKTR actor parameters.
///
/// Weights use source Glorot-uniform bound sqrt(6/(in+out)); biases are zero.
/// Existing named leaves are reused according to ParameterStore::get() rules. No optimizer is
/// created.
///
/// @param p Caller-owned store; names W0,b0,W1,b1,W2,b2 in original creation order.
/// @param input_width Positive input width at most1million.
/// @param hidden Two hidden widths in[1,4096].
/// @param actions Action count in[1,4096].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);n::models::KfacActorParameters(p,4,64,2);
/// @endcode
/// @par References
/// - Wu et al. (2017), ACKTR, https://arxiv.org/abs/1708.05144 .
/// @see KfacActor()
inline void KfacActorParameters(ParameterStore& p, int input_width, int hidden, int actions) {
  if (input_width < 1 || input_width > 1000000 || hidden < 1 || hidden > 4096 || actions < 1 ||
      actions > 4096)
    throw std::invalid_argument("bounded positive K-FAC actor widths required");
  const int sizes[] = {input_width, hidden, hidden, actions};
  for (int l = 0; l < 3; ++l) {
    p.get(
        "W" + std::to_string(l),
        {sizes[l], sizes[l + 1]},
        init::uniform(std::sqrt(6.0 / (sizes[l] + sizes[l + 1]))));
    p.get("b" + std::to_string(l), {sizes[l + 1]}, init::zeros());
  }
}

/// @brief Apply an already-initialized three-layer tanh actor while exposing pre-activations.
///
/// Missing names or incompatible shapes follow ParameterStore/native matmul failure behavior.
/// Inputs and parameters remain unchanged; the caller computes factors, gradients and trust-region
/// steps.
///
/// @param p Read-only store containing source W0,b0,W1,b1,W2,b2 with consistent dimensions.
/// @param obs Nonempty Float32 CPU[B,D]; first two layers use tanh, final layer is linear.
/// @return Live inputs/pre-activations/logits/logp/probs; no parameter creation or RNG consumption.
/// @throws std::invalid_argument Invalid source input geometry.
/// @throws std::out_of_range A required named parameter is missing.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);n::models::KfacActorParameters(p,4,64,2);auto
/// path=n::models::KfacActor(p,n::zeros({2,4}));
/// @endcode
/// @par References
/// - Wu et al. (2017), ACKTR, https://arxiv.org/abs/1708.05144 .
/// @see KfacActorParameters(), KfacNaturalGradient()
inline KfacActorPath KfacActor(const ParameterStore& p, const Tensor& obs) {
  policy_graph_detail::Input(obs, 1, 1);
  KfacActorPath o;
  Tensor        h = obs;
  for (int l = 0; l < 3; ++l) {
    o.inputs.push_back(h);
    o.pre.push_back(
        bias_add(matmul(h, p.at("W" + std::to_string(l))), p.at("b" + std::to_string(l))));
    h = l < 2 ? tanh(o.pre.back()) : o.pre.back();
  }
  o.logits = h;
  o.logp   = log_softmax(h);
  o.probs  = exp(o.logp);
  return o;
}
/// @brief Owning/live endpoints of source factored Tikhonov preconditioning.
struct KfacNaturalPath {
  Tensor A_damped; ///< Input covariance plus pi*sqrt(damping)*I.
  Tensor G_damped; ///< Output covariance plus sqrt(damping)/pi*I.
  Tensor gradient; ///< Bias-augmented raw gradient[in+1,out].
  Tensor natural;  ///< Damped inverse-factor preconditioned gradient[in+1,out].
  float  pi;       ///< Source Float32 trace-balancing scalar, treated as a host constant.
};

/// @brief Solve the source factored damped K-FAC matrix equation without updating parameters.
///
/// pi=sqrt((trace(A)/(in+1))/max(1e-8,trace(G)/out)) is materialized in Float32 on host,
/// as in the source: its trace derivative is intentionally not part of the live graph. Other solve
/// paths remain native. Singular/indefinite caller factors follow native solve behavior. Factor
/// EMA, gradient scaling, KL quadratic, learning rate and parameter writes stay with the caller; no
/// optimizer is owned.
///
/// @param A Finite Float32 CPU covariance[in+1,in+1], strictly positive trace.
/// @param G Finite Float32 CPU covariance[out,out], nonnegative trace.
/// @param dW Finite Float32 CPU weight gradient[in,out].
/// @param db Finite Float32 CPU bias gradient[out].
/// @param damping Finite positive scalar, source default.01.
/// @return Damped factors, bias-augmented gradient and natural=A_damped^-1*gradient*G_damped^-1.
/// @throws std::invalid_argument Invalid finite geometry, trace or damping.
/// @throws std::runtime_error Native solve fails on a singular factor.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// result=n::models::KfacNaturalGradient(n::eye(3),n::eye(2),n::ones({2,2}),n::ones({2}),.01f);
/// @endcode
/// @par References
/// - Martens and Grosse (2015), Section6.3, https://arxiv.org/abs/1503.05671; Wu et al. (2017),
/// https://arxiv.org/abs/1708.05144 .
/// @see KfacActor()
inline KfacNaturalPath KfacNaturalGradient(
    const Tensor& A,
    const Tensor& G,
    const Tensor& dW,
    const Tensor& db,
    float         damping) {
  for (const auto& t : {A, G, dW})
    policy_graph_detail::FiniteMatrix(t);
  if (A.size(0) != A.size(1) || G.size(0) != G.size(1) || A.size(0) != dW.size(0) + 1 ||
      G.size(0) != dW.size(1) || !db.defined() || db.dtype() != DType::Float32 ||
      db.device().backend != compute::Backend::CPU || db.shape() != Shape{dW.size(1)} ||
      !std::isfinite(damping) || damping <= 0)
    throw std::invalid_argument(
        "aligned K-FAC covariance/gradient shapes and positive damping required");
  for (float v : db.to_vector<float>())
    if (!std::isfinite(v))
      throw std::invalid_argument("finite bias gradient required");
  float trA = trace(A).item<float>(), trG = trace(G).item<float>();
  if (!std::isfinite(trA) || !std::isfinite(trG) || trA <= 0 || trG < 0)
    throw std::invalid_argument("positive A trace and nonnegative G trace required");
  float pi      = std::sqrt((trA / A.size(0)) / std::max(1e-8f, trG / G.size(0)));
  auto  AA      = A + pi * std::sqrt(damping) * eye(A.size(0)),
       GG       = G + std::sqrt(damping) / pi * eye(G.size(0));
  auto gradient = concat({dW, reshape(db, {1, G.size(0)})}, 0);
  auto natural  = matmul(solve(AA, gradient), solve(GG, eye(G.size(0))));
  return {AA, GG, gradient, natural, pi};
}
} // namespace datamunge::nn::models
