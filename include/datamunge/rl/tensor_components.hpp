#pragma once
/// @file tensor_components.hpp
/// @brief Explicit native MDP tensors and optimizer-free categorical policy graphs.
#include <cmath>
#include <datamunge/nn/einsum.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>
#include <datamunge/rl/environments/interface.hpp>
#include <limits>
namespace datamunge::rl::foundations {
/// @brief Internal validation for explicit native RL mathematical components.
namespace tensor_component_detail {

/// @brief Validate a nonempty finite Float32/Float64 CPU tensor.
///
/// @param x Tensor inspected without mutation.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
inline void Floating(const nn::Tensor& x) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU ||
      (x.dtype() != nn::DType::Float32 && x.dtype() != nn::DType::Float64) || x.numel() < 1)
    throw std::invalid_argument("finite floating CPU tensor required");
  for (double v : x.to_vector<double>())
    if (!std::isfinite(v))
      throw std::invalid_argument("finite floating tensor required");
}

/// @brief Validate a finite discount in[0,1].
///
/// @param gamma Candidate discount.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
inline void Discount(double gamma) {
  if (!std::isfinite(gamma) || gamma < 0 || gamma > 1)
    throw std::invalid_argument("discount outside[0,1]");
}
} // namespace tensor_component_detail
/// @brief Owning Float32 CPU tensors for a finite continuing-transition model.
/// @details P omits terminal continuation; R includes terminal rewards. No optimizer or training
/// state.
struct MdpTensorModel {
  /// Number of states.
  int S;
  /// Number of actions.
  int A;
  /// Continuation probabilities [S,A,S], freshly allocated by BuildMdpModel().
  nn::Tensor P;
  /// Expected immediate rewards [S,A], freshly allocated by BuildMdpModel().
  nn::Tensor R;
};

/// @brief Materialize a finite environment model without sampling or changing its state.
///
/// Rewards include terminal outcomes; P excludes them. Transition indices, masses and rewards are
/// validated. Zero-mass rows represent unavailable/absorbing transitions. No differentiation
/// through the environment.
///
/// @param env Discrete TabularEnv; positive S,A at most4096 and S*S*A at most16million. Transition
/// masses per row must be0 or1 within1e-5.
/// @return Owning Float32 CPU P[S,A,S] and R[S,A]; probabilities/rewards accumulate in source
/// order.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// dm::rl::environments::GridWorld env;auto model=f::BuildMdpModel(env);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see BellmanActionValues(), EvaluateMdpPolicy()
inline MdpTensorModel BuildMdpModel(const environments::TabularEnv& env) {
  const int S = env.num_states(), A = env.action_space().n;
  if (env.action_space().kind != environments::ActionSpace::kDiscrete || S < 1 || A < 1 ||
      S > 4096 || A > 4096 || int64_t(S) * S * A > 16000000)
    throw std::invalid_argument("bounded discrete MDP geometry required");
  MdpTensorModel model{S, A, nn::zeros({S, A, S}), nn::zeros({S, A})};
  auto           P = model.P.data_ptr<float>(), R = model.R.data_ptr<float>();
  for (int s = 0; s < S; ++s)
    for (int a = 0; a < A; ++a) {
      double mass = 0;
      for (const auto& t : env.transitions(s, a)) {
        if (!std::isfinite(t.prob) || t.prob < 0 || !std::isfinite(t.reward) || t.next_state < 0 ||
            t.next_state >= S)
          throw std::invalid_argument("valid finite MDP transition required");
        mass += t.prob;
        R[s * A + a] += t.prob * t.reward;
        if (!t.terminal)
          P[(s * A + a) * S + t.next_state] += t.prob;
      }
      if (mass != 0 && std::abs(mass - 1) > 1e-5)
        throw std::invalid_argument("MDP transition mass must be zero or one");
    }
  return model;
}

/// @brief Compute R+gamma*einsum("sat,t->sa",P,V) as a live native graph.
///
/// Shape and finite-value validation does not detach the graph. Caller supplies transition
/// semantics. No contraction, policy improvement or convergence loop is hidden here.
///
/// @param P Finite Float32/Float64 CPU continuation tensor [S,A,S].
/// @param R Finite CPU rewards [S,A], same dtype as P.
/// @param V Finite CPU state values [S], same dtype as P.
/// @param gamma Finite discount in[0,1].
/// @return Action values [S,A], retaining derivative paths through P,R,V; inputs unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto q=f::BellmanActionValues(dm::nn::zeros({2,2,2}),dm::nn::ones({2,2}),dm::nn::zeros({2}),.9);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see BuildMdpModel(), EvaluateMdpPolicy()
inline nn::Tensor
BellmanActionValues(const nn::Tensor& P, const nn::Tensor& R, const nn::Tensor& V, double gamma) {
  tensor_component_detail::Floating(P);
  tensor_component_detail::Floating(R);
  tensor_component_detail::Floating(V);
  tensor_component_detail::Discount(gamma);
  if (P.dim() != 3 || P.size(0) != P.size(2) || R.shape() != nn::Shape{P.size(0), P.size(1)} ||
      V.shape() != nn::Shape{P.size(0)} || P.dtype() != R.dtype() || P.dtype() != V.dtype())
    throw std::invalid_argument("MDP P/R/V geometry/dtype");
  return R + gamma * nn::einsum("sat,t->sa", P, V);
}

/// @brief Solve the deterministic-policy Bellman linear system.
///
/// One-hot selection and native solve preserve differentiation; policy indices are discrete.
/// A singular system, possible at gamma1, follows native solve failure behavior. No
/// policy-iteration loop.
///
/// @param P Finite Float32/Float64 CPU continuation probabilities [S,A,S].
/// @param R Finite same-dtype CPU expected rewards [S,A].
/// @param policy CPU Int32/Int64 action indices [S], each in[0,A).
/// @param gamma Finite discount in[0,1].
/// @return State values [S] solving (I-gamma*P_pi)V=R_pi; live graph through P,R.
/// @throws std::invalid_argument Invalid geometry, dtype, values or discount.
/// @throws std::runtime_error Native linear solve fails on a singular system.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// values=f::EvaluateMdpPolicy(dm::nn::zeros({2,2,2}),dm::nn::ones({2,2}),dm::nn::from_vector(std::vector<int>{0,1}),.9);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see BellmanActionValues()
inline nn::Tensor EvaluateMdpPolicy(
    const nn::Tensor& P,
    const nn::Tensor& R,
    const nn::Tensor& policy,
    double            gamma) {
  tensor_component_detail::Floating(P);
  tensor_component_detail::Floating(R);
  tensor_component_detail::Discount(gamma);
  if (!policy.defined() || policy.device().backend != compute::Backend::CPU || P.dim() != 3 ||
      P.size(0) != P.size(2) || R.shape() != nn::Shape{P.size(0), P.size(1)} ||
      P.dtype() != R.dtype() || policy.shape() != nn::Shape{P.size(0)} ||
      (policy.dtype() != nn::DType::Int32 && policy.dtype() != nn::DType::Int64))
    throw std::invalid_argument("MDP policy geometry/dtype");
  for (auto a : policy.to_vector<int64_t>())
    if (a < 0 || a >= P.size(1))
      throw std::invalid_argument("MDP policy action range");
  auto onehot = nn::cast(nn::one_hot(policy, P.size(1)), P.dtype());
  auto PP = nn::einsum("sa,sat->st", onehot, P), RR = nn::sum(onehot * R, {1});
  return nn::squeeze(
      nn::solve(nn::eye(P.size(0), P.dtype()) - gamma * PP, nn::expand_dims(RR, 1)), {1});
}
/// @brief Live endpoints of an explicit two-layer categorical policy graph.
struct CategoricalPolicyPath {
  /// Tanh hidden activations [B,H].
  nn::Tensor hidden;
  /// Unnormalized action scores [B,K].
  nn::Tensor logits;
  /// Log-softmax action probabilities [B,K].
  nn::Tensor log_probability;
};

/// @brief Build an optimizer-free tanh categorical policy from caller-owned parameters.
///
/// No parameters are initialized, mutated or retained; optimizer, objective, sampling and all
/// interaction loops belong to caller.
///
/// @param input Nonempty finite Float32/Float64 CPU batch [B,D].
/// @param W1 Same-dtype CPU input weights [D,H].
/// @param b1 Same-dtype CPU hidden bias [H].
/// @param W2 Same-dtype CPU action weights [H,K].
/// @param b2 Same-dtype CPU action bias [K].
/// @return Live hidden,logit,log-probability tensors retaining first/higher primitive derivatives.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// path=f::TanhCategoricalPolicy(dm::nn::ones({2,3}),dm::nn::ones({3,4}),dm::nn::zeros({4}),dm::nn::ones({4,2}),dm::nn::zeros({2}));
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see CategoricalActionLogProbability(), SampleCategoricalAction()
inline CategoricalPolicyPath TanhCategoricalPolicy(
    const nn::Tensor& input,
    const nn::Tensor& W1,
    const nn::Tensor& b1,
    const nn::Tensor& W2,
    const nn::Tensor& b2) {
  for (const auto& t : {input, W1, b1, W2, b2}) {
    tensor_component_detail::Floating(t);
    if (t.dtype() != input.dtype())
      throw std::invalid_argument("policy dtype mismatch");
  }
  if (input.dim() != 2 || W1.dim() != 2 || W2.dim() != 2 || input.size(1) != W1.size(0) ||
      b1.shape() != nn::Shape{W1.size(1)} || W2.size(0) != W1.size(1) ||
      b2.shape() != nn::Shape{W2.size(1)})
    throw std::invalid_argument("two-layer policy geometry");
  auto h      = nn::tanh(nn::matmul(input, W1) + b1);
  auto logits = nn::matmul(h, W2) + b2;
  return {h, logits, nn::log_softmax(logits)};
}

/// @brief Select each batch row chosen action log-probability with one-hot multiplication.
///
/// @param logp Nonempty finite Float32/Float64 CPU values [B,K]; negative-infinite masks are
/// rejected.
/// @param actions CPU Int32/Int64 [B], each in[0,K).
/// @return Live [B] tensor; derivative through logp only. Inputs unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto
/// chosen=f::CategoricalActionLogProbability(dm::nn::log_softmax(dm::nn::zeros({2,3})),dm::nn::from_vector(std::vector<int>{0,2}));
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see TanhCategoricalPolicy()
inline nn::Tensor CategoricalActionLogProbability(
    const nn::Tensor& logp,
    const nn::Tensor& actions) {
  tensor_component_detail::Floating(logp);
  if (!actions.defined() || actions.device().backend != compute::Backend::CPU || logp.dim() != 2 ||
      actions.shape() != nn::Shape{logp.size(0)} ||
      (actions.dtype() != nn::DType::Int32 && actions.dtype() != nn::DType::Int64))
    throw std::invalid_argument("categorical action geometry/dtype");
  for (auto a : actions.to_vector<int64_t>())
    if (a < 0 || a >= logp.size(1))
      throw std::invalid_argument("categorical action range");
  return nn::sum(logp * nn::cast(nn::one_hot(actions, logp.size(1)), logp.dtype()), {1});
}

/// @brief Sample one categorical action with source Float32 cumulative arithmetic.
///
/// No gradients, parameter updates or retained data. The pointer is read only during this call.
///
/// @param probs Borrowed contiguous Float32 probabilities, each finite in[0,1]; mass within3e-6
/// of1.
/// @param n Positive number of accessible entries.
/// @param rng Caller engine advanced by one uniform_real_distribution<float> draw.
/// @return First index whose cumulative mass exceeds draw; last index if roundoff leaves a gap.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// float p[]={.2f,.8f};std::mt19937_64 rng(1);auto arm=f::SampleCategoricalAction(p,2,rng);
/// @endcode
/// @par References
/// - Sutton and Barto, Reinforcement Learning, second edition, Chapter2
/// (http://incompleteideas.net/book/the-book-2nd.html).
/// @see TanhCategoricalPolicy()
inline int SampleCategoricalAction(const float* probs, int n, std::mt19937_64& rng) {
  if (!probs || n < 1)
    throw std::invalid_argument("nonempty categorical probabilities required");
  double total = 0;
  for (int a = 0; a < n; ++a) {
    if (!std::isfinite(probs[a]) || probs[a] < 0 || probs[a] > 1)
      throw std::invalid_argument("categorical probabilities outside[0,1]");
    total += probs[a];
  }
  if (std::abs(total - 1) > 3e-6)
    throw std::invalid_argument("normalized categorical probabilities required");
  float u = std::uniform_real_distribution<float>(0, 1)(rng), acc = 0;
  for (int a = 0; a < n; ++a) {
    acc += probs[a];
    if (u < acc)
      return a;
  }
  return n - 1;
}

/// @brief Pack finite native observation rows into owning Float32 CPU storage.
///
/// Caller vectors are copied; no lifetime ties or gradient path to host values.
///
/// @param rows Equal-width observations; each axis at most1million, total at most16million; empty
/// input allowed.
/// @return Independent tensor [N,D]; empty input has shape[0,0].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto x=f::ObservationMatrix({{1,2},{3,4}});
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see TanhCategoricalPolicy()
inline nn::Tensor ObservationMatrix(const std::vector<environments::Obs>& rows) {
  const int64_t n = rows.size(), d = n ? rows[0].size() : 0;
  if (n > 1000000 || d > 1000000 || n * d > 16000000)
    throw std::invalid_argument("bounded observation matrix required");
  std::vector<float> flat;
  flat.reserve(n * d);
  for (const auto& row : rows) {
    if (int64_t(row.size()) != d)
      throw std::invalid_argument("ragged observations");
    for (float v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("finite observations required");
    flat.insert(flat.end(), row.begin(), row.end());
  }
  return nn::from_vector(flat, {n, d});
}
} // namespace datamunge::rl::foundations
