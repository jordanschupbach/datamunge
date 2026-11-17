#pragma once
/// @file continuous_control.hpp
/// @brief Explicit source-named continuous-action graphs, densities and recurrent/off-policy Q
/// endpoints.
#include <cmath>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Internal shape validation for source continuous/off-policy model graphs.
namespace control_detail {

/// @brief Validate the source Float32 CPU batch and bounded output/hidden widths.
///
/// @param x Nonempty Float32 CPU[B,D], values not scanned.
/// @param actions Positive output/action count at most4096.
/// @param hidden Positive hidden width at most4096.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
inline void Input(const Tensor& x, int actions, int hidden) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dtype() != DType::Float32 ||
      x.dim() != 2 || x.size(0) < 1 || x.size(1) < 1 || actions < 1 || hidden < 1 ||
      actions > 4096 || hidden > 4096)
    throw std::invalid_argument(
        "nonempty Float32 CPU [B,D] and bounded positive model widths required");
}

/// @brief Validate source Gaussian broadcasting geometry without changing data.
///
/// @param mean Nonempty Float32 CPU[B,D].
/// @param log_std Float32 CPU[B,D] or[D].
/// @param value Float32 CPU[B,D], same shape as mean.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
inline void GaussianShapes(const Tensor& mean, const Tensor& log_std, const Tensor& value) {
  Input(mean, 1, 1);
  Input(value, 1, 1);
  if (mean.shape() != value.shape() || !log_std.defined() ||
      log_std.device().backend != compute::Backend::CPU || log_std.dtype() != mean.dtype() ||
      (log_std.shape() != mean.shape() && log_std.shape() != Shape{mean.size(1)}))
    throw std::invalid_argument("aligned Gaussian value/mean[B,D] and log-std[B,D] or[D] required");
}
} // namespace control_detail

/// @brief Build the source two-ReLU-hidden-layer deterministic tanh actor.
///
/// Names name/mlp/dense0,name/mlp/dense1,name/out; source Pendulum preset has17153 parameters.
/// Calls may advance store initialization RNG. No objective, sampling, optimizer or update loop.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param name Caller parameter prefix, e.g.pi or q.
/// @param obs Nonempty Float32 CPU[B,D], values not scanned for finiteness.
/// @param act_dim Output/action width in[1,4096].
/// @param hidden Two hidden widths in[1,4096], source default128.
/// @return Live Float32 actions[B,A] in[-1,1]; existing leaves and input unchanged.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto action=n::models::DeterministicActor(p,"pi",n::zeros({2,3}),1,128);
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see GaussianActor(), QNetwork()
inline Tensor DeterministicActor(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      obs,
    int                act_dim,
    int                hidden) {
  control_detail::Input(obs, act_dim, hidden);
  const Tensor h = mlp(p, name + "/mlp", obs, {hidden, hidden});
  return tanh(dense(p, name + "/out", h, act_dim));
}

// Gaussian actor: mean and (clipped) log-std heads on a shared trunk.

/// @brief Live diagonal-Gaussian parameter endpoints; no sampling or optimizer state.
///
/// Tensor copies share native storage; network-created tensors retain their parameter derivative
/// paths.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::models::Gaussian g{n::zeros({2,1}),n::zeros({2,1})};
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see GaussianActor(), SquashedSample()
struct Gaussian {
  Tensor mean;    ///< Live means[B,D].
  Tensor log_std; ///< Live log-standard-deviations[B,D], or[D] for compatible broadcast callers.
};

/// @brief Build source mean and clipped log-std heads on a shared ReLU trunk.
///
/// Named leaves are name/mlp/dense0,name/mlp/dense1,name/mean,name/log_std.
/// Geometry/bounds validate before lazy initialization; no samples or existing parameters are
/// changed. The caller owns RNG, objective, temperature, optimizers and fitting schedule.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param name Caller parameter prefix, e.g.pi or q.
/// @param obs Nonempty Float32 CPU[B,D], values not scanned for finiteness.
/// @param act_dim Output/action width in[1,4096].
/// @param hidden Two hidden widths in[1,4096], source default128.
/// @param log_std_min Finite lower clipping bound, default-5.
/// @param log_std_max Finite upper clipping bound at least the lower bound, default2.
/// @return Live Float32 mean/log_std[B,A]; clipping derivatives follow native clip.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto g=n::models::GaussianActor(p,"pi",n::zeros({2,3}),1,128);
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see SquashedSample(), GaussianLogProb()
inline Gaussian GaussianActor(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      obs,
    int                act_dim,
    int                hidden,
    float              log_std_min = -5.f,
    float              log_std_max = 2.f) {
  control_detail::Input(obs, act_dim, hidden);
  if (!std::isfinite(log_std_min) || !std::isfinite(log_std_max) || log_std_max < log_std_min)
    throw std::invalid_argument("finite ordered log-std bounds required");
  const Tensor h = mlp(p, name + "/mlp", obs, {hidden, hidden});
  Gaussian     g;
  g.mean    = dense(p, name + "/mean", h, act_dim);
  g.log_std = clip(dense(p, name + "/log_std", h, act_dim), log_std_min, log_std_max);
  return g;
}

// Reparameterised sample u = mean + std * eps, squashed to a = tanh(u), with
//   log pi(a|s) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2)
// (Haarnoja et al. 2018, appendix C). `eps` is [B, act_dim] standard normal.

/// @brief Live reparameterized tanh-action and corrected-density endpoints.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::models::Gaussian g{n::zeros({2,1}),n::zeros({2,1})};auto
/// sample=n::models::SquashedSample(g,n::zeros({2,1}));
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see SquashedSample()
struct Squashed {
  Tensor action; ///< Live tanh-squashed actions[B,D].
  Tensor logp;   ///< Live source-corrected log-density[B].
};

/// @brief Reparameterize and tanh-squash a caller-supplied diagonal-Gaussian noise sample.
///
/// Preserves the source density constant3.14159265 and Jacobian floor1e-6. No RNG is consumed.
/// logp is sum(-.5*eps^2-log_std-.5*log(2*pi))-sum(log(1-action^2+1e-6)).
/// All primitive derivative paths, including through supplied noise, remain live; no optimizer is
/// owned.
///
/// @param g Float32 CPU mean[B,D], log_std[B,D] or[D]; IEEE finite behavior follows tensor
/// primitives.
/// @param eps Float32 CPU standard-normal noise[B,D], same shape as mean; caller supplies it.
/// @return Live action=tanh(mean+exp(log_std)*eps) and corrected logp[B].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::models::Gaussian g{n::zeros({2,1}),n::zeros({2,1})};auto
/// sample=n::models::SquashedSample(g,n::ones({2,1}));
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see GaussianLogProb()
inline Squashed SquashedSample(const Gaussian& g, const Tensor& eps) {
  control_detail::GaussianShapes(g.mean, g.log_std, eps);
  const Tensor u = g.mean + exp(g.log_std) * eps;
  Squashed     out;
  out.action = tanh(u);
  const Tensor gauss_logp =
      sum(-0.5 * square(eps) - (g.log_std + 0.5 * std::log(2 * 3.14159265)), {1});
  const Tensor correction = sum(log(1.0 - square(out.action) + 1e-6), {1});
  out.logp                = gauss_logp - correction;
  return out;
}

// log N(a; mean, exp(log_std)) summed over action dimensions, rowwise: [B].

/// @brief Evaluate the source unsquashed diagonal-Gaussian log-density rowwise.
///
/// Uses the source3.14159265 constant. This does not include tanh correction or clipped-bound point
/// masses. Values are not scanned; extreme log_std follows native exp/division IEEE behavior. No
/// RNG/state mutation.
///
/// @param a Nonempty Float32 CPU actions[B,D].
/// @param mean Same-shape Float32 CPU means[B,D].
/// @param log_std Float32 CPU log-standard-deviations[B,D] or[D].
/// @return Live summed Float32 log-density[B], retaining derivatives through all floating inputs.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto logp=n::models::GaussianLogProb(n::zeros({2,1}),n::zeros({2,1}),n::zeros({1}));
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see SquashedSample()
inline Tensor GaussianLogProb(const Tensor& a, const Tensor& mean, const Tensor& log_std) {
  control_detail::GaussianShapes(mean, log_std, a);
  const Tensor z = (a - mean) / exp(log_std);
  return sum(-0.5 * square(z) - (log_std + 0.5 * std::log(2 * 3.14159265)), {1});
}

// Q(s, a) as an MLP on the concatenated state and action: [B].

/// @brief Build the source state-action value MLP on concatenated native inputs.
///
/// Names name/mlp/dense0,name/mlp/dense1,name/out; source Pendulum preset has17281 parameters.
/// Two ReLU hidden layers; existing leaves unchanged, missing leaves initialize lazily. Caller owns
/// fitting.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param name Caller parameter prefix, e.g.pi or q.
/// @param obs Nonempty Float32 CPU[B,D], values not scanned for finiteness.
/// @param act Float32 CPU[B,A], same row count as obs, nonempty width.
/// @param hidden Two hidden widths in[1,4096], source default128.
/// @return Live scalar Q[B], including action-input derivatives needed by deterministic policy
/// gradients.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(1);auto q=n::models::QNetwork(p,"q",n::zeros({2,3}),n::zeros({2,1}),128);
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see DeterministicActor()
inline Tensor QNetwork(
    ParameterStore&    p,
    const std::string& name,
    const Tensor&      obs,
    const Tensor&      act,
    int                hidden) {
  control_detail::Input(obs, 1, hidden);
  control_detail::Input(act, 1, hidden);
  if (obs.size(0) != act.size(0))
    throw std::invalid_argument("aligned state/action batch required");
  const Tensor x = concat({obs, act}, 1);
  const Tensor h = mlp(p, name + "/mlp", x, {hidden, hidden});
  return squeeze(dense(p, name + "/out", h, 1), {1});
}

/// @brief Live categorical state-action critic endpoints.
struct ActionDistributionPath {
  Tensor logits; ///< Atom logits[B,Z].
  Tensor probs;  ///< Softmax atom probabilities[B,Z].
  Tensor q;      ///< Expected return[B] on supplied support.
};

/// @brief Build the source D4PG categorical state-action critic.
///
/// Source names critic/mlp and critic/logits; no projection, objective, optimizer or fitting
/// runner.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param state Nonempty Float32 CPU[B,D].
/// @param action Nonempty Float32 CPU[B,A], same batch count.
/// @param hidden Two hidden widths in[1,4096], source default128.
/// @param support Float32 CPU atom values[Z], Z in[2,4096]; values are caller-defined.
/// @return Live logits/probs[B,Z], expected Q[B]; action and parameter derivative paths retained.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(1);auto
/// path=n::models::DistributionalActionCritic(p,n::zeros({2,3}),n::zeros({2,1}),128,n::from_vector(std::vector<float>{-2,-1,0}));
/// @endcode
/// @par References
/// - Barth-Maron et al. (2018), D4PG, https://arxiv.org/abs/1804.08617 .
/// @see QNetwork()
inline ActionDistributionPath DistributionalActionCritic(
    ParameterStore& p,
    const Tensor&   state,
    const Tensor&   action,
    int             hidden,
    const Tensor&   support) {
  control_detail::Input(state, 1, hidden);
  control_detail::Input(action, 1, hidden);
  if (state.size(0) != action.size(0) || !support.defined() || support.dtype() != DType::Float32 ||
      support.device().backend != compute::Backend::CPU || support.dim() != 1 ||
      support.size(0) < 2 || support.size(0) > 4096)
    throw std::invalid_argument("aligned state/action and Float32 atom support required");
  auto h      = mlp(p, "critic/mlp", concat({state, action}, 1), {hidden, hidden});
  auto logits = dense(p, "critic/logits", h, support.size(0));
  auto probs  = softmax(logits);
  return {logits, probs, sum(probs * support, {1})};
}
/// @brief Live source continuous-PPO actor and separate critic endpoints.
struct GaussianValuePath {
  Tensor mean;    ///< Unbounded Gaussian means[B,A].
  Tensor log_std; ///< State-independent trainable log-standard-deviations[A].
  Tensor value;   ///< State values[B].
};

/// @brief Build the source continuous-PPO tanh actor and independent value trunk.
///
/// Source names actor/dense0,dense1,actor/mean,actor/log_std
/// initialized-.5,critic/dense0,dense1,critic/value. No clipping, squashing or sampling is
/// implicit; source clipping at the environment remains caller code. Calls may initialize missing
/// leaves; existing parameters and observations are unchanged.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param obs Nonempty Float32 CPU[B,D], values not scanned for finiteness.
/// @param actions Action count in[1,4096].
/// @param hidden Two actor/critic hidden widths in[1,4096], source default64.
/// @return Live means[B,A], state-independent log_std[A] and values[B].
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto path=n::models::GaussianValuePolicy(p,n::zeros({2,3}),1,64);
/// @endcode
/// @par References
/// - Schulman et al. (2017), PPO, https://arxiv.org/abs/1707.06347 .
/// @see GaussianLogProb()
inline GaussianValuePath
GaussianValuePolicy(ParameterStore& p, const Tensor& obs, int actions, int hidden) {
  control_detail::Input(obs, actions, hidden);
  auto h       = mlp(p, "actor", obs, {hidden, hidden}, Activation::Tanh);
  auto mean    = dense(p, "actor/mean", h, actions);
  auto log_std = p.get("actor/log_std", {actions}, init::constant(-.5));
  auto hv      = mlp(p, "critic", obs, {hidden, hidden}, Activation::Tanh);
  auto value   = squeeze(dense(p, "critic/value", hv, 1), {1});
  return {mean, log_std, value};
}
/// @brief Live source two-choice hybrid reaching policy endpoints.
struct HybridReachPath {
  Tensor logits;      ///< Direction logits[B,2].
  Tensor log_softmax; ///< Direction log-probabilities[B,2].
  Tensor probs;       ///< Direction probabilities[B,2].
  Tensor means;       ///< Per-direction magnitude means[B,2].
  Tensor log_std;     ///< Trainable per-direction log-standard-deviations[2].
};

/// @brief Build the source64-wide tanh two-choice parameterized-action policy.
///
/// Source names trunk/dense0,discrete,means,log_std; log_std starts at-1.
/// Caller generates task observations, samples direction/magnitudes and constructs the REINFORCE
/// objective. There is no task rollout or optimizer state here.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param obs Float32 CPU[B,2], columns current position and target.
/// @return Live direction probabilities and per-direction Gaussian magnitude parameters.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto path=n::models::HybridReachPolicy(p,n::zeros({2,2}));
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see HybridReachReward()
inline HybridReachPath HybridReachPolicy(ParameterStore& p, const Tensor& obs) {
  control_detail::Input(obs, 2, 64);
  if (obs.size(1) != 2)
    throw std::invalid_argument("hybrid reaching observations[B,2] required");
  auto h       = mlp(p, "trunk", obs, {64}, Activation::Tanh);
  auto logits  = dense(p, "discrete", h, 2);
  auto logp    = log_softmax(logits);
  auto probs   = softmax(logits);
  auto means   = dense(p, "means", h, 2);
  auto log_std = p.get("log_std", {2}, init::constant(-1));
  return {logits, logp, probs, means, log_std};
}

/// @brief Compute the source one-step parameterized reaching task reward.
///
/// No state or RNG is retained; the oracle can reach reward0 when the target lies within one step.
///
/// @param position Finite current position, source sampled in[-1,1].
/// @param target Finite target, source sampled in[-1,1].
/// @param direction Discrete0 means left,1 means right.
/// @param magnitude Finite Gaussian parameter, clamped to[0,1] before movement.
/// @return Float32 negative absolute final-position error, at most0; extreme finite inputs follow
/// Float32 arithmetic.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto reward=n::models::HybridReachReward(0,.5f,1,.5f);
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see HybridReachPolicy()
inline float HybridReachReward(float position, float target, int direction, float magnitude) {
  if (!std::isfinite(position) || !std::isfinite(target) || !std::isfinite(magnitude) ||
      direction < 0 || direction > 1)
    throw std::invalid_argument("finite hybrid reaching values and direction0/1 required");
  const float dirs[] = {-1, 1};
  return -std::abs(position + dirs[direction] * std::max(0.f, std::min(1.f, magnitude)) - target);
}
/// @brief Live source ACER actor/action-value endpoints.
struct AcerPolicyPath {
  Tensor logits; ///< Policy logits[B,A].
  Tensor logp;   ///< Policy log-probabilities[B,A].
  Tensor probs;  ///< Policy probabilities[B,A].
  Tensor q;      ///< Learned action values[B,A].
  Tensor v;      ///< sum(stop_gradient(probs)*q)[B]; no policy derivative through this expectation.
};

/// @brief Build the source shared tanh ACER policy and learned Q-value heads.
///
/// Names trunk/dense0,dense1,logits,q in source initialization order. Retrace, bias correction,
/// trust-region projection, replay and optimizer steps are explicit caller operations.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param obs Nonempty Float32 CPU[B,D], values not scanned for finiteness.
/// @param actions Action count in[1,4096].
/// @param hidden Two shared hidden widths in[1,4096], source default64.
/// @return Live actor/Q tensors with V=sum(detached_probs*Q); no policy derivative through V.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto path=n::models::AcerQPolicy(p,n::zeros({2,4}),2,64);
/// @endcode
/// @par References
/// - Lillicrap et al. (2016), https://arxiv.org/abs/1509.02971; Haarnoja et al. (2018),
/// https://arxiv.org/abs/1801.01290; Wang et al. (2017), https://arxiv.org/abs/1611.01224 .
/// @see GaussianValuePolicy()
inline AcerPolicyPath AcerQPolicy(ParameterStore& p, const Tensor& obs, int actions, int hidden) {
  control_detail::Input(obs, actions, hidden);
  auto h      = mlp(p, "trunk", obs, {hidden, hidden}, Activation::Tanh);
  auto logits = dense(p, "logits", h, actions);
  auto logp   = log_softmax(logits);
  auto probs  = exp(logp);
  auto q      = dense(p, "q", h, actions);
  auto v      = sum(probs.detach() * q, {1});
  return {logits, logp, probs, q, v};
}
/// @brief Live source recurrent-Q step endpoints, with caller-owned recurrent state.
struct RecurrentQPath {
  LstmState state; ///< New hidden/cell tensors[B,H].
  Tensor    q;     ///< New action values[B,A].
};

/// @brief Apply one source embedding/LSTM/action-head step without owning rollout state.
///
/// Source names embed (ReLU),lstm,q; initialization order and native LSTM defaults are unchanged.
/// Caller controls zero start, detach/burn-in, sequence unrolling, replay masks and optimizer
/// steps. No recurrent state is retained inside this function.
///
/// @param p Caller ParameterStore; missing named leaves initialize lazily in source order, existing
/// leaves reused.
/// @param input Nonempty Float32 CPU[B,D], source masked CartPole width2.
/// @param state Float32 CPU hidden/cell tensors[B,H], matching batch and hidden width.
/// @param hidden H in[1,4096], source64.
/// @param actions Action count in[1,4096], source2.
/// @return Live next LSTM state and Q[B,A], preserving derivatives through inputs, prior state and
/// parameters.
/// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
/// invalid.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto
/// path=n::models::RecurrentQStep(p,n::zeros({2,2}),n::lstm_zero_state(2,64),64,2);
/// @endcode
/// @par References
/// - Kapturowski et al. (2019), https://openreview.net/forum?id=r1lyTjAqYX .
/// @see AcerQPolicy()
inline RecurrentQPath RecurrentQStep(
    ParameterStore&  p,
    const Tensor&    input,
    const LstmState& state,
    int              hidden,
    int              actions) {
  control_detail::Input(input, actions, hidden);
  if (!state.h.defined() || !state.c.defined() || state.h.shape() != Shape{input.size(0), hidden} ||
      state.c.shape() != state.h.shape() || state.h.dtype() != input.dtype() ||
      state.c.dtype() != input.dtype() || state.h.device().backend != compute::Backend::CPU ||
      state.c.device().backend != compute::Backend::CPU)
    throw std::invalid_argument("aligned Float32 CPU LSTM state[B,H] required");
  auto embedded = dense(p, "embed", input, hidden, Activation::Relu);
  auto next     = lstm_step(p, "lstm", embedded, state);
  return {next, dense(p, "q", next.h, actions)};
}
} // namespace datamunge::nn::models
