#pragma once
/// @file model_based_control.hpp
/// @brief Optimizer-free dynamics, board-policy, MuZero and world-model graph primitives.
#include <cmath>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Internal geometry checks for source model-based-control graphs.
namespace planning_model_detail {

/// @brief Validate a nonempty Float32 CPU matrix without scanning values.
///
/// @param x Candidate[B,D] tensor.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
inline void Matrix(const Tensor& x) {
  if (!x.defined() || x.device().backend != compute::Backend::CPU || x.dtype() != DType::Float32 ||
      x.dim() != 2 || x.size(0) < 1 || x.size(1) < 1)
    throw std::invalid_argument("nonempty Float32 CPU matrix required");
}

/// @brief Validate one positive model width at most4096.
///
/// @param width Candidate hidden/output width.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
inline void Width(int width) {
  if (width < 1 || width > 4096)
    throw std::invalid_argument("positive model width at most4096 required");
}
} // namespace planning_model_detail
/// @brief Live per-member Gaussian prediction of state delta and reward.
///
/// Tensor copies share native storage and retain their live graphs; shapes are described per field.
/// @par Example
/// @code
/// dm::nn::models::DynamicsGaussian g{dm::nn::zeros({2,4}),dm::nn::zeros({2,4})};
/// @endcode
/// @see DynamicsMember(), DynamicsNext()
struct DynamicsGaussian {
  Tensor mean;   ///< Float32[B,D+1] means, final column reward.
  Tensor logvar; ///< Float32[B,D+1] softly bounded log-variances.
};

/// @brief Build the source probabilistic delta-state/reward member graph.
///
/// Names mlp/dense0,dense1,mean,logvar; final output column is reward.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param state Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param action Nonempty Float32 CPU[B,A], aligned batch.
/// @param hidden Two ReLU hidden widths in[1,4096], source128.
/// @return Live mean/logvar[B,D+1]; logvar uses source nested softplus bounds near[-10,.5].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto
/// out=n::models::DynamicsMember(p,n::zeros({2,3}),n::zeros({2,1}),128);
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline DynamicsGaussian
DynamicsMember(ParameterStore& p, const Tensor& state, const Tensor& action, int hidden) {
  planning_model_detail::Matrix(state);
  planning_model_detail::Matrix(action);
  planning_model_detail::Width(hidden);
  if (state.size(0) != action.size(0))
    throw std::invalid_argument("dynamics state/action batch mismatch");
  int64_t out_dim = state.size(1) + 1;
  auto    x       = concat({state, action}, 1);
  auto    h       = mlp(p, "mlp", x, {hidden, hidden}, Activation::Relu);
  auto    mean    = dense(p, "mean", h, out_dim);
  auto    raw     = dense(p, "logvar", h, out_dim);
  auto    bounded = .5 - softplus(.5 - raw);
  return {mean, -10. + softplus(bounded + 10.)};
}
/// @brief Live next-state and reward endpoints of an explicit model prediction.
///
/// Tensor copies share native storage and retain their live graphs; shapes are described per field.
/// @par Example
/// @code
/// dm::nn::models::DynamicsGaussian g{dm::nn::zeros({2,4}),dm::nn::zeros({2,4})};
/// auto prediction=dm::nn::models::DynamicsNext(dm::nn::ones({2,3}),g);
/// @endcode
/// @see DynamicsNext()
struct DynamicsPrediction {
  Tensor next_s; ///< Float32[B,D] next-state prediction.
  Tensor r;      ///< Float32[B] reward prediction.
};

/// @brief Convert a member Gaussian into explicit next-state/reward predictions.
///
/// No RNG is consumed or NoGradGuard imposed. Caller supplies noise and chooses whether to retain
/// gradients. All native primitive derivative paths remain live. No objective schedule, optimizer,
/// target synchronization, ensemble owner, environment collection or fitting/search loop is
/// retained. Inputs/existing leaves are unchanged; first calls may advance the store initializer.
/// Parameter dimensions must stay consistent on reuse.
///
/// @param state Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param prediction Float32 CPU mean/logvar[B,D+1].
/// @param noise Optional Float32 CPU[B,D+1] standard-normal sample; undefined selects mean.
/// @return Live next_s=state+predicted_delta[B,D], reward[B].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto x=n::zeros({2,3});auto
/// g=n::models::DynamicsMember(p,x,n::zeros({2,1}),128);auto out=n::models::DynamicsNext(x,g);
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline DynamicsPrediction
DynamicsNext(const Tensor& state, const DynamicsGaussian& prediction, const Tensor& noise = {}) {
  planning_model_detail::Matrix(state);
  planning_model_detail::Matrix(prediction.mean);
  planning_model_detail::Matrix(prediction.logvar);
  if (prediction.mean.shape() != Shape{state.size(0), state.size(1) + 1} ||
      prediction.logvar.shape() != prediction.mean.shape())
    throw std::invalid_argument("dynamics Gaussian geometry");
  auto out = prediction.mean;
  if (noise.defined()) {
    planning_model_detail::Matrix(noise);
    if (noise.shape() != out.shape())
      throw std::invalid_argument("dynamics noise shape");
    out = prediction.mean + exp(.5 * prediction.logvar) * noise;
  }
  return {
      state + slice_axis(out, 1, 0, state.size(1)),
      squeeze(slice_axis(out, 1, state.size(1), state.size(1) + 1), {1})};
}
/// @brief Live source AlphaZero board-policy and bounded-value outputs.
///
/// Tensor copies share native storage and retain their live graphs; shapes are described per field.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);
/// auto path=dm::nn::models::BoardPolicyValue(p,dm::nn::zeros({1,9}),9);
/// @endcode
/// @see BoardPolicyValue()
struct BoardPolicyValuePath {
  Tensor logp;  ///< Log-probabilities[B,A], legality masking belongs to search/caller.
  Tensor value; ///< Tanh state values[B], from the observation player's perspective.
};

/// @brief Build the source two-ReLU-layer AlphaZero policy/value graph.
///
/// Names trunk/dense0,dense1,policy,value. The caller/search masks illegal actions.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param board Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param actions Action count in[1,4096], source tic-tac-toe9.
/// @param hidden Two shared hidden widths in[1,4096], default64.
/// @return Live logp[B,A] and tanh value[B].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto out=n::models::BoardPolicyValue(p,n::zeros({2,9}),9);
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline BoardPolicyValuePath
BoardPolicyValue(ParameterStore& p, const Tensor& board, int actions, int hidden = 64) {
  planning_model_detail::Matrix(board);
  planning_model_detail::Width(actions);
  planning_model_detail::Width(hidden);
  auto h     = mlp(p, "trunk", board, {hidden, hidden}, Activation::Relu);
  auto logp  = log_softmax(dense(p, "policy", h, actions), 1);
  auto value = squeeze(tanh(dense(p, "value", h, 1)), {1});
  return {logp, value};
}

/// @brief Apply source per-row latent min-max scaling with a1e-6 denominator floor.
///
/// First/higher primitives follow min/max tie behavior. No parameters, RNG or mutation.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param state Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @return Live Float32[B,S] scaled states; constant rows become0.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto scaled=n::models::MuZeroScale(n::from_vector(std::vector<float>{2,4,6},{1,3}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline Tensor MuZeroScale(const Tensor& state) {
  planning_model_detail::Matrix(state);
  auto mn = min(state, {1}, true), mx = max(state, {1}, true);
  return (state - mn) / ((mx - mn) + 1e-6);
}

/// @brief Build the source64-wide observation-to-scaled-latent representation.
///
/// Names h/mlp/dense0,h/out. Hidden activation follows native mlp defaultReLU.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param obs Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param latent Latent width in[1,4096], default32.
/// @return Live scaled state[B,S].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto state=n::models::MuZeroRepresentation(p,n::zeros({2,4}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline Tensor MuZeroRepresentation(ParameterStore& p, const Tensor& obs, int latent = 32) {
  planning_model_detail::Matrix(obs);
  planning_model_detail::Width(latent);
  return MuZeroScale(dense(p, "h/out", mlp(p, "h/mlp", obs, {64}), latent));
}
/// @brief Live source MuZero latent dynamics endpoints.
///
/// Tensor copies share native storage and retain their live graphs; shapes are described per field.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);
/// auto path=dm::nn::models::MuZeroDynamics(p,dm::nn::zeros({1,32}),dm::nn::zeros({1,2}));
/// @endcode
/// @see MuZeroDynamics()
struct MuZeroDynamicsPath {
  Tensor state;  ///< Min-max-scaled latent states[B,S].
  Tensor reward; ///< Predicted scalar transition rewards[B].
};

/// @brief Build source latent-state/action dynamics and reward heads.
///
/// Names g/mlp/dense0,g/state,g/reward; source64-wide ReLU trunk.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param state Float32 CPU[B,S], width latent.
/// @param action_onehot Float32 CPU[B,A], aligned batch; one-hot semantics are caller-owned.
/// @param latent Latent width in[1,4096], default32.
/// @return Live scaled successor[B,S] and reward[B].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto
/// out=n::models::MuZeroDynamics(p,n::zeros({2,32}),n::one_hot(n::from_vector(std::vector<int>{0,1}),2));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline MuZeroDynamicsPath MuZeroDynamics(
    ParameterStore& p,
    const Tensor&   state,
    const Tensor&   action_onehot,
    int             latent = 32) {
  planning_model_detail::Matrix(state);
  planning_model_detail::Matrix(action_onehot);
  planning_model_detail::Width(latent);
  if (state.size(0) != action_onehot.size(0) || state.size(1) != latent)
    throw std::invalid_argument("MuZero latent/action geometry");
  auto h = mlp(p, "g/mlp", concat({state, action_onehot}, 1), {64});
  return {MuZeroScale(dense(p, "g/state", h, latent)), squeeze(dense(p, "g/reward", h, 1), {1})};
}
/// @brief Live source MuZero policy/value endpoints.
///
/// Tensor copies share native storage and retain their live graphs; shapes are described per field.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);
/// auto path=dm::nn::models::MuZeroPrediction(p,dm::nn::zeros({1,32}),2);
/// @endcode
/// @see MuZeroPrediction()
struct MuZeroPredictionPath {
  Tensor logits; ///< Unnormalized policy scores[B,A].
  Tensor value;  ///< Scalar latent-state values[B].
};

/// @brief Build source latent policy-logit and value prediction heads.
///
/// Names f/mlp/dense0,f/policy,f/value; source64-wide ReLU trunk.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param state Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param actions Action count in[1,4096], source2.
/// @return Live logits[B,A], value[B].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto out=n::models::MuZeroPrediction(p,n::zeros({2,32}),2);
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline MuZeroPredictionPath MuZeroPrediction(ParameterStore& p, const Tensor& state, int actions) {
  planning_model_detail::Matrix(state);
  planning_model_detail::Width(actions);
  auto h = mlp(p, "f/mlp", state, {64});
  return {dense(p, "f/policy", h, actions), squeeze(dense(p, "f/value", h, 1), {1})};
}

/// @brief Apply the optional source MuZero scalar value transform.
///
/// No autograd path to host floats, RNG or state mutation. Inverse arithmetic retains source
/// cancellation and rounding at small values; extreme finite values follow Float32 arithmetic.
///
/// @param x Finite source scalar.
/// @param enabled Enable transform; defaultfalse matches the CartPole source preset.
/// @return Source Float32 sign(x)*(sqrt(abs(x)+1)-1)+.001*x, or x when disabled.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto y=n::models::MuZeroValueTransform(10,true);
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see MuZeroScale()
inline float MuZeroValueTransform(float x, bool enabled = false) {
  if (!std::isfinite(x))
    throw std::invalid_argument("finite MuZero value required");
  if (!enabled)
    return x;
  return (x >= 0 ? 1.f : -1.f) * (std::sqrt(std::abs(x) + 1) - 1) + .001f * x;
}

/// @brief Invert the optional source Float32 MuZero transform algebra.
///
/// No autograd path to host floats, RNG or state mutation. Inverse arithmetic retains source
/// cancellation and rounding at small values; extreme finite values follow Float32 arithmetic.
///
/// @param y Finite source scalar.
/// @param enabled Enable transform; defaultfalse matches the CartPole source preset.
/// @return Source approximate Float32 inverse, or identity when disabled.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto y=n::models::MuZeroValueTransform(10,true);auto
/// x=n::models::MuZeroInverseValueTransform(y,true);
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see MuZeroScale()
inline float MuZeroInverseValueTransform(float y, bool enabled = false) {
  if (!std::isfinite(y))
    throw std::invalid_argument("finite MuZero value required");
  if (!enabled)
    return y;
  float sign = y >= 0 ? 1.f : -1.f;
  float t    = (std::sqrt(1 + 4 * .001f * (std::abs(y) + 1 + .001f)) - 1) / (2 * .001f);
  return sign * (t * t - 1);
}
/// @brief Live source binary-pixel VAE posterior endpoints.
///
/// Tensor copies share native storage and retain their live graphs; shapes are described per field.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);
/// auto posterior=dm::nn::models::WorldEncoder(p,dm::nn::zeros({1,288}));
/// @endcode
/// @see WorldEncoder()
struct WorldPosterior {
  Tensor z_mean;   ///< Posterior means[B,Z].
  Tensor z_logvar; ///< Unclipped posterior log-variances[B,Z].
};

/// @brief Build the source128-wide binary CartPole VAE posterior graph.
///
/// Names enc,z_mean,z_logvar; posterior log-variance is not clipped.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param frame Float32 CPU[B,288] flattened12x24 source pixels.
/// @param latent Latent width in[1,4096], default8.
/// @return Live mean/log-variance[B,Z].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto out=n::models::WorldEncoder(p,n::zeros({2,288}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline WorldPosterior WorldEncoder(ParameterStore& p, const Tensor& frame, int latent = 8) {
  planning_model_detail::Matrix(frame);
  planning_model_detail::Width(latent);
  if (frame.size(1) != 288)
    throw std::invalid_argument("source world-model binary frame width288 required");
  auto h = dense(p, "enc", frame, 128, Activation::Relu);
  return {dense(p, "z_mean", h, latent), dense(p, "z_logvar", h, latent)};
}

/// @brief Build the source128-wide Bernoulli-logit frame decoder.
///
/// Names dec,dec_out; first hidden activationReLU. No image or file I/O.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param latent Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @return Live pixel logits[B,288]; caller applies Bernoulli objective or sigmoid.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);auto logits=n::models::WorldDecoder(p,n::zeros({2,8}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline Tensor WorldDecoder(ParameterStore& p, const Tensor& latent) {
  planning_model_detail::Matrix(latent);
  return dense(p, "dec_out", dense(p, "dec", latent, 128, Activation::Relu), 288);
}

/// @brief Build the source mixture-density recurrent output head.
///
/// Parameter prefix mdn; no mixture sample or RNG consumption.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param hidden Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param latent Predicted latent width in[1,4096], default8.
/// @param mixtures Mixture count in[1,256], default3.
/// @return Live raw head[B,K*(2*Z+1)] in means/log-stds/logits source order.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(1);auto raw=n::models::WorldMdnHead(p,n::zeros({2,64}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline Tensor
WorldMdnHead(ParameterStore& p, const Tensor& hidden, int latent = 8, int mixtures = 3) {
  planning_model_detail::Matrix(hidden);
  planning_model_detail::Width(latent);
  if (mixtures < 1 || mixtures > 256)
    throw std::invalid_argument("positive bounded mixture count required");
  return dense(p, "mdn", hidden, mixtures * (2 * latent + 1));
}

/// @brief Compute source diagonal-Gaussian-mixture negative log-density per row.
///
/// Log-stds clip to[-4,2]; source Float32 pi constant3.14159265f is retained.
/// Stable logsumexp mixes component log-densities; no fitting loop or parameters are owned.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param output Float32 CPU[B,K*(2*Z+1)] raw head.
/// @param target Float32 CPU[B,Z] next latent targets.
/// @param latent Z in[1,4096], default8.
/// @param mixtures K in[1,256], default3.
/// @return Live NLL[B]; caller chooses averaging, stopping targets and optimizer.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// auto nll=n::models::WorldMdnNll(n::zeros({2,51}),n::zeros({2,8}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline Tensor
WorldMdnNll(const Tensor& output, const Tensor& target, int latent = 8, int mixtures = 3) {
  planning_model_detail::Matrix(output);
  planning_model_detail::Matrix(target);
  planning_model_detail::Width(latent);
  if (mixtures < 1 || mixtures > 256 || target.shape() != Shape{output.size(0), latent} ||
      output.size(1) != mixtures * (2 * latent + 1))
    throw std::invalid_argument("aligned MDN output/target geometry required");
  int  K = mixtures, Z = latent;
  auto means     = reshape(slice_axis(output, 1, 0, K * Z), {-1, K, Z});
  auto log_std   = clip(reshape(slice_axis(output, 1, K * Z, 2 * K * Z), {-1, K, Z}), -4., 2.);
  auto log_pi    = log_softmax(slice_axis(output, 1, 2 * K * Z, 2 * K * Z + K), -1);
  auto diff      = (expand_dims(target, 1) - means) / exp(log_std);
  auto comp_logp = sum(-.5 * square(diff) - (log_std + .5 * std::log(2 * 3.14159265f)), {2});
  return -logsumexp(log_pi + comp_logp, {1});
}

/// @brief Apply the source latent/action LSTM step on caller recurrent state.
///
/// Parameter prefix lstm; inputs concat in latent/action order. No recurrent state is retained.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param latent Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @param action Nonempty Float32 CPU[B,A], aligned batch.
/// @param state Caller native LSTM hidden/cell state with aligned batch/dtype; native lstm_step
/// validates geometry.
/// @return Live next hidden/cell state; caller controls resets, detaches and unrolling.
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(1);auto
/// state=n::models::WorldMdnStep(p,n::zeros({2,8}),n::zeros({2,2}),n::lstm_zero_state(2,64));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline LstmState WorldMdnStep(
    ParameterStore&  p,
    const Tensor&    latent,
    const Tensor&    action,
    const LstmState& state) {
  planning_model_detail::Matrix(latent);
  planning_model_detail::Matrix(action);
  if (latent.size(0) != action.size(0))
    throw std::invalid_argument("world-model recurrent batch mismatch");
  return lstm_step(p, "lstm", concat({latent, action}, 1), state);
}

/// @brief Build the source64-wide ReLU imagined-feature value head.
///
/// Names v/h,v/out; actor/critic objectives and imagined lambda returns remain caller code.
/// All native primitive derivative paths remain live. No objective schedule, optimizer, target
/// synchronization, ensemble owner, environment collection or fitting/search loop is retained.
/// Inputs/existing leaves are unchanged; first calls may advance the store initializer. Parameter
/// dimensions must stay consistent on reuse.
///
/// @param p Caller-owned ParameterStore; missing source-named leaves initialize lazily, existing
/// leaves reused.
/// @param features Nonempty Float32 CPU[B,D]; tensor values are not scanned for finiteness.
/// @return Live scalar values[B].
/// @throws std::invalid_argument Invalid checked geometry or scalar configuration.
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(2);auto value=n::models::DreamerValue(p,n::zeros({2,80}));
/// @endcode
/// @par References
/// - Chua et al. (2018), PETS, https://arxiv.org/abs/1805.12114; Schrittwieser et al. (2020),
/// MuZero, https://doi.org/10.1038/s41586-020-03051-4; Ha and Schmidhuber (2018),
/// https://worldmodels.github.io/ .
/// @see DynamicsMember(), BoardPolicyValue(), MuZeroRepresentation(), WorldEncoder()
inline Tensor DreamerValue(ParameterStore& p, const Tensor& features) {
  planning_model_detail::Matrix(features);
  auto h = dense(p, "v/h", features, 64, Activation::Relu);
  return squeeze(dense(p, "v/out", h, 1), {1});
}
} // namespace datamunge::nn::models
