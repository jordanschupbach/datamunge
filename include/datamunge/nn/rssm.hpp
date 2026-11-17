#pragma once
/// @file rssm.hpp
/// @brief Source recurrent state-space model primitives and live posterior sequence graphs.
#include <algorithm>
#include <cmath>
#include <datamunge/nn/layers.hpp>
#include <datamunge/nn/ops.hpp>
namespace datamunge::nn::models {
/// @brief Source RSSM graph dimensions and caller-owned objective configuration.
/// @par Example
/// @code
/// dm::nn::models::RssmConfig cfg;cfg.obs_dim=3;cfg.stoch=16;
/// @endcode
/// @see Rssm
struct RssmConfig {
  int   obs_dim   = 3;   ///< Observation width, default Pendulum3.
  int   act_dim   = 1;   ///< Action width, default1.
  int   det       = 64;  ///< Deterministic LSTM hidden/cell width.
  int   stoch     = 16;  ///< Diagonal-Gaussian stochastic latent width.
  int   hidden    = 64;  ///< Embedding/head/decoder hidden width.
  float free_nats = 1.f; ///< Caller loss threshold; Kl()/Observe() do not clamp to it.
};
/// @brief Live diagonal-Gaussian latent parameter endpoints.
/// @par Example
/// @code
/// dm::nn::models::RssmGauss gaussian{dm::nn::zeros({2,16}),dm::nn::zeros({2,16})};
/// @endcode
/// @see Rssm::Sample(), Rssm::Kl()
struct RssmGauss {
  datamunge::nn::Tensor mean; ///< Latent means[B,Z].
  datamunge::nn::Tensor
      log_std; ///< Latent log-standard-deviations[B,Z], clipped by source heads to[-4,1].
};
/// @brief Explicit caller-owned recurrent state tensors.
/// @par Example
/// @code
/// dm::nn::ParameterStore p(0);dm::nn::models::Rssm model(p,{});auto state=model.Zero(2);
/// @endcode
/// @see Rssm::Zero(), Rssm::Transition()
struct RssmState {
  datamunge::nn::Tensor h; ///< Deterministic LSTM hidden state[B,H].
  datamunge::nn::Tensor c; ///< Deterministic LSTM cell state[B,H].
  datamunge::nn::Tensor z; ///< Stochastic latent[B,Z].
};

/// @brief Borrow a ParameterStore to build the source LSTM recurrent state-space model.
///
/// The store must outlive this object and every graph that uses its leaves. Copies share the
/// borrowed store; Tensor copies share native storage. Missing named parameters initialize lazily
/// in exact source call order; existing leaves are reused. Graph methods may create parameters
/// despite const qualification. All matrices are nonempty Float32 CPU tensors; value finiteness
/// follows primitive operations. No optimizer, objective, random engine, environment collection or
/// fitting/planning loop is owned.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// n::ParameterStore p(0);n::models::Rssm model(p,{});auto state=model.Zero(2);
/// @endcode
/// @par References
/// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020), Dreamer,
/// https://arxiv.org/abs/1912.01603 .
/// @see RssmConfig, Observe(), Transition()
class Rssm {
 public:
  using Tensor = datamunge::nn::Tensor; ///< Native graph tensor shorthand.

  /// @brief Bind a caller store and validate source configuration dimensions.
  ///
  /// @param p Borrowed mutable parameter store, not retained by ownership.
  /// @param cfg All five widths in[1,4096]; free_nats finite and nonnegative.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see cfg()
  Rssm(datamunge::nn::ParameterStore& p, RssmConfig cfg) : p_(p), cfg_(cfg) {
    for (int width : {cfg.obs_dim, cfg.act_dim, cfg.det, cfg.stoch, cfg.hidden})
      if (width < 1 || width > 4096)
        throw std::invalid_argument("positive RSSM widths at most4096 required");
    if (!std::isfinite(cfg.free_nats) || cfg.free_nats < 0)
      throw std::invalid_argument("finite nonnegative free-nats configuration required");
  }

  /// @brief Borrow the immutable owned configuration copy.
  ///
  /// @return Const reference valid for this model lifetime.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto width=model.cfg().det;
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see feature_dim()
  const RssmConfig& cfg() const { return cfg_; }

  // Zero start state for a batch of `batch` sequences.

  /// @brief Create independent zero recurrent start tensors.
  ///
  /// No RNG or autograd path to host dimension.
  ///
  /// @param batch Positive batch count with batch*max(det,stoch) at most64million.
  /// @return Owning Float32 CPU h/c[B,H],z[B,Z], with no parameter initialization.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto state=model.Zero(2);
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  RssmState Zero(int batch) const {
    namespace nn = datamunge::nn;
    if (batch < 1 || int64_t(batch) * std::max(cfg_.det, cfg_.stoch) > 64000000)
      throw std::invalid_argument("positive bounded RSSM batch required");
    return {
        nn::zeros({batch, cfg_.det}), nn::zeros({batch, cfg_.det}), nn::zeros({batch, cfg_.stoch})};
  }

  /// @brief Build/apply the source observation embedding.
  ///
  /// Source parameter name embed.
  ///
  /// @param obs Float32 CPU[B,obs_dim].
  /// @return Live ReLU embedding[B,hidden].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// embedding=model.Embed(n::zeros({2,3}));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Tensor Embed(const Tensor& obs) const {
    Matrix(obs, cfg_.obs_dim);
    return datamunge::nn::dense(p_, "embed", obs, cfg_.hidden, datamunge::nn::Activation::Relu);
  }

  // Deterministic transition: new (h, c) from the previous state and action.

  /// @brief Apply source deterministic recurrent transition from prior latent and action.
  ///
  /// Concatenates previous z then action, source trans_in ReLU embedding then lstm. Caller supplies
  /// next z.
  ///
  /// @param prev Float32 CPU h/c[B,det],z[B,stoch].
  /// @param action Float32 CPU[B,act_dim], aligned batch.
  /// @return Live next LSTM h/c[B,det]; supplied state is unchanged.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// next=model.Transition(model.Zero(2),n::zeros({2,1}));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  datamunge::nn::LstmState Transition(const RssmState& prev, const Tensor& action) const {
    namespace nn = datamunge::nn;
    State(prev);
    Matrix(action, cfg_.act_dim);
    if (action.size(0) != prev.h.size(0))
      throw std::invalid_argument("RSSM action/state batch mismatch");
    const Tensor x  = nn::concat({prev.z, action}, 1);
    const Tensor xe = nn::dense(p_, "trans_in", x, cfg_.hidden, nn::Activation::Relu);
    return nn::lstm_step(p_, "lstm", xe, {prev.h, prev.c});
  }

  /// @brief Build the source latent prior conditional on deterministic state.
  ///
  /// Source prefix prior; log-std clips to[-4,1].
  ///
  /// @param h Float32 CPU[B,det].
  /// @return Live means/log-stds[B,stoch].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto prior=model.Prior(n::zeros({2,64}));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  RssmGauss Prior(const Tensor& h) const {
    Matrix(h, cfg_.det);
    return Head("prior", h);
  }

  /// @brief Build the source latent posterior conditional on state and embedded observation.
  ///
  /// Source prefix posterior; concatenation order h,embed; log-std clips to[-4,1].
  ///
  /// @param h Float32 CPU[B,det].
  /// @param embed Float32 CPU[B,hidden], same batch.
  /// @return Live means/log-stds[B,stoch].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// post=model.Posterior(n::zeros({2,64}),model.Embed(n::zeros({2,3})));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  RssmGauss Posterior(const Tensor& h, const Tensor& embed) const {
    Matrix(h, cfg_.det);
    Matrix(embed, cfg_.hidden);
    if (h.size(0) != embed.size(0))
      throw std::invalid_argument("RSSM posterior batch mismatch");
    return Head("posterior", datamunge::nn::concat({h, embed}, 1));
  }

  /// @brief Reparameterize one explicit caller-supplied latent noise sample.
  ///
  /// No RNG is sampled or stored. Inputs remain unchanged.
  ///
  /// @param g Matching Float32 CPU means/log-stds[B,stoch].
  /// @param eps Matching Float32 CPU[B,stoch] standard-normal sample from caller.
  /// @return Live mean+exp(log_std)*eps[B,stoch], retaining all floating-input derivatives.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto g=model.Prior(n::zeros({2,64}));auto
  /// z=model.Sample(g,n::zeros({2,16}));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Tensor Sample(const RssmGauss& g, const Tensor& eps) const {
    Matrix(g.mean, cfg_.stoch);
    Matrix(g.log_std, cfg_.stoch);
    Matrix(eps, cfg_.stoch);
    if (g.mean.shape() != g.log_std.shape() || g.mean.shape() != eps.shape())
      throw std::invalid_argument("RSSM sample shape mismatch");
    return g.mean + datamunge::nn::exp(g.log_std) * eps;
  }

  /// @brief Concatenate deterministic hidden state and stochastic latent.
  ///
  /// The cell tensor is validated but is not concatenated into features.
  ///
  /// @param st Matching Float32 CPU h/c[B,det],z[B,stoch].
  /// @return Live features[B,det+stoch] in h,z order.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// features=model.Features(model.Zero(2));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Tensor Features(const RssmState& st) const {
    State(st);
    return datamunge::nn::concat({st.h, st.z}, 1);
  }

  /// @brief Read the combined deterministic/stochastic feature width.
  ///
  /// No graph construction or mutation.
  ///
  /// @return det+stoch.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto width=model.feature_dim();
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  int feature_dim() const { return cfg_.det + cfg_.stoch; }

  /// @brief Decode source observation predictions from latent features.
  ///
  /// Names dec_obs/h (ReLU),dec_obs/out; no probabilistic output scale is implicit.
  ///
  /// @param features Float32 CPU[B,det+stoch].
  /// @return Live reconstructed observations[B,obs_dim].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// observation=model.DecodeObs(model.Features(model.Zero(2)));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Tensor DecodeObs(const Tensor& features) const {
    Matrix(features, feature_dim());
    namespace nn   = datamunge::nn;
    const Tensor h = nn::dense(p_, "dec_obs/h", features, cfg_.hidden, nn::Activation::Relu);
    return nn::dense(p_, "dec_obs/out", h, cfg_.obs_dim);
  }

  /// @brief Decode source scalar reward predictions from latent features.
  ///
  /// Names dec_rew/h (ReLU),dec_rew/out; source reward scaling remains caller code.
  ///
  /// @param features Float32 CPU[B,det+stoch].
  /// @return Live rewards[B].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// reward=model.DecodeReward(model.Features(model.Zero(2)));
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Tensor DecodeReward(const Tensor& features) const {
    Matrix(features, feature_dim());
    namespace nn   = datamunge::nn;
    const Tensor h = nn::dense(p_, "dec_rew/h", features, cfg_.hidden, nn::Activation::Relu);
    return nn::squeeze(nn::dense(p_, "dec_rew/out", h, 1), {1});
  }
  // KL(q || p) per row for diagonal Gaussians.

  /// @brief Compute source rowwise diagonal-Gaussian KL(q||p).
  ///
  /// Uses exp(2*(q.log_std-p.log_std)) and squared standardized mean displacement in source order.
  /// Primitive first/higher derivatives are retained; callers apply free-nats/reduction/objective
  /// weights.
  ///
  /// @param q Float32 CPU mean/log-std[B,stoch].
  /// @param p Same-shape Float32 CPU prior mean/log-std.
  /// @return Live KL[B], not clamped by free_nats.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto g=model.Prior(n::zeros({2,64}));auto
  /// kl=model.Kl(g,g);
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Tensor Kl(const RssmGauss& q, const RssmGauss& p) const {
    namespace nn = datamunge::nn;
    for (const auto& t : {q.mean, q.log_std, p.mean, p.log_std}) {
      Matrix(t, cfg_.stoch);
      if (t.shape() != q.mean.shape())
        throw std::invalid_argument("RSSM Gaussian KL shape mismatch");
    }
    const Tensor d         = q.log_std - p.log_std;
    const Tensor var_ratio = nn::exp(2.0 * d);
    const Tensor mean_term = nn::square(q.mean - p.mean) / nn::exp(2.0 * p.log_std);
    return nn::sum(0.5 * ((var_ratio + mean_term) - (1.0 + 2.0 * d)), {1});
  }

  // Unroll the model over a batch of sequences with the posterior (training
  // / filtering). obs_t: T tensors [B, obs]; act_t: T tensors [B, act] (the
  // action taken *after* obs_t); eps_t: T noise tensors [B, stoch].

  /// @brief Retain the native graph endpoints of one explicit posterior sequence.
  ///
  /// Tensor copies share storage/graphs; containers own their handles, not independent parameter
  /// stores.
  ///
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// path=model.Observe(model.Zero(1),{n::zeros({1,3})},{n::zeros({1,1})},{n::zeros({1,16})});
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see Observe()
  struct Unrolled {
    std::vector<RssmState> states; ///< Live posterior states, one per observation.
    std::vector<Tensor>    kl;     ///< Live per-step KL(q||p)[B].
    std::vector<Tensor>    recon;  ///< Live per-step reconstructed observations[B,D].
    std::vector<Tensor>    reward; ///< Live per-step reward predictions[B].
  };

  /// @brief Build a live source posterior/filtering sequence graph from an explicit start state.
  ///
  /// Lists and batches validate before parameter creation. Empty aligned lists yield empty outputs.
  /// At t0 the supplied deterministic h/c is used; for t>0 transition uses action[t-1]. The final
  /// action is therefore unused, as in source. Prior precedes posterior; source lazy-initialization
  /// order is retained. This is model graph unrolling, without environment steps, target loss or
  /// optimizer updates.
  ///
  /// @param start Valid Float32 CPU h/c/z start state.
  /// @param obs_t T Float32 CPU[B,obs_dim] tensors, T at most1million.
  /// @param act_t T Float32 CPU[B,act_dim] tensors; action t is taken after observation t.
  /// @param eps_t T Float32 CPU[B,stoch] caller noise tensors.
  /// @return Live per-step posterior states, KL, reconstructed observations and rewards.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// n::ParameterStore p(0);n::models::Rssm model(p,{});auto
  /// path=model.Observe(model.Zero(2),{n::zeros({2,3})},{n::zeros({2,1})},{n::zeros({2,16})});
  /// @endcode
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see RssmConfig, Zero(), Transition(), Observe()
  Unrolled Observe(
      const RssmState&           start,
      const std::vector<Tensor>& obs_t,
      const std::vector<Tensor>& act_t,
      const std::vector<Tensor>& eps_t) const {
    State(start);
    if (obs_t.size() != act_t.size() || obs_t.size() != eps_t.size() || obs_t.size() > 1000000)
      throw std::invalid_argument("aligned bounded RSSM sequence lists required");
    for (size_t t = 0; t < obs_t.size(); ++t) {
      Matrix(obs_t[t], cfg_.obs_dim);
      Matrix(act_t[t], cfg_.act_dim);
      Matrix(eps_t[t], cfg_.stoch);
      if (obs_t[t].size(0) != start.h.size(0) || act_t[t].size(0) != start.h.size(0) ||
          eps_t[t].size(0) != start.h.size(0))
        throw std::invalid_argument("RSSM sequence batch mismatch");
    }
    Unrolled  u;
    RssmState st = start;
    for (size_t t = 0; t < obs_t.size(); ++t) {
      if (t > 0) {
        const datamunge::nn::LstmState hc = Transition(st, act_t[t - 1]);
        st.h = hc.h, st.c = hc.c;
      }
      const RssmGauss prior = Prior(st.h), post = Posterior(st.h, Embed(obs_t[t]));
      st.z = Sample(post, eps_t[t]);
      u.states.push_back(st);
      u.kl.push_back(Kl(post, prior));
      const Tensor f = Features(st);
      u.recon.push_back(DecodeObs(f));
      u.reward.push_back(DecodeReward(f));
    }
    return u;
  }

 private:
  /// @brief Validate one source RSSM matrix geometry.
  ///
  /// @param x Nonempty Float32 CPU[B,width], values not scanned.
  /// @param width Expected final-axis width.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  static void Matrix(const Tensor& x, int width) {
    if (!x.defined() || x.device().backend != datamunge::compute::Backend::CPU ||
        x.dtype() != datamunge::nn::DType::Float32 || x.dim() != 2 || x.size(0) < 1 ||
        x.size(1) != width)
      throw std::invalid_argument("Float32 CPU RSSM matrix[B,width] required");
  }

  /// @brief Validate consistent source recurrent state shapes.
  ///
  /// @param state Float32 CPU h/c[B,det],z[B,stoch].
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  void State(const RssmState& state) const {
    Matrix(state.h, cfg_.det);
    Matrix(state.c, cfg_.det);
    Matrix(state.z, cfg_.stoch);
    if (state.h.shape() != state.c.shape() || state.h.size(0) != state.z.size(0))
      throw std::invalid_argument("RSSM state batch mismatch");
  }

  /// @brief Build the shared source ReLU/Gaussian latent head recipe.
  ///
  /// @param name Caller source prefix prior or posterior.
  /// @param x Valid Float32 CPU conditioning matrix.
  /// @return Live means and clipped log-stds[B,stoch].
  /// @par References
  /// - Hafner et al. (2019), PlaNet, https://arxiv.org/abs/1811.04551; Hafner et al. (2020),
  /// Dreamer, https://arxiv.org/abs/1912.01603 .
  /// @see Prior(), Posterior()
  RssmGauss Head(const std::string& name, const Tensor& x) const {
    namespace nn   = datamunge::nn;
    const Tensor h = nn::dense(p_, name + "/h", x, cfg_.hidden, nn::Activation::Relu);
    RssmGauss    g;
    g.mean    = nn::dense(p_, name + "/mean", h, cfg_.stoch);
    g.log_std = nn::clip(nn::dense(p_, name + "/log_std", h, cfg_.stoch), -4.0, 1.0);
    return g;
  }
  datamunge::nn::ParameterStore& p_;   ///< Borrowed caller store, which must outlive this model.
  RssmConfig                     cfg_; ///< Owned immutable-after-construction source configuration.
};

} // namespace datamunge::nn::models
