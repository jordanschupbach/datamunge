// datamunge::nn port of tflow_examples src/common/rl/rssm.h.
// A recurrent state-space model (Hafner et al. 2019, "RSSM") in a small
// form: a deterministic LSTM path h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) and a
// stochastic latent z_t with a prior p(z_t | h_t) and a posterior
// q(z_t | h_t, o_t). Decoders predict the observation and the reward from
// (h_t, z_t). PlaNet plans in this latent space; Dreamer learns an actor and
// a critic in it.
//
// Changes from the TF version: the model is eager — every method takes and
// returns nn::Tensor values and builds its weights in a ParameterStore on
// first use (the same names as tflow's nn::Params: "embed", "trans_in",
// "lstm", "prior/...", "posterior/...", "dec_obs/...", "dec_rew/..."), so
// calling a method twice shares its weights as the TF graph did. Gauss is
// rl::RssmGauss here (rl::Gaussian already names continuous.hpp's actor head).
#pragma once

#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <string>
#include <vector>

namespace rl {

struct RssmConfig {
  int obs_dim = 3, act_dim = 1;
  int det = 64;      // deterministic (LSTM) state size
  int stoch = 16;    // stochastic latent size
  int hidden = 64;
  float free_nats = 1.f;
};

struct RssmGauss {
  datamunge::nn::Tensor mean, log_std;
};
struct RssmState {
  datamunge::nn::Tensor h, c, z;  // [B, det], [B, det], [B, stoch]
};

class Rssm {
 public:
  using Tensor = datamunge::nn::Tensor;
  Rssm(datamunge::nn::ParameterStore& p, RssmConfig cfg) : p_(p), cfg_(cfg) {}
  const RssmConfig& cfg() const { return cfg_; }

  // Zero start state for a batch of `batch` sequences.
  RssmState Zero(int batch) const {
    namespace nn = datamunge::nn;
    return {nn::zeros({batch, cfg_.det}), nn::zeros({batch, cfg_.det}), nn::zeros({batch, cfg_.stoch})};
  }

  Tensor Embed(const Tensor& obs) const {
    return datamunge::nn::dense(p_, "embed", obs, cfg_.hidden, datamunge::nn::Activation::Relu);
  }

  // Deterministic transition: new (h, c) from the previous state and action.
  datamunge::nn::LstmState Transition(const RssmState& prev, const Tensor& action) const {
    namespace nn = datamunge::nn;
    const Tensor x = nn::concat({prev.z, action}, 1);
    const Tensor xe = nn::dense(p_, "trans_in", x, cfg_.hidden, nn::Activation::Relu);
    return nn::lstm_step(p_, "lstm", xe, {prev.h, prev.c});
  }
  RssmGauss Prior(const Tensor& h) const { return Head("prior", h); }
  RssmGauss Posterior(const Tensor& h, const Tensor& embed) const {
    return Head("posterior", datamunge::nn::concat({h, embed}, 1));
  }
  Tensor Sample(const RssmGauss& g, const Tensor& eps) const { return g.mean + datamunge::nn::exp(g.log_std) * eps; }
  Tensor Features(const RssmState& st) const { return datamunge::nn::concat({st.h, st.z}, 1); }
  int feature_dim() const { return cfg_.det + cfg_.stoch; }
  Tensor DecodeObs(const Tensor& features) const {
    namespace nn = datamunge::nn;
    const Tensor h = nn::dense(p_, "dec_obs/h", features, cfg_.hidden, nn::Activation::Relu);
    return nn::dense(p_, "dec_obs/out", h, cfg_.obs_dim);
  }
  Tensor DecodeReward(const Tensor& features) const {
    namespace nn = datamunge::nn;
    const Tensor h = nn::dense(p_, "dec_rew/h", features, cfg_.hidden, nn::Activation::Relu);
    return nn::squeeze(nn::dense(p_, "dec_rew/out", h, 1), {1});
  }
  // KL(q || p) per row for diagonal Gaussians.
  Tensor Kl(const RssmGauss& q, const RssmGauss& p) const {
    namespace nn = datamunge::nn;
    const Tensor d = q.log_std - p.log_std;
    const Tensor var_ratio = nn::exp(2.0 * d);
    const Tensor mean_term = nn::square(q.mean - p.mean) / nn::exp(2.0 * p.log_std);
    return nn::sum(0.5 * ((var_ratio + mean_term) - (1.0 + 2.0 * d)), {1});
  }

  // Unroll the model over a batch of sequences with the posterior (training
  // / filtering). obs_t: T tensors [B, obs]; act_t: T tensors [B, act] (the
  // action taken *after* obs_t); eps_t: T noise tensors [B, stoch].
  struct Unrolled {
    std::vector<RssmState> states;  // posterior states, one per step
    std::vector<Tensor> kl;         // per-step KL(q || p), [B]
    std::vector<Tensor> recon;      // decoded obs, [B, obs]
    std::vector<Tensor> reward;     // decoded reward, [B]
  };
  Unrolled Observe(const RssmState& start, const std::vector<Tensor>& obs_t, const std::vector<Tensor>& act_t,
                   const std::vector<Tensor>& eps_t) const {
    Unrolled u;
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
  RssmGauss Head(const std::string& name, const Tensor& x) const {
    namespace nn = datamunge::nn;
    const Tensor h = nn::dense(p_, name + "/h", x, cfg_.hidden, nn::Activation::Relu);
    RssmGauss g;
    g.mean = nn::dense(p_, name + "/mean", h, cfg_.stoch);
    g.log_std = nn::clip(nn::dense(p_, name + "/log_std", h, cfg_.stoch), -4.0, 1.0);
    return g;
  }
  datamunge::nn::ParameterStore& p_;
  RssmConfig cfg_;
};

}  // namespace rl
