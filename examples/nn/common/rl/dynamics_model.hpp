// datamunge::nn port of tflow_examples src/common/rl/dynamics_model.h.
// A probabilistic dynamics-model ensemble (Chua et al. 2018): each member is
// an MLP mapping (s, a) to a Gaussian over (s' - s, r) with a learned mean
// and log-variance, trained by negative log-likelihood. Used by the MPC,
// PETS and MBPO examples.
//
// Changes from the TF version: no session or placeholders — each member owns
// a ParameterStore (seeded seed + m, as tflow's nn::Params were) and an Adam
// (rl::Adam, no clipping, as in tflow); Train, Predict and TransitionSet::Sample
// take and return nn::Tensor values. Train steps the members one after the
// other on the same batch (they share no weights, so this equals TF's single
// session run). Forward(m, S, A) exposes a member's mean and bounded
// log-variance for callers that want them directly.
#pragma once

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "common/rl/dqn_common.hpp"
#include "common/rl/env.hpp"

namespace rl {

class DynamicsEnsemble {
 public:
  using Tensor = datamunge::nn::Tensor;

  DynamicsEnsemble(int members, int obs_dim, int act_dim, int hidden, float lr, uint64_t seed)
      : obs_dim_(obs_dim), hidden_(hidden) {
    namespace nn = datamunge::nn;
    for (int m = 0; m < members; ++m) {
      Member mem;
      mem.params = std::make_unique<nn::ParameterStore>(seed + m);
      members_.push_back(std::move(mem));
      {
        nn::NoGradGuard no_grad;
        Forward(m, nn::zeros({1, obs_dim}), nn::zeros({1, act_dim}));
      }
      members_.back().adam = std::make_unique<Adam>(*members_.back().params, lr);
    }
  }

  int members() const { return static_cast<int>(members_.size()); }

  struct Output {
    Tensor mean, logvar;  // [N, obs + 1] each: (s' - s, r)
  };
  // Member m's Gaussian over (s' - s, r) for states S [N, obs] and actions A [N, act].
  Output Forward(int m, const Tensor& S, const Tensor& A) const {
    namespace nn = datamunge::nn;
    nn::ParameterStore& p = *members_[m].params;
    const int out_dim = obs_dim_ + 1;
    const Tensor x = nn::concat({S, A}, 1);
    const Tensor h = nn::mlp(p, "mlp", x, {hidden_, hidden_}, nn::Activation::Relu);
    Output o;
    o.mean = nn::dense(p, "mean", h, out_dim);
    // Soft bounds on the log-variance keep the likelihood well behaved.
    const Tensor raw = nn::dense(p, "logvar", h, out_dim);
    const Tensor bounded = 0.5 - nn::softplus(0.5 - raw);
    o.logvar = -10.0 + nn::softplus(bounded + 10.0);
    return o;
  }

  // One gradient step for every member on the same batch; returns the mean
  // NLL and the mean squared error over members.
  std::pair<float, float> Train(const Tensor& S, const Tensor& A, const Tensor& S2, const Tensor& R) {
    namespace nn = datamunge::nn;
    const Tensor target = nn::concat({S2 - S, nn::expand_dims(R, 1)}, 1);
    float nll = 0, mse = 0;
    for (int m = 0; m < members(); ++m) {
      const Output o = Forward(m, S, A);
      const Tensor err = target - o.mean;
      const Tensor loss = nn::mean(nn::sum(nn::square(err) * nn::exp(-o.logvar) + o.logvar, {1}), {0});
      const float mse_m = nn::mean(nn::sum(nn::square(err.detach()), {1}), {0}).item<float>();
      const float loss_m = loss.item<float>();
      members_[m].adam->Minimize(loss);
      nll += loss_m / members(), mse += mse_m / members();
    }
    return {nll, mse};
  }

  // Predict next states and rewards with member m, sampling from its
  // Gaussian when `sample` is set, else using the mean.
  struct Prediction {
    Tensor next_s, r;  // [N, obs], [N]
  };
  Prediction Predict(int m, const Tensor& S, const Tensor& A, bool sample, std::mt19937_64& rng) const {
    namespace nn = datamunge::nn;
    nn::NoGradGuard no_grad;
    const int n = static_cast<int>(S.shape()[0]);
    const Output o = Forward(m, S, A);
    Tensor out = o.mean;
    if (sample) {
      std::vector<float> e(static_cast<size_t>(n) * (obs_dim_ + 1));
      std::normal_distribution<float> normal(0.f, 1.f);
      for (float& x : e) x = normal(rng);
      out = o.mean + nn::exp(0.5 * o.logvar) * nn::from_vector(e, {n, obs_dim_ + 1});
    }
    Prediction p;
    p.next_s = S + nn::slice_axis(out, 1, 0, obs_dim_);
    p.r = nn::squeeze(nn::slice_axis(out, 1, obs_dim_, obs_dim_ + 1), {1});
    return p;
  }

 private:
  struct Member {
    std::unique_ptr<datamunge::nn::ParameterStore> params;
    std::unique_ptr<Adam> adam;
  };
  int obs_dim_, hidden_;
  std::vector<Member> members_;
};

// Simple dataset of transitions with continuous actions.
struct TransitionSet {
  std::vector<Obs> s, s2;
  std::vector<std::vector<float>> a;
  std::vector<float> r;
  void Add(const Obs& st, const std::vector<float>& at, const Obs& st2, float rt) {
    s.push_back(st), a.push_back(at), s2.push_back(st2), r.push_back(rt);
  }
  int size() const { return static_cast<int>(r.size()); }
  // Random minibatch as tensors.
  void Sample(int n, std::mt19937_64& rng, datamunge::nn::Tensor* S, datamunge::nn::Tensor* A, datamunge::nn::Tensor* S2,
              datamunge::nn::Tensor* R) const {
    namespace nn = datamunge::nn;
    const int obs_dim = static_cast<int>(s[0].size()), act_dim = static_cast<int>(a[0].size());
    std::vector<float> vs(static_cast<size_t>(n) * obs_dim), vs2(vs.size()), va(static_cast<size_t>(n) * act_dim), vr(n);
    for (int i = 0; i < n; ++i) {
      const int k = static_cast<int>(rng() % r.size());
      for (int j = 0; j < obs_dim; ++j) vs[i * obs_dim + j] = s[k][j], vs2[i * obs_dim + j] = s2[k][j];
      for (int j = 0; j < act_dim; ++j) va[i * act_dim + j] = a[k][j];
      vr[i] = r[k];
    }
    *S = nn::from_vector(vs, {n, obs_dim});
    *S2 = nn::from_vector(vs2, {n, obs_dim});
    *A = nn::from_vector(va, {n, act_dim});
    *R = nn::from_vector(vr, {n});
  }
};

}  // namespace rl
