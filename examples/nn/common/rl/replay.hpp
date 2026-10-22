// datamunge::nn port of tflow_examples src/common/rl/replay.h.
// Experience replay buffers: uniform, prioritised (sum tree), n-step
// accumulation, and fixed-length sequences for recurrent agents.
//
// The buffer logic (ring buffers, sum tree, sampling order from the
// std::mt19937_64 the caller passes) is unchanged, so a port draws the same
// indices as the TF version. Only the minibatches differ: they are
// nn::Tensor values instead of tensorflow::Tensor feeds, and actions are
// int64 (the index type nn::one_hot and nn::gather use) instead of int32.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

// One stored step of experience (the model-based rl::Transition in env.hpp is different).
struct Experience {
  Obs s;
  int a;
  float r;
  Obs s2;
  float done;  // 1 if s2 is terminal (no bootstrapping), else 0
  std::vector<float> act_c;  // continuous action, if any (then `a` is unused)
};

// A minibatch of transitions as tensors.
struct Batch {
  datamunge::nn::Tensor s, a, r, s2, done, weights;  // [n, obs] int64 [n] [n] [n, obs] [n] [n]
  datamunge::nn::Tensor act_c;  // [n, act_dim] when the experiences carry continuous actions
  std::vector<int> indices;     // positions in the buffer (for priority updates)
};

inline Batch MakeBatch(const std::vector<const Experience*>& ts, int obs_dim, const std::vector<float>& weights,
                       std::vector<int> indices) {
  namespace nn = datamunge::nn;
  const int n = static_cast<int>(ts.size());
  const int act_dim = n ? static_cast<int>(ts[0]->act_c.size()) : 0;
  std::vector<float> s(static_cast<std::size_t>(n) * obs_dim), s2(s.size()), r(n), done(n), act_c(static_cast<std::size_t>(n) * act_dim);
  std::vector<std::int64_t> a(n);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < act_dim; ++j) act_c[i * act_dim + j] = ts[i]->act_c[j];
    for (int j = 0; j < obs_dim; ++j) {
      s[i * obs_dim + j] = ts[i]->s[j];
      s2[i * obs_dim + j] = ts[i]->s2[j];
    }
    a[i] = ts[i]->a;
    r[i] = ts[i]->r;
    done[i] = ts[i]->done;
  }
  Batch b;
  b.s = nn::from_vector(s, {n, obs_dim});
  b.s2 = nn::from_vector(s2, {n, obs_dim});
  b.a = nn::from_vector(a, {n});
  b.r = nn::from_vector(r, {n});
  b.done = nn::from_vector(done, {n});
  b.weights = nn::from_vector(weights, {n});
  if (act_dim) b.act_c = nn::from_vector(act_c, {n, act_dim});
  b.indices = std::move(indices);
  return b;
}

// Uniform replay: a ring buffer sampled uniformly at random.
class ReplayBuffer {
 public:
  ReplayBuffer(int capacity, int obs_dim) : capacity_(capacity), obs_dim_(obs_dim) {}
  void Add(Experience t) {
    if (static_cast<int>(data_.size()) < capacity_) {
      data_.push_back(std::move(t));
    } else {
      data_[next_] = std::move(t);
    }
    next_ = (next_ + 1) % capacity_;
  }
  int size() const { return static_cast<int>(data_.size()); }
  Batch Sample(int n, std::mt19937_64& rng) const {
    std::vector<const Experience*> ts(n);
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
      idx[i] = static_cast<int>(rng() % data_.size());
      ts[i] = &data_[idx[i]];
    }
    return MakeBatch(ts, obs_dim_, std::vector<float>(n, 1.f), idx);
  }

 private:
  int capacity_, obs_dim_;
  std::vector<Experience> data_;
  int next_ = 0;
};

// Prioritised replay (Schaul et al. 2016): sampling probability proportional
// to priority^alpha, with importance weights (N p)^-beta normalised by their
// maximum. A sum tree makes sampling O(log N).
class PrioritizedReplay {
 public:
  PrioritizedReplay(int capacity, int obs_dim, float alpha = 0.6f)
      : capacity_(capacity), obs_dim_(obs_dim), alpha_(alpha), tree_(2 * capacity, 0.f) {}
  void Add(Experience t) {
    const int pos = next_;
    if (static_cast<int>(data_.size()) < capacity_) data_.push_back(std::move(t));
    else data_[pos] = std::move(t);
    SetPriority(pos, max_priority_);  // new samples are tried at least once
    next_ = (next_ + 1) % capacity_;
  }
  int size() const { return static_cast<int>(data_.size()); }
  float total() const { return tree_[1]; }
  float max_priority() const { return max_priority_; }

  Batch Sample(int n, float beta, std::mt19937_64& rng) const {
    std::vector<const Experience*> ts(n);
    std::vector<int> idx(n);
    std::vector<float> w(n);
    const float segment = total() / n;  // stratified sampling
    float max_w = 0;
    for (int i = 0; i < n; ++i) {
      const float u = segment * (i + std::uniform_real_distribution<float>(0, 1)(rng));
      idx[i] = Find(u);
      ts[i] = &data_[idx[i]];
      const float p = tree_[idx[i] + capacity_] / total();
      w[i] = std::pow(size() * p, -beta);
      max_w = std::max(max_w, w[i]);
    }
    for (float& x : w) x /= max_w;
    return MakeBatch(ts, obs_dim_, w, idx);
  }
  // A uniformly sampled batch (weights 1), for comparisons.
  Batch SampleUniform(int n, std::mt19937_64& rng) const {
    std::vector<const Experience*> ts(n);
    std::vector<int> idx(n);
    for (int i = 0; i < n; ++i) {
      idx[i] = static_cast<int>(rng() % data_.size());
      ts[i] = &data_[idx[i]];
    }
    return MakeBatch(ts, obs_dim_, std::vector<float>(n, 1.f), idx);
  }
  // New priorities from the absolute TD errors of the last batch.
  void UpdatePriorities(const std::vector<int>& indices, const std::vector<float>& td_abs) {
    for (size_t i = 0; i < indices.size(); ++i) {
      const float p = std::pow(td_abs[i] + 1e-6f, alpha_);
      max_priority_ = std::max(max_priority_, p);
      SetPriority(indices[i], p);
    }
  }

 private:
  void SetPriority(int pos, float p) {
    int i = pos + capacity_;
    tree_[i] = p;
    for (i /= 2; i >= 1; i /= 2) tree_[i] = tree_[2 * i] + tree_[2 * i + 1];
  }
  // Index of the leaf whose cumulative range contains u.
  int Find(float u) const {
    int i = 1;
    while (i < capacity_) {
      if (u <= tree_[2 * i]) i = 2 * i;
      else u -= tree_[2 * i], i = 2 * i + 1;
    }
    return std::min(i - capacity_, size() - 1);
  }
  int capacity_, obs_dim_;
  float alpha_;
  std::vector<float> tree_;  // 1-indexed binary tree; leaves at [capacity, 2 capacity)
  std::vector<Experience> data_;
  int next_ = 0;
  float max_priority_ = 1.f;
};

// Turns one-step transitions into n-step ones: reward r_t + g r_{t+1} + ... +
// g^{n-1} r_{t+n-1}, next state s_{t+n}. Push() returns the completed
// transition once n steps are available; Flush() drains the rest when an
// episode ends (shorter horizons, bootstrapping off the terminal state).
class NStepBuffer {
 public:
  NStepBuffer(int n, float gamma) : n_(n), gamma_(gamma) {}
  std::vector<Experience> Push(const Experience& t) {
    pending_.push_back(t);
    std::vector<Experience> out;
    if (t.done > 0) return Flush();
    if (static_cast<int>(pending_.size()) >= n_) {
      out.push_back(Combine(0));
      pending_.erase(pending_.begin());
    }
    return out;
  }
  std::vector<Experience> Flush() {
    std::vector<Experience> out;
    for (size_t i = 0; i < pending_.size(); ++i) out.push_back(Combine(i));
    pending_.clear();
    return out;
  }

 private:
  Experience Combine(size_t start) const {
    Experience t = pending_[start];
    float discount = gamma_;
    for (size_t i = start + 1; i < pending_.size() && i < start + n_; ++i) {
      t.r += discount * pending_[i].r;
      discount *= gamma_;
      t.s2 = pending_[i].s2;
      t.done = pending_[i].done;
    }
    return t;
  }
  int n_;
  float gamma_;
  std::vector<Experience> pending_;
};

// Fixed-length windows of episodes for recurrent Q-learning. Each window
// holds L + 1 observations (the last one only for bootstrapping), L actions,
// rewards and done flags, and a mask that is 0 past the episode's end.
struct SeqBatch {
  datamunge::nn::Tensor obs, a, r, done, mask;  // [B, L+1, obs], int64 [B, L], [B, L], [B, L], [B, L]
};
class SequenceReplay {
 public:
  SequenceReplay(int capacity, int length, int obs_dim) : capacity_(capacity), L_(length), obs_dim_(obs_dim) {}
  // obs has T + 1 entries, the others T.
  void AddEpisode(const std::vector<Obs>& obs, const std::vector<int>& a, const std::vector<float>& r,
                  const std::vector<float>& done) {
    const int T = static_cast<int>(a.size());
    for (int start = 0; start < T; start += std::max(1, L_ / 2)) {
      Window w;
      for (int t = 0; t <= L_; ++t) {
        const Obs& o = obs[std::min(start + t, T)];
        w.obs.insert(w.obs.end(), o.begin(), o.end());
      }
      for (int t = 0; t < L_; ++t) {
        const bool valid = start + t < T;
        w.a.push_back(valid ? a[start + t] : 0);
        w.r.push_back(valid ? r[start + t] : 0.f);
        w.done.push_back(valid ? done[start + t] : 1.f);
        w.mask.push_back(valid ? 1.f : 0.f);
      }
      if (static_cast<int>(windows_.size()) < capacity_) windows_.push_back(std::move(w));
      else windows_[next_] = std::move(w);
      next_ = (next_ + 1) % capacity_;
    }
  }
  int size() const { return static_cast<int>(windows_.size()); }
  SeqBatch Sample(int B, std::mt19937_64& rng) const {
    namespace nn = datamunge::nn;
    std::vector<float> obs, r, done, mask;
    std::vector<std::int64_t> a;
    for (int i = 0; i < B; ++i) {
      const Window& w = windows_[rng() % windows_.size()];
      obs.insert(obs.end(), w.obs.begin(), w.obs.end());
      a.insert(a.end(), w.a.begin(), w.a.end());
      r.insert(r.end(), w.r.begin(), w.r.end());
      done.insert(done.end(), w.done.begin(), w.done.end());
      mask.insert(mask.end(), w.mask.begin(), w.mask.end());
    }
    SeqBatch b;
    b.obs = nn::from_vector(obs, {B, L_ + 1, obs_dim_});
    b.a = nn::from_vector(a, {B, L_});
    b.r = nn::from_vector(r, {B, L_});
    b.done = nn::from_vector(done, {B, L_});
    b.mask = nn::from_vector(mask, {B, L_});
    return b;
  }

 private:
  struct Window {
    std::vector<float> obs;
    std::vector<std::int64_t> a;
    std::vector<float> r, done, mask;
  };
  int capacity_, L_, obs_dim_;
  std::vector<Window> windows_;
  int next_ = 0;
};

}  // namespace rl
