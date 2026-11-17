#pragma once
// Read-only numeric/data oracle frozen from the original common replay header.
#include <datamunge/nn/nn.hpp>
#include <random>
#include <vector>
namespace sequence_reference {
using Obs=std::vector<float>;
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


}
