// datamunge::nn port: copied from tflow_examples src/common/rl/vec_env.h (TF-free code, unchanged).
// A batch of environments stepped together, for parallel rollouts.
//
// step() takes one action per environment and returns batched results.
// Environments that finish are reset automatically; the observation returned
// for them is the first observation of the new episode, and the last
// observation of the finished episode is kept in `final_obs`.
//
// With threads > 1 a persistent worker pool steps the environments in
// parallel. For the tiny environments here that only pays off when a step is
// expensive; the vec_env example measures both.
#pragma once

#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "common/rl/env.hpp"

namespace rl {

struct VecStep {
  std::vector<Obs> obs;
  std::vector<float> reward;
  std::vector<uint8_t> terminated, truncated;
  std::vector<Obs> final_obs;  // only set for environments that just ended
  bool done(int i) const { return terminated[i] || truncated[i]; }
};

class VecEnv {
 public:
  using Factory = std::function<std::unique_ptr<Env>()>;

  VecEnv(const Factory& factory, int n, uint64_t seed, int threads = 1) : threads_(threads) {
    for (int i = 0; i < n; ++i) {
      envs_.push_back(factory());
      envs_.back()->Seed(seed + i);
    }
    if (threads_ > 1) StartWorkers();
  }
  ~VecEnv() { StopWorkers(); }
  int size() const { return static_cast<int>(envs_.size()); }
  Env& env(int i) { return *envs_[i]; }
  int obs_size() const { return envs_[0]->obs_size(); }
  ActionSpace action_space() const { return envs_[0]->action_space(); }

  std::vector<Obs> reset() {
    std::vector<Obs> obs(size());
    ForEach([&](int i) { obs[i] = envs_[i]->reset(); });
    return obs;
  }

  VecStep step(const std::vector<Action>& actions) {
    VecStep out;
    const int n = size();
    out.obs.resize(n);
    out.reward.resize(n);
    out.terminated.resize(n);
    out.truncated.resize(n);
    out.final_obs.resize(n);
    ForEach([&](int i) {
      StepResult r = envs_[i]->step(actions[i]);
      out.reward[i] = r.reward;
      out.terminated[i] = r.terminated;
      out.truncated[i] = r.truncated;
      if (r.done()) {
        out.final_obs[i] = std::move(r.obs);
        out.obs[i] = envs_[i]->reset();
      } else {
        out.obs[i] = std::move(r.obs);
      }
    });
    return out;
  }

 private:
  // Run fn(i) for every environment, sequentially or on the worker pool.
  void ForEach(const std::function<void(int)>& fn) {
    if (threads_ <= 1) {
      for (int i = 0; i < size(); ++i) fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      task_ = &fn;
      next_index_ = 0;
      remaining_ = size();
      ++generation_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [&] { return remaining_ == 0; });
    task_ = nullptr;
  }

  void StartWorkers() {
    for (int t = 0; t < threads_; ++t)
      workers_.emplace_back([this] {
        uint64_t seen = 0;
        while (true) {
          std::unique_lock<std::mutex> lock(mu_);
          cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
          if (stop_) return;
          seen = generation_;
          while (next_index_ < size()) {
            const int i = next_index_++;
            const std::function<void(int)>* task = task_;
            lock.unlock();
            (*task)(i);
            lock.lock();
            if (--remaining_ == 0) done_cv_.notify_one();
          }
        }
      });
  }
  void StopWorkers() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
  }

  std::vector<std::unique_ptr<Env>> envs_;
  int threads_;
  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int)>* task_ = nullptr;
  int next_index_ = 0, remaining_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace rl

