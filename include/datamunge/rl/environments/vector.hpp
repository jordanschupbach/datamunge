// datamunge::nn port: copied from tflow_examples src/common/rl/vec_env.h (TF-free dynamics; checked
// native public inputs). A batch of environments stepped together, for parallel rollouts.
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
/// @file vector.hpp
/// @brief Native vector simulation components for explicit caller-owned rollouts.

#include <algorithm>
#include <condition_variable>
#include <datamunge/rl/environments/interface.hpp>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace datamunge::rl::environments {

/// @brief Own the VecStep simulation state or interface.
///
/// This value owns its described fields; it has no global state or hidden random generator.
/// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
/// require the owner to remain alive and expire when their storage changes.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::VecStep
/// obj;obj.obs={{0}};obj.reward={0};obj.terminated={0};obj.truncated={0};obj.final_obs={{}};
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
struct VecStep {
  std::vector<Obs>   obs;    ///< Owning next observation, or owning batch rows for vector stepping.
  std::vector<float> reward; ///< Immediate Float32 reward, or one reward per vector row.
  std::vector<uint8_t>
      terminated; ///< True for MDP terminality, or one zero/one flag per vector row.
  std::vector<uint8_t>
      truncated; ///< True for an external/time-limit end, distinct from terminality.
  std::vector<Obs>
      final_obs; ///< Owning last observations for just-ended vector rows; other entries are empty.

  /// @brief Return whether either terminal or time-limit truncation is true.
  /// @param i Environment index within size(), or source row index for this overload.
  /// @return True when terminated or truncated is set for the queried result row.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::out_of_range The checked cell, state or row index is outside its owning storage.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecStep
  /// obj;obj.obs={{0}};obj.reward={0};obj.terminated={0};obj.truncated={0};obj.final_obs={{}};bool
  /// finished=obj.done(0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool done(int i) const {
    if (i < 0 || size_t(i) >= terminated.size() || terminated.size() != truncated.size())
      throw std::out_of_range("vector result row");
    return terminated[i] || truncated[i];
  }
};

/// @brief Owned homogeneous environment batch with deterministic per-instance streams and
/// persistent worker threads.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::VecEnv obj([]{return
/// std::make_unique<e::CartPole>();},2,1);obj.reset();
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class VecEnv {
 public:
  /// @brief Callable returning a newly owned non-null homogeneous Env instance.
  /// @par Example
  /// @code
  /// dm::rl::environments::VecEnv::Factory factory=[]{return
  /// std::make_unique<dm::rl::environments::CartPole>();};
  /// @endcode
  using Factory = std::function<std::unique_ptr<Env>()>;

  /// @brief Construct VecEnv with independent owned simulation state.
  /// @param factory Nonempty callable producing a new non-null owned Env of matching
  /// observation/action geometry.
  /// @param n Positive discrete action or environment count; vector pools accept1..4096.
  /// @param seed UInt64 local generator seed; reset(seed) restarts this instance stream.
  /// @param threads Worker count in1..64, default1; capped at environment count.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  VecEnv(const Factory& factory, int n, uint64_t seed, int threads = 1) : threads_(threads) {
    if (!factory || n < 1 || n > 4096 || threads < 1 || threads > 64)
      throw std::invalid_argument("vector environment factory/count/threads");
    threads_ = std::min(threads_, n);
    for (int i = 0; i < n; ++i) {
      envs_.push_back(factory());
      if (!envs_.back())
        throw std::invalid_argument("null environment factory result");
      if (envs_.back()->obs_size() < 0)
        throw std::invalid_argument("negative observation width");
      if (i) {
        auto a = envs_[0]->action_space(), z = envs_.back()->action_space();
        if (envs_.back()->obs_size() != envs_[0]->obs_size() || a.kind != z.kind || a.n != z.n ||
            a.dim != z.dim || a.low != z.low || a.high != z.high)
          throw std::invalid_argument("heterogeneous vector environment geometry");
      }
      envs_.back()->Seed(seed + i);
    }
    if (threads_ > 1) {
      try {
        StartWorkers();
      } catch (...) {
        StopWorkers();
        throw;
      }
    }
  }
  /// @brief Release owned simulation resources; VecEnv first joins every worker.
  ~VecEnv() { StopWorkers(); }
  /// @brief Return the number of owned environments.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();auto count=obj.size();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int size() const { return static_cast<int>(envs_.size()); }
  /// @brief Borrow one checked mutable environment until VecEnv destruction.
  /// @param i Environment index within size(), or source row index for this overload.
  /// @return Mutable borrowed Env reference owned by this VecEnv.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::out_of_range The checked cell, state or row index is outside its owning storage.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();auto& child=obj.env(0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Env& env(int i) {
    if (i < 0 || i >= size())
      throw std::out_of_range("environment index");
    return *envs_[i];
  }
  /// @brief Return the flattened observation width in Float32 values.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();auto value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int obs_size() const { return envs_[0]->obs_size(); }
  /// @brief Return an owning copy of the discrete or continuous action space.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();auto value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ActionSpace action_space() const { return envs_[0]->action_space(); }

  /// @brief Reset the episode and return an owning observation; a seed argument first reseeds the
  /// local RNG.
  /// @return Owning observation, or batch of observations, from the new episode.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();auto observation=obj.reset();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<Obs> reset() {
    std::vector<Obs> obs(size());
    ForEach([&](int i) { obs[i] = envs_[i]->reset(); });
    return obs;
  }

  /// @brief Advance the mutable simulation once, returning observation, reward and distinct
  /// terminal/time-limit flags.
  /// @param actions Exactly size() actions, validated before dispatch.
  /// @return Owning StepResult; vector stepping returns one owning row per environment with
  /// separate final_obs on auto-reset.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::VecEnv obj([]{return
  /// std::make_unique<e::CartPole>();},2,1);obj.reset();auto
  /// result=obj.step(std::vector<e::Action>{e::Action(0),e::Action(1)});
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  VecStep step(const std::vector<Action>& actions) {
    if (actions.size() != envs_.size())
      throw std::invalid_argument("one action per environment required");
    for (int i = 0; i < size(); ++i)
      ValidateAction(envs_[i]->action_space(), actions[i]);
    VecStep   out;
    const int n = size();
    out.obs.resize(n);
    out.reward.resize(n);
    out.terminated.resize(n);
    out.truncated.resize(n);
    out.final_obs.resize(n);
    ForEach([&](int i) {
      StepResult r      = envs_[i]->step(actions[i]);
      out.reward[i]     = r.reward;
      out.terminated[i] = r.terminated;
      out.truncated[i]  = r.truncated;
      if (r.done()) {
        out.final_obs[i] = std::move(r.obs);
        out.obs[i]       = envs_[i]->reset();
      } else {
        out.obs[i] = std::move(r.obs);
      }
    });
    return out;
  }

 private:
  // Run fn(i) for every environment, sequentially or on the worker pool.
  /// @brief Run one callback per environment, preserving independent per-instance RNG streams.
  /// @param fn Caller callback on one environment index; exceptions propagate after in-flight work
  /// completes.
  void ForEach(const std::function<void(int)>& fn) {
    if (threads_ <= 1) {
      for (int i = 0; i < size(); ++i)
        fn(i);
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      error_      = nullptr;
      task_       = &fn;
      next_index_ = 0;
      remaining_  = size();
      ++generation_;
    }
    cv_.notify_all();
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [&] { return remaining_ == 0; });
    task_ = nullptr;
    if (error_)
      std::rethrow_exception(error_);
  }

  /// @brief Create the persistent worker pool.
  void StartWorkers() {
    for (int t = 0; t < threads_; ++t)
      workers_.emplace_back([this] {
        uint64_t seen = 0;
        while (true) {
          std::unique_lock<std::mutex> lock(mu_);
          cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
          if (stop_)
            return;
          seen = generation_;
          while (next_index_ < size()) {
            const int                       i    = next_index_++;
            const std::function<void(int)>* task = task_;
            lock.unlock();
            std::exception_ptr error;
            try {
              (*task)(i);
            } catch (...) {
              error = std::current_exception();
            }
            lock.lock();
            if (error && !error_)
              error_ = error;
            if (--remaining_ == 0)
              done_cv_.notify_one();
          }
        }
      });
  }
  /// @brief Request shutdown and join every started worker.
  void StopWorkers() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_)
      w.join();
  }

  std::vector<std::unique_ptr<Env>> envs_; ///< Exclusively owned homogeneous environment instances.
  int                      threads_; ///< Persistent worker count, capped at environment count.
  std::vector<std::thread> workers_; ///< Owned worker threads joined before destruction.
  std::mutex               mu_;      ///< Mutex protecting task dispatch and completion state.
  std::condition_variable  cv_;      ///< Worker dispatch condition variable.
  std::condition_variable  done_cv_; ///< Caller completion condition variable.
  const std::function<void(int)>* task_ =
      nullptr; ///< Borrowed callback valid only until the synchronous ForEach returns.
  int                next_index_ = 0;     ///< Next undispatched environment row, protected by mu_.
  int                remaining_  = 0;     ///< Unfinished row count, protected by mu_.
  uint64_t           generation_ = 0;     ///< Dispatch generation, protected by mu_.
  bool               stop_       = false; ///< Shutdown request, protected by mu_.
  std::exception_ptr error_; ///< First worker failure retained until synchronous propagation.
};

} // namespace datamunge::rl::environments
