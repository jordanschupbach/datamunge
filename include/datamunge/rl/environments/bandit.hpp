// datamunge::nn port: copied from tflow_examples src/common/rl/bandit.h (TF-free dynamics; checked
// native public inputs). Bandit environments: k-armed (stateless) and contextual.
//
// Both are one-step episodes: reset() gives the (possibly empty) context,
// step(arm) gives the reward and terminates.
#pragma once
/// @file bandit.hpp
/// @brief Native bandit simulation components for explicit caller-owned rollouts.

#include <algorithm>
#include <cmath>
#include <datamunge/rl/environments/interface.hpp>
#include <string>
#include <vector>

namespace datamunge::rl::environments {

/// @brief Return owning labels arm0 through arm(count-1).
/// @param arms Caller-supplied arms in the owning simulation geometry/units.
/// @return Owning value or borrowed reference exactly as indicated by the return type;
/// observation/state units follow the owning class.
///
/// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references expire
/// at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid explicit
/// bounds/actions throw std::invalid_argument or std::out_of_range before their checked access.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;auto names=e::ArmNames(3);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, StepResult
inline std::vector<std::string> ArmNames(int arms) {
  std::vector<std::string>
      names; ///< Owned source simulation state; units and geometry follow the owning class.
  for (int a = 0; a < arms; ++a)
    names.push_back("arm " + std::to_string(a));
  return names;
}

/// @brief Own the MultiArmedBandit simulation state or interface.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class MultiArmedBandit : public EnvBase {
 public:
  /// @brief Source action-direction or reward-family enumeration.
  /// @par Example
  /// @code
  /// auto choice=dm::rl::environments::MultiArmedBandit::kGaussian;
  /// @endcode
  enum Kind {
    kGaussian,  ///< Gaussian arm rewards.
    kBernoulli, ///< Bernoulli arm rewards.
  };

  // Gaussian arms: reward ~ N(mean[arm], stddev).
  /// @brief Construct MultiArmedBandit with independent owned simulation state.
  /// @param means Nonempty finite Gaussian arm means, at most1000000.
  /// @param stddev Finite nonnegative reward noise scale; zero is used for Bernoulli construction.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  MultiArmedBandit(std::vector<float> means, float stddev)
      : kind_(kGaussian), means_(std::move(means)), stddev_(stddev) {
    CheckedCount(int(means_.size()));
    if (!std::isfinite(stddev) || stddev < 0)
      throw std::invalid_argument("finite nonnegative reward deviation");
    for (float v : means_)
      if (!std::isfinite(v))
        throw std::invalid_argument("finite arm means required");
  }
  // Bernoulli arms: reward is 1 with probability p[arm], else 0.
  /// @brief Create one-step Bernoulli arms from probabilities in[0,1].
  /// @param probs Nonempty Bernoulli success probabilities in[0,1].
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
  /// namespace e=dm::rl::environments;auto bandit=e::MultiArmedBandit::Bernoulli({.2f,.8f});
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  static MultiArmedBandit Bernoulli(std::vector<float> probs) {
    MultiArmedBandit b(std::move(probs), 0.f);
    for (float v : b.means_)
      if (v < 0 || v > 1)
        throw std::invalid_argument("Bernoulli probabilities outside[0,1]");
    b.kind_ = kBernoulli;
    return b;
  }
  // Sutton & Barto's 10-armed testbed: means drawn from N(0, 1), unit noise.
  /// @brief Create Gaussian arm means from a separate seeded generator with unit reward noise.
  /// @param k Positive arm count, at most1000000.
  /// @param seed UInt64 local generator seed; reset(seed) restarts this instance stream.
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
  /// namespace e=dm::rl::environments;auto bandit=e::MultiArmedBandit::Testbed(3,1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  static MultiArmedBandit Testbed(int k, uint64_t seed) {
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> n(0.f, 1.f);
    std::vector<float>              means(CheckedCount(k));
    for (float& m : means)
      m = n(rng);
    return MultiArmedBandit(means, 1.f);
  }

  /// @brief Return the stable human-readable simulation name.
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.name();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override {
    return std::string(kind_ == kGaussian ? "Gaussian" : "Bernoulli") + "Bandit(" +
           std::to_string(num_arms()) + ")";
  }
  /// @brief Return the number of reward arms.
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.num_arms();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int num_arms() const { return static_cast<int>(means_.size()); }
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int obs_size() const override { return 0; }
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ActionSpace action_space() const override { return ActionSpace::Discrete(num_arms()); }
  /// @brief Return owning labels for discrete actions or continuous dimensions.
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.action_names();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<std::string> action_names() const override { return ArmNames(num_arms()); }
  /// @brief Return the true mean reward of one arm.
  /// @param arm Valid arm index in0..num_arms()-1.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// reward=obj.mean(0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  float mean(int arm) const {
    if (arm < 0 || arm >= num_arms())
      throw std::out_of_range("bandit arm");
    return means_[arm];
  }
  /// @brief Return the first arm attaining the largest true mean.
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.best_arm();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int best_arm() const {
    return static_cast<int>(std::max_element(means_.begin(), means_.end()) - means_.begin());
  }
  /// @brief Return the true mean of the best arm.
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// value=obj.best_mean();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  float best_mean() const { return means_[best_arm()]; }

  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  /// @return Owning initial observation for environments; board Reset mutates in place.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto
  /// obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);obj.Reset();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs Reset() override {
    BeginEpisode();
    return {};
  }
  /// @brief Advance the mutable simulation once, returning observation, reward and distinct
  /// terminal/time-limit flags.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
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
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// result=obj.step(e::Action(0));
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    StepResult r;
    r.reward     = Pull(action.discrete);
    r.terminated = true;
    return r;
  }
  /// @brief Draw one reward for a valid arm, advancing this environment local RNG.
  /// @param arm Valid arm index in0..num_arms()-1.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto obj=e::MultiArmedBandit::Testbed(3,1);obj.reset(1);auto
  /// reward=obj.Pull(0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  float Pull(int arm) {
    if (arm < 0 || arm >= num_arms())
      throw std::out_of_range("bandit arm");
    if (kind_ == kGaussian)
      return Normal(means_[arm], stddev_);
    return Uniform(0, 1) < means_[arm] ? 1.f : 0.f;
  }

 private:
  Kind               kind_;   ///< Gaussian versus Bernoulli reward family.
  std::vector<float> means_;  ///< Owned Gaussian means or Bernoulli success probabilities.
  float              stddev_; ///< Gaussian reward standard deviation.
};

// Contextual bandit: a context vector x ~ N(0, I) is drawn each round and
// arm a pays 1 with probability sigmoid(w_a . x + b_a). The weights are
// random but fixed per instance, so the optimal arm depends on the context.
/// @brief Own the ContextualBandit simulation state or interface.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class ContextualBandit : public EnvBase {
 public:
  /// @brief Construct ContextualBandit with independent owned simulation state.
  /// @param context_dim Positive context width, at most4096.
  /// @param num_arms Positive contextual arm count, at most4096.
  /// @param seed UInt64 local generator seed; reset(seed) restarts this instance stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ContextualBandit(int context_dim, int num_arms, uint64_t seed)
      : dim_(context_dim),
        arms_(num_arms),
        w_(CheckedCount(num_arms, 4096), std::vector<float>(CheckedCount(context_dim, 4096))),
        b_(CheckedCount(num_arms, 4096)) {
    std::mt19937_64                 rng(seed);
    std::normal_distribution<float> n(0.f, 1.f);
    for (auto& row : w_)
      for (float& v : row)
        v = n(rng);
    for (float& v : b_)
      v = 0.5f * n(rng);
  }
  /// @brief Return the stable human-readable simulation name.
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.name();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "ContextualBandit"; }
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int obs_size() const override { return dim_; }
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ActionSpace action_space() const override { return ActionSpace::Discrete(arms_); }
  /// @brief Return owning labels for discrete actions or continuous dimensions.
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.action_names();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::vector<std::string> action_names() const override { return ArmNames(arms_); }
  /// @brief Return the number of reward arms.
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.num_arms();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int num_arms() const { return arms_; }

  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  /// @return Owning initial observation for environments; board Reset mutates in place.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);obj.Reset();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs Reset() override {
    BeginEpisode();
    context_.resize(dim_);
    for (float& v : context_)
      v = Normal(0, 1);
    return context_;
  }
  /// @brief Borrow the current Float32 context until the next reset or object mutation.
  /// @return Const borrowed current-context vector, valid until mutation.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.context();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  const Obs& context() const { return context_; }
  // P(reward = 1 | context, arm): the quantity a learner is trying to find.
  /// @brief Evaluate sigmoid(w_arm dot context + bias_arm) using the current reset context.
  /// @param arm Valid arm index in0..num_arms()-1.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @throws std::out_of_range Checked cell, arm, action or environment index is out of range.
  /// @throws std::logic_error Required reset/captured-frame state is absent.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// reward=obj.ExpectedReward(0);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  float ExpectedReward(int arm) const {
    if (arm < 0 || arm >= arms_)
      throw std::out_of_range("contextual arm");
    if (context_.size() != size_t(dim_))
      throw std::logic_error("reset contextual bandit before reward query");
    float z = b_[arm];
    for (int i = 0; i < dim_; ++i)
      z += w_[arm][i] * context_[i];
    return 1.f / (1.f + std::exp(-z));
  }
  /// @brief Return the first arm maximizing expected reward for the current context.
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// value=obj.BestArm();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int BestArm() const {
    int best = 0;
    for (int a = 1; a < arms_; ++a)
      if (ExpectedReward(a) > ExpectedReward(best))
        best = a;
    return best;
  }
  /// @brief Advance the mutable simulation once, returning observation, reward and distinct
  /// terminal/time-limit flags.
  /// @param action Action in this instance space; board actions must appear in LegalActions().
  /// Continuous physics clips finite torque/throttle to bounds.
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
  /// namespace e=dm::rl::environments;e::ContextualBandit obj(2,3,1);obj.reset(1);auto
  /// result=obj.step(e::Action(0));
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    StepResult r;
    r.reward     = Uniform(0, 1) < ExpectedReward(action.discrete) ? 1.f : 0.f;
    r.terminated = true;
    return r;
  }

 private:
  int                             dim_;  ///< Context feature count.
  int                             arms_; ///< Contextual arm count.
  std::vector<std::vector<float>> w_;    ///< Owned snake/filmstrip board width or contextual weight

  ///< matrix, according to its declared type.
  std::vector<float> b_;       ///< Owned fixed arm logistic biases.
  Obs                context_; ///< Owned current context; empty until reset.
};

} // namespace datamunge::rl::environments
