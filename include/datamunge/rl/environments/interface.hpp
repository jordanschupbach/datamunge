// datamunge::nn port: copied from tflow_examples src/common/rl/env.h (TF-free dynamics; checked
// native public inputs). Environment interface shared by the reinforcement learning examples.
//
// Modelled on the Gymnasium API: reset() returns the first observation,
// step() returns the next observation, a reward, and two end-of-episode
// flags: `terminated` (the MDP reached a terminal state) and `truncated`
// (a time limit cut the episode short; bootstrapping is still valid).
#pragma once
/// @file interface.hpp
/// @brief Native interface simulation components for explicit caller-owned rollouts.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/// @brief Seeded simulation components with explicit actions and owning observations; callers own
/// every policy and rollout.
namespace datamunge::rl::environments {

/// @brief Owning flattened Float32 observation values; no tensor or model lifetime is borrowed.
/// @par Example
/// @code
/// dm::rl::environments::Obs observation{0.f,1.f};
/// @endcode
using Obs = std::vector<float>;

// Either a discrete action (an index) or a continuous one (a vector).
/// @brief Own the Action simulation state or interface.
///
/// This value owns its described fields; it has no global state or hidden random generator.
/// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
/// require the owner to remain alive and expire when their storage changes.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Action obj(0);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
struct Action {
  int discrete = -1; ///< Discrete action index; default-1 marks an unused discrete field.
  std::vector<float>
      continuous; ///< Owning continuous action coordinates; empty for a discrete action.

  /// @brief Construct an owning discrete or continuous action value.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Action obj;
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Action() = default;
  /// @brief Construct an owning discrete or continuous action value.
  /// @param a Discrete action index or owning action vector, according to this overload.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Action obj(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Action(int a) : discrete(a) {} // NOLINT: implicit by design
  /// @brief Construct an owning discrete or continuous action value.
  /// @param a Discrete action index or owning action vector, according to this overload.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Action obj(std::vector<float>{.2f});
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Action(std::vector<float> a) : continuous(std::move(a)) {} // NOLINT: implicit by design
  /// @brief Construct an owning discrete or continuous action value.
  /// @param a Discrete action index or owning action vector, according to this overload.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::Action obj({.2f});
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Action(std::initializer_list<float> a) : continuous(a) {} // NOLINT: implicit by design
};

/// @brief Own the ActionSpace simulation state or interface.
///
/// This value owns its described fields; it has no global state or hidden random generator.
/// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
/// require the owner to remain alive and expire when their storage changes.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;auto obj=e::ActionSpace::Discrete(2);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
struct ActionSpace {
  /// @brief Source action-direction or reward-family enumeration.
  /// @par Example
  /// @code
  /// auto choice=dm::rl::environments::ActionSpace::kDiscrete;
  /// @endcode
  enum Kind {
    kDiscrete,   ///< Discrete integer-indexed action space.
    kContinuous, ///< Continuous vector-valued action space.
  };
  Kind               kind = kDiscrete; ///< Discrete or continuous action/reward-family tag.
  int                n    = 0;         ///< Number of legal discrete action indices.
  int                dim  = 0;         ///< Number of continuous action coordinates.
  std::vector<float> low;              ///< Owning finite continuous lower bounds.
  std::vector<float> high;             ///< Owning finite continuous upper bounds.

  /// @brief Create a discrete action space with positive action count.
  /// @param n Positive discrete action or environment count; vector pools accept1..4096.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto space=e::ActionSpace::Discrete(2);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  static ActionSpace Discrete(int n) {
    if (n < 1)
      throw std::invalid_argument("positive discrete action count required");
    ActionSpace s;
    s.kind = kDiscrete;
    s.n    = n;
    return s;
  }
  /// @brief Create a continuous action space with finite increasing per-coordinate bounds.
  /// @param low Finite per-coordinate lower bounds, each strictly below its matching upper bound.
  /// @param high Matching finite per-coordinate upper bounds.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto space=e::ActionSpace::Continuous({-1.f},{1.f});
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  static ActionSpace Continuous(std::vector<float> low, std::vector<float> high) {
    if (low.empty() || low.size() != high.size())
      throw std::invalid_argument("aligned continuous action bounds required");
    for (size_t i = 0; i < low.size(); ++i)
      if (!std::isfinite(low[i]) || !std::isfinite(high[i]) || low[i] >= high[i])
        throw std::invalid_argument("finite increasing action bounds required");
    ActionSpace s;
    s.kind = kContinuous;
    s.dim  = static_cast<int>(low.size());
    s.low  = std::move(low);
    s.high = std::move(high);
    return s;
  }
  // A uniformly random action from this space.
  /// @brief Sample an action using the caller-owned generator; discrete sampling preserves source
  /// modulo selection.
  /// @param rng Caller-owned generator, advanced by sampling; no global RNG is used.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto obj=e::ActionSpace::Discrete(2);std::mt19937_64
  /// bank(1);auto action=obj.Sample(bank);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Action Sample(std::mt19937_64& rng) const {
    if (kind == kDiscrete) {
      if (n < 1)
        throw std::invalid_argument("positive action count required");
      return Action(static_cast<int>(rng() % n));
    }
    if (dim < 1 || low.size() != size_t(dim) || high.size() != low.size())
      throw std::invalid_argument("action space geometry");
    for (int i = 0; i < dim; ++i)
      if (!std::isfinite(low[i]) || !std::isfinite(high[i]) || low[i] >= high[i])
        throw std::invalid_argument("finite increasing action bounds required");
    std::vector<float> a(dim);
    for (int i = 0; i < dim; ++i)
      a[i] = std::uniform_real_distribution<float>(low[i], high[i])(rng);
    return Action(a);
  }
  /// @brief Format the action-space geometry and continuous bounds as an owning string.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Action-space bounds have incompatible geometry.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;auto obj=e::ActionSpace::Discrete(2);auto
  /// value=obj.ToString();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string ToString() const {
    if (kind == kDiscrete)
      return "Discrete(" + std::to_string(n) + ")";
    if (dim < 0 || low.size() != size_t(dim) || high.size() != low.size())
      throw std::invalid_argument("action space geometry");
    std::string s = "Box(" + std::to_string(dim) + ") [";
    for (int i = 0; i < dim; ++i)
      s += (i ? ", " : "") + std::to_string(low[i]) + ".." + std::to_string(high[i]);
    return s + "]";
  }
};

// Validate shape and finite action values before any dynamics or RNG mutation.
/// @brief Validate action kind, dimensionality and finite values before simulation mutation.
/// @param space Declared action-space geometry.
/// @param action Action in this instance space; board actions must appear in LegalActions().
/// Continuous physics clips finite torque/throttle to bounds.
///
/// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references expire
/// at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid explicit
/// bounds/actions throw std::invalid_argument or std::out_of_range before their checked access.
/// @throws std::invalid_argument Invalid checked bounds, geometry or action.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::ValidateAction(e::ActionSpace::Discrete(2),e::Action(1));
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, StepResult
inline void ValidateAction(const ActionSpace& space, const Action& action) {
  if (space.kind == ActionSpace::kDiscrete) {
    if (action.discrete < 0 || action.discrete >= space.n || !action.continuous.empty())
      throw std::invalid_argument("discrete action outside space");
  } else {
    if (action.continuous.size() != size_t(space.dim))
      throw std::invalid_argument("continuous action dimension");
    for (float v : action.continuous)
      if (!std::isfinite(v))
        throw std::invalid_argument("finite continuous action required");
  }
}
// Bound dynamic simulation allocation before conversion to container sizes.
/// @brief Check a positive bounded simulation allocation count before unsigned conversion.
/// @param count Positive allocation count.
/// @param maximum Positive upper allocation bound, default1000000.
/// @return Owning value or borrowed reference exactly as indicated by the return type;
/// observation/state units follow the owning class.
///
/// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references expire
/// at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid explicit
/// bounds/actions throw std::invalid_argument or std::out_of_range before their checked access.
/// @throws std::invalid_argument Invalid checked bounds, geometry or action.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;auto count=e::CheckedCount(8);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, StepResult
inline size_t CheckedCount(int count, int maximum = 1000000) {
  if (count < 1 || count > maximum)
    throw std::invalid_argument("simulation dimension outside bounds");
  return size_t(count);
}

/// @brief Own the StepResult simulation state or interface.
///
/// This value owns its described fields; it has no global state or hidden random generator.
/// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
/// require the owner to remain alive and expire when their storage changes.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::StepResult obj;
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
struct StepResult {
  Obs   obs;                ///< Owning next observation, or owning batch rows for vector stepping.
  float reward     = 0.f;   ///< Immediate Float32 reward, or one reward per vector row.
  bool  terminated = false; ///< True for MDP terminality, or one zero/one flag per vector row.
  bool  truncated  = false; ///< True for an external/time-limit end, distinct from terminality.

  /// @brief Return whether either terminal or time-limit truncation is true.
  /// @return True when terminated or truncated is set for the queried result row.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::StepResult obj;bool finished=obj.done();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  bool done() const { return terminated || truncated; }
};

/// @brief Own the Env simulation state or interface.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class Env {
 public:
  /// @brief Release owned simulation resources; VecEnv first joins every worker.
  virtual ~Env() = default;
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// value=obj.name();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::string name() const = 0;
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int obs_size() const = 0;
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual ActionSpace action_space() const = 0;
  // Start a new episode. The seeded overload reseeds the environment's RNG
  // first, which makes the whole episode reproducible. Environments implement
  // the virtual Reset(); the public reset() wrappers are non-virtual so that
  // overriding does not hide the seeded overload.
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// observation=obj.reset();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs reset() { return Reset(); }
  /// @brief Reset the episode and return an owning observation; a seed argument first reseeds the
  /// local RNG.
  /// @param seed UInt64 local generator seed; reset(seed) restarts this instance stream.
  /// @return Owning observation, or batch of observations, from the new episode.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// observation=obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs reset(uint64_t seed) {
    Seed(seed);
    return Reset();
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// result=obj.step(e::Action(0));
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual StepResult step(const Action& action) = 0;
  // Human-readable picture of the current state (optional).
  /// @brief Return an owning ASCII picture of the current simulation state.
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// value=obj.render();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::string render() const { return ""; }
  // A name per discrete action, or per dimension of a continuous one
  // (optional; the video overlay labels the moves with them).
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// value=obj.action_names();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::vector<std::string> action_names() const { return {}; }
  /// @brief Reseed this simulation local MT19937-64 generator without resetting its physical state.
  /// @param seed UInt64 local generator seed; reset(seed) restarts this instance stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env&
  /// obj=concrete;obj.reset(1);obj.Seed(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual void Seed(uint64_t seed) = 0;
  // 0 means no time limit.
  /// @brief Return the configured time limit; zero disables time-limit truncation.
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::Env& obj=concrete;obj.reset(1);auto
  /// value=obj.max_steps();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int max_steps() const { return 0; }

 protected:
  /// @brief Reset the simulation state and per-episode counter, retaining the current local RNG
  /// stream.
  /// @return Owning initial observation for environments; board Reset mutates in place.
  virtual Obs Reset() = 0;
};

// Common plumbing: an RNG, a step counter and time-limit truncation.
/// @brief Own the EnvBase simulation state or interface.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::CartPole concrete;e::EnvBase& obj=concrete;obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class EnvBase : public Env {
 public:
  /// @brief Construct EnvBase with independent owned simulation state.
  /// @param max_steps Nonnegative source time limit; zero means unlimited.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @throws std::invalid_argument Invalid checked bounds, geometry or action.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::EnvBase& obj=concrete;obj.reset(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  explicit EnvBase(int max_steps = 0) : max_steps_(max_steps) {
    if (max_steps < 0)
      throw std::invalid_argument("nonnegative time limit required");
  }
  /// @brief Reseed this simulation local MT19937-64 generator without resetting its physical state.
  /// @param seed UInt64 local generator seed; reset(seed) restarts this instance stream.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::EnvBase&
  /// obj=concrete;obj.reset(1);obj.Seed(1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  void Seed(uint64_t seed) override { rng_.seed(seed); }
  /// @brief Return the configured time limit; zero disables time-limit truncation.
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::EnvBase&
  /// obj=concrete;obj.reset(1);auto value=obj.max_steps();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int max_steps() const override { return max_steps_; }
  /// @brief Return calls counted by the source finite-time-limit bookkeeping.
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
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::EnvBase&
  /// obj=concrete;obj.reset(1);auto value=obj.steps();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int steps() const { return steps_; }
  /// @brief Borrow the mutable local MT19937-64 generator; external access advances the same
  /// rollout stream.
  /// @return Mutable borrowed generator reference, valid until this environment is destroyed.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::CartPole concrete;e::EnvBase&
  /// obj=concrete;obj.reset(1);auto value=obj.rng();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::mt19937_64& rng() { return rng_; }

 protected:
  // Call at the start of reset()/step() implementations.
  /// @brief Reset the per-episode step counter without reseeding.
  void BeginEpisode() { steps_ = 0; }
  // Returns true if the time limit is reached after this step.
  /// @brief Count one source time-limit check and report whether the configured finite limit is
  /// reached.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  bool CountStep() { return max_steps_ > 0 && ++steps_ >= max_steps_; }
  /// @brief Draw one Float32 uniform value from the local generator.
  /// @param lo Finite lower uniform/clamp endpoint.
  /// @param hi Finite upper uniform/clamp endpoint.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  float Uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng_); }
  /// @brief Draw one Float32 Gaussian value from the local generator.
  /// @param mean Finite Gaussian location in reward/observation units.
  /// @param stddev Finite nonnegative reward noise scale; zero is used for Bernoulli construction.
  /// @return Owning value or borrowed reference exactly as indicated by the return type;
  /// observation/state units follow the owning class.
  float Normal(float mean, float stddev) {
    if (stddev == 0) {
      const float z = std::normal_distribution<float>(0.f, 1.f)(rng_);
      return z * 0.f + mean;
    }
    return std::normal_distribution<float>(mean, stddev)(rng_);
  }

 private:
  std::mt19937_64 rng_{0}; ///< Owned per-instance MT19937-64 state, seeded0 initially.
  int max_steps_ = 0;      ///< Configured source time limit; zero disables time-limit counting.
  int steps_ = 0; ///< Source finite-limit checks since reset; terminal short-circuits and unlimited

  ///< simulations do not count.
};

// Outcome of taking action a in state s, for environments that expose their
// model (needed by dynamic programming).
/// @brief Own the Transition simulation state or interface.
///
/// This value owns its described fields; it has no global state or hidden random generator.
/// Returned owning values remain valid after the source object is destroyed. Borrowed buffers
/// require the owner to remain alive and expire when their storage changes.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Transition obj{1,0,0,false};
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
struct Transition {
  float prob;       ///< Transition probability mass.
  int   next_state; ///< Successor finite-state index.
  float reward;     ///< Immediate Float32 reward, or one reward per vector row.
  bool  terminal;   ///< True when this outcome reaches an MDP terminal state.
};

// A finite MDP: states are indices, observations are the index as a float.
/// @brief Own the TabularEnv simulation state or interface.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::GridWorld concrete;e::TabularEnv& obj=concrete;obj.reset(1);
/// @endcode
/// @par References
/// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/ ;
/// https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class TabularEnv : public EnvBase {
 public:
  using EnvBase::EnvBase;
  /// @brief Return the number of finite MDP state indices.
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
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::TabularEnv&
  /// obj=concrete;obj.reset(1);auto value=obj.num_states();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int num_states() const = 0;
  /// @brief Return the current flattened finite-state index or borrow the physical state as
  /// specified by its type.
  /// @return Current state value, or const borrowed physical-state array valid until mutation.
  ///
  /// Mutable operations advance only this instance or caller-supplied RNG. Borrowed references
  /// expire at owner destruction; storage-changing operations invalidate borrowed buffers. Invalid
  /// explicit bounds/actions throw std::invalid_argument or std::out_of_range before their checked
  /// access. VecEnv propagates worker exceptions after all in-flight tasks finish; some children
  /// may advance.
  /// @par Example
  /// @code
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::TabularEnv&
  /// obj=concrete;obj.reset(1);auto value=obj.state();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual int state() const = 0;
  // The full transition model p(s', r | s, a), for planning methods.
  /// @brief Enumerate owning probability/reward/successor records for one finite-MDP state and
  /// action.
  /// @param s Finite MDP state index in0..num_states()-1.
  /// @param a Discrete action index or owning action vector, according to this overload.
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
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::TabularEnv&
  /// obj=concrete;obj.reset(1);auto outcomes=obj.transitions(0,1);
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  virtual std::vector<Transition> transitions(int s, int a) const = 0;
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
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::TabularEnv&
  /// obj=concrete;obj.reset(1);auto value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int obs_size() const override { return 1; }
  /// @brief Return the current finite-state index as one Float32 observation coordinate.
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
  /// namespace e=dm::rl::environments;e::GridWorld concrete;e::TabularEnv&
  /// obj=concrete;obj.reset(1);auto value=obj.StateObs();
  /// @endcode
  /// @par References
  /// - Gymnasium environment API and classic-control source: https://gymnasium.farama.org/api/env/
  /// ; https://github.com/Farama-Foundation/Gymnasium/tree/main/gymnasium/envs/classic_control .
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs StateObs() const { return {static_cast<float>(state())}; }
};

} // namespace datamunge::rl::environments
