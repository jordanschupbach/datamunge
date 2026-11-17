#pragma once
/// @file corridor.hpp
/// @brief Seeded slipping-corridor teaching MDP.
#include <datamunge/rl/environments/interface.hpp>
namespace datamunge::rl::environments {
/// @brief Local slipping corridor with80-percent intended direction and separate
/// terminal/time-limit outcomes.
///
/// Instances own mutable state and a local RNG; concurrent access to one instance requires external
/// synchronization. Returned observations own their values. Reset before stepping and after done.
/// VecEnv owns its children; client calls remain serialized, while workers access distinct
/// children.
/// @par Example
/// @code
/// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);
/// @endcode
/// @par References
/// - Repository original local simulation protocol, preserved in
/// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
/// - Sutton and Barto (2018), Reinforcement Learning, second edition,
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see Env, VecEnv
class Corridor : public EnvBase {
 public:
  /// @brief Construct Corridor with independent owned simulation state.
  /// @param length Corridor cell count in3..1000000, default7.
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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  explicit Corridor(int length = 7, int max_steps = 20) : EnvBase(max_steps), length_(length) {
    if (length < 3 || length > 1000000)
      throw std::invalid_argument("corridor length3..1000000 required");
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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);auto value=obj.name();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string name() const override { return "Corridor"; }
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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);auto value=obj.obs_size();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  int obs_size() const override { return 1; }
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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);auto value=obj.action_space();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  ActionSpace action_space() const override { return ActionSpace::Discrete(2); }

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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);obj.Reset();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  Obs Reset() override {
    BeginEpisode(); // resets the step counter used for truncation
    pos_ = length_ / 2;
    return {static_cast<float>(pos_)};
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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);auto
  /// result=obj.step(e::Action(0));
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  StepResult step(const Action& action) override {
    ValidateAction(action_space(), action);
    if (pos_ <= 0 || pos_ >= length_ - 1)
      throw std::logic_error("reset corridor after termination");
    int dir = action.discrete == 1 ? 1 : -1;
    if (Uniform(0, 1) > 0.8f)
      dir = -dir; // slip
    pos_ += dir;
    StepResult r;
    r.obs = {static_cast<float>(pos_)};
    if (pos_ == length_ - 1) {
      r.reward     = 1.f;
      r.terminated = true;
    } else if (pos_ == 0) {
      r.reward     = -1.f;
      r.terminated = true;
    } else {
      r.truncated = CountStep(); // time limit: not a terminal state
    }
    return r;
  }
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
  /// namespace e=dm::rl::environments;e::Corridor obj;obj.reset(1);auto value=obj.render();
  /// @endcode
  /// @par References
  /// - Repository original local simulation protocol, preserved in
  /// docs/mepml/nn/6_deep_reinforcement_learning/6.1_environments/.
  /// - Sutton and Barto (2018), Reinforcement Learning, second edition,
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see Env, StepResult
  std::string render() const override {
    std::string s(length_, '.');
    s[0]           = 'x';
    s[length_ - 1] = 'G';
    s[pos_]        = 'A';
    return s;
  }

 private:
  int length_;  ///< Owned corridor cell count including both terminal endpoints.
  int pos_ = 0; ///< Current corridor or mountain-car position in the owning model units.
};

} // namespace datamunge::rl::environments
