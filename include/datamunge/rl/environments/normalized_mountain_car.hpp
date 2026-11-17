#pragma once
/// @file normalized_mountain_car.hpp
/// @brief Source affine observation adapter for the native mountain-car dynamics.
#include <datamunge/rl/environments/classic_control.hpp>
namespace datamunge::rl::environments {

/// @brief Affine observation adapter for the source native mountain-car task.
///
/// Maps[position,velocity] to[(position+.3)/.9,velocity/.07] in Float32.
/// The owning inner MountainCar retains source force/gravity, goal, rewards and200-step limit.
/// Copies own independent physical state/RNG; this adapter changes observations, not dynamics or
/// reward.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// e::NormalizedMountainCar env;auto row=env.reset(1);auto step=env.step(1);
/// @endcode
/// @par References
/// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
/// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
/// @see MountainCar, Scale()
class NormalizedMountainCar : public Env {
 public:
  /// @brief Return the stable normalized-task name.
  ///
  /// @return NormalizedMountainCar.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;auto name=env.name();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  std::string name() const override { return "NormalizedMountainCar"; }

  /// @brief Return normalized observation width.
  ///
  /// @return Two Float32 coordinates.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;auto size=env.obs_size();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  int obs_size() const override { return 2; }

  /// @brief Copy the inner discrete throttle action space.
  ///
  /// @return Discrete action space with3 actions.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;auto space=env.action_space();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  ActionSpace action_space() const override { return inner_.action_space(); }

  /// @brief Copy source throttle action labels.
  ///
  /// @return Owning three-element label vector.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;auto labels=env.action_names();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  std::vector<std::string> action_names() const override { return inner_.action_names(); }

  /// @brief Reseed the owned inner task without resetting its physical state.
  ///
  /// @param seed New MT19937-64 seed.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;env.Seed(1);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  void Seed(uint64_t seed) override { inner_.Seed(seed); }

  /// @brief Read the inner source episode time limit.
  ///
  /// @return Source200-step limit.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;auto limit=env.max_steps();
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  int max_steps() const override { return inner_.max_steps(); }

  /// @brief Apply one native action and normalize the owning successor observation.
  ///
  /// Mutation, validation and RNG behavior follow MountainCar::step().
  ///
  /// @param a Discrete source action0,1,2.
  /// @return Owning step result; reward/terminal/truncation unchanged.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;env.reset(1);auto row=env.step(1);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  StepResult step(const Action& a) override {
    StepResult r = inner_.step(a);
    r.obs        = Scale(r.obs);
    return r;
  }

  /// @brief Transform one finite source observation with the exact Float32 affine map.
  ///
  /// Does not clip values outside the physical state box.
  ///
  /// @param o Two finite Float32 coordinates[position,velocity].
  /// @return Owning scaled two-coordinate vector; input unchanged.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// auto row=e::NormalizedMountainCar::Scale({-.3f,0});
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  static Obs Scale(const Obs& o) {
    if (o.size() != 2 || !std::isfinite(o[0]) || !std::isfinite(o[1]))
      throw std::invalid_argument("finite mountain-car observation width2 required");
    return {(o[0] + 0.3f) / 0.9f, o[1] / 0.07f};
  }

 protected:
  /// @brief Reset the inner physical state and return an owning normalized observation.
  ///
  /// Uses the existing inner RNG, which Env::reset(seed) can reseed first.
  ///
  /// @return Scaled source reset observation.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::NormalizedMountainCar env;auto row=env.reset(1);
  /// @endcode
  /// @par References
  /// - Sutton and Barto (2018), Chapter13, http://incompleteideas.net/book/the-book-2nd.html;
  /// Schulman et al. (2016), Generalized Advantage Estimation, https://arxiv.org/abs/1506.02438 .
  /// @see MountainCar
  Obs Reset() override { return Scale(inner_.reset()); }

 private:
  MountainCar inner_; ///< Owning source dynamics, episode state and local RNG.
};

} // namespace datamunge::rl::environments
