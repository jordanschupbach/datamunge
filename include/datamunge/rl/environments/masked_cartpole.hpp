#pragma once
/// @file masked_cartpole.hpp
/// @brief Source partial-observation CartPole adapter retaining position and angle only.
#include <datamunge/rl/environments/classic_control.hpp>
namespace datamunge::rl::environments {

/// @brief Expose only position and pole angle from the source native CartPole.
///
/// The owned inner dynamics, rewards,500-step limit, terminality and RNG are unchanged.
/// Only observation coordinates0 and2 are exposed, so velocities must be inferred from history.
/// Copies own independent dynamics/RNG. This is a partial-observation adapter, not a new physics
/// model.
///
/// @par Example
/// @code
/// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
/// e::MaskedCartPole env;auto observation=env.reset(1);auto step=env.step(1);
/// @endcode
/// @par References
/// - Kapturowski et al. (2019), Recurrent Experience Replay,
/// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
/// http://incompleteideas.net/book/the-book-2nd.html .
/// @see CartPole, inner()
class MaskedCartPole : public Env {
 public:
  /// @brief Return the stable partial-observation task name.
  ///
  /// @return MaskedCartPole.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto name=env.name();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  std::string name() const override { return "MaskedCartPole"; }

  /// @brief Return the partial-observation width.
  ///
  /// @return Two Float32 coordinates[position,angle].
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto width=env.obs_size();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  int obs_size() const override { return 2; }

  /// @brief Borrow the read-only inner source dynamics object.
  ///
  /// @return Const reference valid until adapter destruction; reflects later adapter steps.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto& physics=env.inner();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  const CartPole& inner() const { return inner_; }

  /// @brief Copy the inner discrete force action space.
  ///
  /// @return Discrete action space with2 choices.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto space=env.action_space();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  ActionSpace action_space() const override { return inner_.action_space(); }

  /// @brief Copy the inner source force action names.
  ///
  /// @return Owning left/right label vector.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto names=env.action_names();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  std::vector<std::string> action_names() const override { return inner_.action_names(); }

  /// @brief Reseed the inner RNG without resetting physical state.
  ///
  /// @param seed New MT19937-64 seed.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;env.Seed(1);
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  void Seed(uint64_t seed) override { inner_.Seed(seed); }

  /// @brief Read the inner source episode time limit.
  ///
  /// @return Source500-step limit.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto limit=env.max_steps();
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  int max_steps() const override { return inner_.max_steps(); }

  /// @brief Apply one native force and retain only position/angle in the successor observation.
  ///
  /// @param a Source discrete action0 or1.
  /// @return Owning step result with reward/terminal/truncation unchanged.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;env.reset(1);auto result=env.step(1);
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  StepResult step(const Action& a) override {
    StepResult r = inner_.step(a);
    r.obs        = Mask(r.obs);
    return r;
  }

 protected:
  /// @brief Reset inner physical state and return the masked owning observation.
  ///
  /// @return Position/angle from source reset, retaining the current seeded stream.
  /// @par Example
  /// @code
  /// namespace n=dm::nn;namespace f=dm::rl::foundations;namespace e=dm::rl::environments;
  /// e::MaskedCartPole env;auto observation=env.reset(1);
  /// @endcode
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  Obs Reset() override { return Mask(inner_.reset()); }

 private:
  /// @brief Copy position and angle from one full CartPole observation.
  ///
  /// @param o Four-coordinate source observation.
  /// @return Owning Float32 vector{o[0],o[2]}.
  /// @throws std::invalid_argument Nonfinite values, incompatible geometry or explicit bounds are
  /// invalid.
  /// @par References
  /// - Kapturowski et al. (2019), Recurrent Experience Replay,
  /// https://openreview.net/forum?id=r1lyTjAqYX; Sutton and Barto (2018),
  /// http://incompleteideas.net/book/the-book-2nd.html .
  /// @see CartPole
  static Obs Mask(const Obs& o) {
    if (o.size() != 4)
      throw std::invalid_argument("CartPole observation width4 required");
    return {o[0], o[2]};
  }
  CartPole inner_; ///< Owning source physics, episode counter and RNG.
};

} // namespace datamunge::rl::environments
