#pragma once

/// @file enas_controller.hpp
/// @brief Two-decision autoregressive policy for architecture search.

#include <array>
#include <cmath>
#include <stdexcept>

namespace datamunge::nn {

/// @brief Bernoulli architecture controller and moving reward baseline.
/// The second decision has a separate logit for each possible first choice.
/// @par Example
/// @code
/// dm::nn::EnasController controller;
/// @endcode
struct EnasController {
  std::array<double,3> logits{}; ///< First logit followed by second-decision logits.
  double baseline=0; ///< Exponential moving reward average.
};

/// @brief Convert a controller logit into the probability of selecting operation one.
/// @param logit One Bernoulli logit.
/// @return Probability in [0,1].
/// @par Example
/// @code
/// auto p=dm::nn::enas_probability(0); // 0.5
/// @endcode
/// @see enas_policy_update()
inline double enas_probability(double logit) {
  return 1/(1+std::exp(-logit));
}

/// @brief Apply one REINFORCE update to the selected autoregressive path.
/// @param state Mutable logits and moving baseline.
/// @param choice Two binary operation indices.
/// @param reward Finite validation reward, with larger values preferred.
/// @param rate Finite nonnegative learning rate.
/// @param decay Baseline decay in [0,1].
/// @throws std::invalid_argument For invalid choices, reward, rate or decay.
/// @par Example
/// @code
/// dm::nn::EnasController state;
/// dm::nn::enas_policy_update(state,{0,1},-.3,.8,.9);
/// @endcode
/// @par References
/// - Pham et al. (2018), Efficient Neural Architecture Search via Parameters Sharing.
///   https://proceedings.mlr.press/v80/pham18a.html
/// @see enas_probability()
inline void enas_policy_update(EnasController& state,std::array<int,2> choice,
                               double reward,double rate,double decay) {
  if (choice[0]<0 || choice[0]>1 || choice[1]<0 || choice[1]>1 ||
      !std::isfinite(reward) || !std::isfinite(rate) || rate<0 || !(decay>=0 && decay<=1))
    throw std::invalid_argument("Invalid ENAS policy update");
  const auto scale=rate*(reward-state.baseline);
  const auto branch=1+choice[0];
  state.logits[0]+=scale*(choice[0]-enas_probability(state.logits[0]));
  state.logits[branch]+=scale*(choice[1]-enas_probability(state.logits[branch]));
  state.baseline=decay*state.baseline+(1-decay)*reward;
}

}  // namespace datamunge::nn
