/// @file karn.cpp
/// @brief Atomic checked actual ACK and timeout transitions for the educational Karn estimator.
#include <cmath>
#include <datamunge/algorithms/karn.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
KarnEstimator::KarnEstimator(double initial) {
  if (!std::isfinite(initial) || initial <= 0)
    throw std::invalid_argument("Karn positive finite initial RTO");
  state_.rto = initial;
}
KarnEvent KarnEstimator::ack(double sample, bool retransmitted) {
  KarnEvent e;
  e.sample = sample;
  e.before = state_;
  e.after  = state_;
  if (retransmitted)
    return e;
  if (!std::isfinite(sample) || sample < 0)
    throw std::invalid_argument("Karn clean sample");
  if (!state_.initialized) {
    e.action            = KarnAction::FirstSample;
    e.after.srtt        = sample;
    e.after.rttvar      = sample / 2.;
    e.after.initialized = true;
  } else {
    e.action       = KarnAction::CleanSample;
    e.after.rttvar = .75 * state_.rttvar + .25 * std::fabs(state_.srtt - sample);
    e.after.srtt   = .875 * state_.srtt + .125 * sample;
  }
  e.after.rto = e.after.srtt + 4. * e.after.rttvar;
  if (!std::isfinite(e.after.srtt) || !std::isfinite(e.after.rttvar) || !std::isfinite(e.after.rto))
    throw std::overflow_error("Karn estimate overflow");
  state_ = e.after;
  return e;
}
KarnEvent KarnEstimator::timeout() {
  KarnEvent e;
  e.action = KarnAction::Timeout;
  e.before = state_;
  e.after  = state_;
  e.after.rto *= 2.;
  if (!std::isfinite(e.after.rto))
    throw std::overflow_error("Karn timeout overflow");
  state_ = e.after;
  return e;
}
void KarnEstimator::on_ack(double sample, bool retransmitted) {
  (void)ack(sample, retransmitted);
}
void KarnEstimator::on_timeout() {
  (void)timeout();
}
} // namespace datamunge::algorithms
