/// @file lamport_clock.cpp
/// @brief Shared atomic checked logical-clock update and actual semantic event wrappers.
#include <algorithm>
#include <datamunge/algorithms/lamport_clock.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
LamportEvent LamportClock::step(LamportEventKind kind, std::int64_t message) {
  if (kind == LamportEventKind::Receive && message < 0)
    throw std::invalid_argument("Lamport negative message stamp");
  const auto base = kind == LamportEventKind::Receive ? std::max(time_, message) : time_;
  if (base == std::numeric_limits<std::int64_t>::max())
    throw std::overflow_error("Lamport timestamp overflow");
  LamportEvent out{kind, time_, message, base + 1};
  time_ = out.after;
  return out;
}
std::int64_t LamportClock::local_event() {
  return local_step().after;
}
std::int64_t LamportClock::send() {
  return send_step().after;
}
std::int64_t LamportClock::receive(std::int64_t message_time) {
  return receive_step(message_time).after;
}
LamportEvent LamportClock::local_step() {
  return step(LamportEventKind::Local, 0);
}
LamportEvent LamportClock::send_step() {
  return step(LamportEventKind::Send, 0);
}
LamportEvent LamportClock::receive_step(std::int64_t message_time) {
  return step(LamportEventKind::Receive, message_time);
}
} // namespace datamunge::algorithms
