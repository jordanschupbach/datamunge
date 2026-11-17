/// @file nagle.cpp
/// @brief Shared checked sender event producer with actual state capture.
#include <algorithm>
#include <datamunge/algorithms/nagle.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private complete validation and actual event producer.
namespace {
/// @brief Validate every event and MSS before simulation or capture.
/// @param events Complete application/ACK requests.
/// @param mss Positive maximum payload size.
/// @throws std::invalid_argument If any type/byte count or mss is invalid.
void validate(const std::vector<NagleEvent>& events, int mss) {
  if (mss <= 0)
    throw std::invalid_argument("Nagle: positive MSS required");
  for (auto e : events)
    if ((e.type != NagleEvent::Write && e.type != NagleEvent::Ack) || e.bytes < 0)
      throw std::invalid_argument("Nagle: invalid event");
}
/// @brief Checked addition of actual cumulative bytes.
/// @param value Counter to update.
/// @param amount Nonnegative actual byte increment.
/// @throws std::overflow_error If uint64 addition is unrepresentable.
void add(std::uint64_t& value, std::uint64_t amount) {
  if (amount > std::numeric_limits<std::uint64_t>::max() - value)
    throw std::overflow_error("Nagle: byte count overflow");
  value += amount;
}
/// @brief Shared ordinary/trace sender state transitions.
/// @param events Complete validated events.
/// @param mss Positive segment size.
/// @param enabled True coalesces; false sends writes immediately.
/// @param out Optional actual-state sink.
/// @param cap Optional quota, zero disables.
/// @return Actual packets and terminal accounting.
/// @throws std::overflow_error If byte counters overflow.
/// @throws std::length_error If actual packet capture admission exceeds quota.
NagleAnalysis run(
    const std::vector<NagleEvent>& events,
    int                            mss,
    bool                           enabled,
    NagleTrace*                    out,
    std::size_t                    cap) {
  NagleAnalysis a;
  for (std::size_t i = 0; i < events.size(); ++i) {
    const auto e = events[i];
    NagleStep  step;
    step.buffer_before  = a.buffered;
    step.unacked_before = a.unacked;
    step.first_packet   = a.packets.size();
    /// @brief Emit one actual payload and record its immediate scheduling cause.
    auto emit = [&](int bytes, unsigned reason) {
      if (out) {
        if (out->captured_entries > std::numeric_limits<std::size_t>::max() - 2 ||
            (cap && (out->captured_entries > cap || 2 > cap - out->captured_entries)))
          throw std::length_error("Nagle: packet capture quota exceeded");
        out->captured_entries += 2;
        out->emissions.push_back({i, bytes, reason});
      }
      add(a.sent, bytes);
      add(a.unacked, bytes);
      a.packets.push_back(bytes);
      if (bytes == mss)
        ++a.full_packets;
      else
        ++a.partial_packets;
    };
    if (e.type == NagleEvent::Write) {
      add(a.written, e.bytes);
      add(a.buffered, e.bytes);
      if (!enabled) {
        while (a.buffered) {
          auto bytes = static_cast<int>(std::min(a.buffered, std::uint64_t(mss)));
          emit(bytes, 3);
          a.buffered -= bytes;
        }
      } else {
        while (a.buffered >= std::uint64_t(mss)) {
          emit(mss, 0);
          a.buffered -= mss;
        }
        if (a.buffered && a.unacked == 0) {
          auto bytes = static_cast<int>(a.buffered);
          emit(bytes, 1);
          a.buffered = 0;
        }
      }
    } else {
      step.ack_applied = std::min(a.unacked, std::uint64_t(e.bytes));
      a.unacked -= step.ack_applied;
      add(a.acknowledged, step.ack_applied);
      if (enabled && a.unacked == 0 && a.buffered) {
        auto bytes = static_cast<int>(a.buffered);
        emit(bytes, 2);
        a.buffered = 0;
      }
    }
    step.buffer_after  = a.buffered;
    step.unacked_after = a.unacked;
    step.packet_count  = a.packets.size() - step.first_packet;
    if (out)
      out->steps.push_back(step);
  }
  return a;
}
} // namespace
std::vector<int> nagle_simulate(const std::vector<NagleEvent>& events, int mss) {
  validate(events, mss);
  return run(events, mss, true, nullptr, 0).packets;
}
std::vector<int> no_nagle_simulate(const std::vector<NagleEvent>& events, int mss) {
  validate(events, mss);
  return run(events, mss, false, nullptr, 0).packets;
}
NagleAnalysis nagle_analyze(const std::vector<NagleEvent>& events, int mss, bool enabled) {
  validate(events, mss);
  return run(events, mss, enabled, nullptr, 0);
}
NagleTrace
nagle_trace(const std::vector<NagleEvent>& events, int mss, bool enabled, std::size_t cap) {
  validate(events, mss);
  if (events.size() > (std::numeric_limits<std::size_t>::max() - 2) / 2)
    throw std::length_error("Nagle: capture geometry overflow");
  NagleTrace t;
  t.captured_entries = 2 + 2 * events.size();
  if (cap && t.captured_entries > cap)
    throw std::length_error("Nagle: capture quota exceeded");
  t.mss     = mss;
  t.enabled = enabled;
  t.events  = events;
  t.steps.reserve(events.size());
  t.result = run(events, mss, enabled, &t, cap);
  return t;
}
std::vector<NagleEvent> nagle_demo_events(bool drain) {
  std::vector<NagleEvent> e{
      {NagleEvent::Write, 100},
      {NagleEvent::Write, 100},
      {NagleEvent::Write, 100},
      {NagleEvent::Write, 100},
      {NagleEvent::Ack, 100},
      {NagleEvent::Write, 100},
      {NagleEvent::Write, 100}};
  if (drain) {
    e.push_back({NagleEvent::Ack, 300});
    e.push_back({NagleEvent::Ack, 200});
  }
  return e;
}
std::uint64_t nagle_header_bytes(const NagleAnalysis& a, std::uint64_t bytes) {
  if (bytes && a.packets.size() > std::numeric_limits<std::uint64_t>::max() / bytes)
    throw std::overflow_error("nagle_header_bytes: product overflow");
  return a.packets.size() * bytes;
}
} // namespace datamunge::algorithms
