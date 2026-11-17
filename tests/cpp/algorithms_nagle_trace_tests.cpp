#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <datamunge/algorithms/nagle.hpp>
#include <stdexcept>
using namespace datamunge::algorithms;
TEST(NagleTrace, ExhaustiveSmallEventStreamsActualConservationAndPacketSpans) {
  const std::vector<NagleEvent> alphabet{
      {NagleEvent::Write, 0},
      {NagleEvent::Write, 1},
      {NagleEvent::Write, 2},
      {NagleEvent::Write, 3},
      {NagleEvent::Ack, 0},
      {NagleEvent::Ack, 1},
      {NagleEvent::Ack, 3},
      {NagleEvent::Ack, 10}};
  for (int n = 0, total = 1; n <= 4; ++n, total *= 8)
    for (int code = 0; code < total; ++code)
      for (int mss = 1; mss <= 4; ++mss)
        for (bool enabled : {false, true}) {
          int                     q = code;
          std::vector<NagleEvent> events;
          for (int i = 0; i < n; ++i) {
            events.push_back(alphabet[q % 8]);
            q /= 8;
          }
          auto             t = nagle_trace(events, mss, enabled);
          std::vector<int> expected;
          std::uint64_t    buffer = 0, outstanding = 0, written = 0, acked = 0;
          for (std::size_t i = 0; i < events.size(); ++i) {
            auto  e = events[i];
            auto& s = t.steps[i];
            EXPECT_EQ(s.buffer_before, buffer);
            EXPECT_EQ(s.unacked_before, outstanding);
            EXPECT_EQ(s.first_packet, expected.size());
            auto first = expected.size();
            if (e.type == NagleEvent::Write) {
              written += e.bytes;
              buffer += e.bytes;
              while (buffer && (buffer >= unsigned(mss) || !enabled || outstanding == 0)) {
                int payload = buffer >= unsigned(mss) ? mss : int(buffer);
                expected.push_back(payload);
                buffer -= payload;
                outstanding += payload;
              }
            } else {
              auto actual = std::min(outstanding, std::uint64_t(e.bytes));
              EXPECT_EQ(s.ack_applied, actual);
              outstanding -= actual;
              acked += actual;
              if (enabled && !outstanding && buffer) {
                expected.push_back(int(buffer));
                outstanding += buffer;
                buffer = 0;
              }
            }
            EXPECT_EQ(s.buffer_after, buffer);
            EXPECT_EQ(s.unacked_after, outstanding);
            EXPECT_EQ(s.packet_count, expected.size() - first);
            for (std::size_t k = first; k < expected.size(); ++k) {
              EXPECT_EQ(t.emissions[k].event, i);
              EXPECT_EQ(t.emissions[k].bytes, expected[k]);
            }
          }
          EXPECT_EQ(t.result.packets, expected);
          EXPECT_EQ(t.result.written, written);
          EXPECT_EQ(t.result.buffered, buffer);
          EXPECT_EQ(t.result.unacked, outstanding);
          EXPECT_EQ(t.result.acknowledged, acked);
          EXPECT_EQ(t.result.sent, written - buffer);
          EXPECT_EQ(t.result.sent, outstanding + acked);
          EXPECT_EQ(t.captured_entries, 2 + 2 * events.size() + 2 * expected.size());
          EXPECT_EQ(nagle_analyze(events, mss, enabled).packets, expected);
        }
}
TEST(NagleTrace, OriginalUnsentTailExplicitDrainGuardsAndWideCounters) {
  auto a = nagle_analyze(nagle_demo_events(), 500);
  EXPECT_EQ(a.packets, (std::vector<int>{100, 300}));
  EXPECT_EQ(a.buffered, 200);
  EXPECT_EQ(a.unacked, 300);
  EXPECT_EQ(a.written, 600);
  EXPECT_EQ(nagle_header_bytes(a), 80);
  auto drained = nagle_analyze(nagle_demo_events(true), 500);
  EXPECT_EQ(drained.packets, (std::vector<int>{100, 300, 200}));
  EXPECT_EQ(drained.buffered, 0);
  EXPECT_EQ(drained.unacked, 0);
  EXPECT_EQ(drained.sent, 600);
  EXPECT_EQ(nagle_analyze(nagle_demo_events(true), 500, false).sent, 600);
  EXPECT_THROW(nagle_simulate({}, 0), std::invalid_argument);
  EXPECT_THROW(nagle_simulate({}, -1), std::invalid_argument);
  EXPECT_THROW(
      nagle_simulate({{NagleEvent::Write, 1}, {NagleEvent::Ack, -1}}, 1), std::invalid_argument);
  EXPECT_THROW(nagle_trace({{static_cast<NagleEvent::Type>(2), 0}}, 1), std::invalid_argument);
  auto t = nagle_trace(nagle_demo_events(), 500);
  EXPECT_NO_THROW(nagle_trace(nagle_demo_events(), 500, true, t.captured_entries));
  EXPECT_THROW(
      nagle_trace(nagle_demo_events(), 500, true, t.captured_entries - 1), std::length_error);
  auto wide = nagle_analyze({{NagleEvent::Write, INT_MAX}, {NagleEvent::Write, INT_MAX}}, INT_MAX);
  EXPECT_EQ(wide.unacked, 2ULL * INT_MAX);
  EXPECT_EQ(wide.packets.size(), 2);
  auto over = nagle_analyze({{NagleEvent::Write, 1}, {NagleEvent::Ack, INT_MAX}}, 10);
  EXPECT_EQ(over.acknowledged, 1);
  EXPECT_EQ(over.unacked, 0);
}
