#pragma once
/// @file nagle.hpp
/// @brief Checked event-driven packet coalescing with actual sender-state histories.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief One application write or acknowledgment in an ordinal sender simulation.
/// Events have no timestamps, network window, retransmission or delayed-ACK model.
/// @par Example
/// @code
/// dm::algorithms::NagleEvent e{dm::algorithms::NagleEvent::Write,100};std::print("{}\n",e.bytes); // 100
/// @endcode
/// @see nagle_simulate()
struct NagleEvent {
  /// @brief Supported input-event kinds; arbitrary cast values are rejected.
  /// @par Example
  /// @code
  /// auto type=dm::algorithms::NagleEvent::Ack;std::print("{}\n",int(type)); // 1
  /// @endcode
  /// @see NagleEvent
  enum Type {
    Write,       ///< Application supplies nonnegative bytes.
    Ack          ///< Peer acknowledges up to currently outstanding sent bytes.
  } type{Write}; ///< Actual event kind, default Write.
  int bytes{0};  ///< Nonnegative write/acknowledgment byte request.
};
/// @brief Actual final packet stream and widened sender accounting.
/// Conservation holds: written=sent+buffered and sent=unacked+acknowledged.
/// Oversized ACK requests saturate at outstanding bytes; no implicit final ACK
/// or flush is supplied. Zero-byte events remain events but emit no zero packet.
/// @par Example
/// @code
/// auto a=dm::algorithms::nagle_analyze(dm::algorithms::nagle_demo_events(),500);
/// std::print("{} {}\n",a.sent,a.buffered); // 400 200
/// @endcode
/// @see nagle_analyze()
struct NagleAnalysis {
  std::vector<int> packets;            ///< Actual emitted packet payload sizes, each1..mss.
  std::uint64_t    written{0};         ///< Actual cumulative application bytes.
  std::uint64_t    sent{0};            ///< Actual cumulative emitted payload bytes.
  std::uint64_t    buffered{0};        ///< Actual terminal unsent payload bytes.
  std::uint64_t    unacked{0};         ///< Actual terminal sent but unacknowledged bytes.
  std::uint64_t    acknowledged{0};    ///< Actual accepted ACK bytes, excluding excess requests.
  std::size_t      full_packets{0};    ///< Actually emitted payloads equal to mss.
  std::size_t      partial_packets{0}; ///< Actually emitted payloads strictly below mss.
};
/// @brief Actual packet emission and its immediate scheduling reason.
/// Reason0=full segment,1=idle partial write,2=ACK-released partial,3=baseline.
/// @par Example
/// @code
/// auto t=dm::algorithms::nagle_trace(dm::algorithms::nagle_demo_events(),500);
/// std::print("{}\n",t.emissions.front().bytes); // 100
/// @endcode
/// @see NagleTrace
struct NagleEmission {
  std::size_t event{0};  ///< Original event index that triggered this packet.
  int         bytes{0};  ///< Actual positive payload bytes.
  unsigned    reason{0}; ///< Actual emission route0..3.
};
/// @brief Actual buffer/outstanding states and packet span of one input event.
/// @par Example
/// @code
/// auto t=dm::algorithms::nagle_trace(dm::algorithms::nagle_demo_events(),500);
/// std::print("{}\n",t.steps.back().buffer_after); // 200
/// @endcode
/// @see NagleTrace
struct NagleStep {
  std::uint64_t buffer_before{0};  ///< Actual unsent bytes before the event.
  std::uint64_t unacked_before{0}; ///< Actual outstanding sent bytes before the event.
  std::uint64_t buffer_after{0};   ///< Actual unsent bytes after all triggered emissions.
  std::uint64_t unacked_after{0};  ///< Actual outstanding sent bytes after all emissions.
  std::uint64_t ack_applied{0};    ///< Actual acknowledged amount, zero on writes.
  std::size_t   first_packet{0};   ///< Actual first emitted packet index for this event.
  std::size_t   packet_count{0};   ///< Actual emitted packet count for this event.
};
/// @brief Owned original events, actual sender states/emissions and final accounting.
/// Capture units count mss/enabled scalars, original events, step records,
/// result packet sizes and emission records:2+2*events+2*packets.
/// @par Example
/// @code
/// auto t=dm::algorithms::nagle_trace(dm::algorithms::nagle_demo_events(),500);
/// std::print("{} {}\n",t.result.packets,t.captured_entries); // [100, 300] 20
/// @endcode
/// @see nagle_trace()
struct NagleTrace {
  int  mss{0};        ///< Original positive maximum segment payload size.
  bool enabled{true}; ///< True coalesces partial writes; false is the immediate-write baseline.
  std::vector<NagleEvent>    events;    ///< Owned complete original validated events.
  std::vector<NagleStep>     steps;     ///< Actual event-order sender states.
  std::vector<NagleEmission> emissions; ///< Actual packets, in result packet order.
  NagleAnalysis              result;    ///< Actual final sender accounting and packet stream.
  std::size_t                captured_entries{0}; ///< Actual admitted logical capture units.
};
/// @brief Simulate full-segment bypass and ACK-gated partial packet coalescing.
/// On writes, emit every available full segment, then emit a remainder only
/// if no bytes are outstanding. An ACK clamps outstanding bytes at zero and
/// releases a held remainder when it clears them. No implicit end flush.
/// Complete type/byte/mss validation precedes packetization. O(events+packets)
/// work, constant driver state plus owned packet output. Widened byte counters
/// reject overflow instead of legacy signed wrapping. No actual networking.
/// @param events Nonnegative write/ACK requests with valid event kinds.
/// @param mss Strictly positive maximum segment payload size.
/// @return Owned actual positive packet sizes.
/// @throws std::invalid_argument If mss, event kind or byte count is invalid.
/// @throws std::overflow_error If cumulative byte counters overflow uint64.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nagle_simulate(dm::algorithms::nagle_demo_events(),500)); // [100, 300]
/// @endcode
/// @par References
/// - J. Nagle (1984), Congestion Control in IP/TCP Internetworks, RFC896, https://www.rfc-editor.org/rfc/rfc896.html (ordinal sender simplification here).
/// @see nagle_analyze(), nagle_trace()
std::vector<int> nagle_simulate(const std::vector<NagleEvent>& events, int mss);
/// @brief Emit each write immediately, split into positive MSS-bounded payloads.
/// ACKs still update accounting in the common producer but cannot delay output.
/// Same complete validation, complexity and counter guards as nagle_simulate().
/// @param events Complete nonnegative valid write/ACK events.
/// @param mss Strictly positive maximum payload size.
/// @return Owned immediate-write baseline packet sizes.
/// @throws std::invalid_argument If any input event or mss is invalid.
/// @throws std::overflow_error If cumulative counters overflow uint64.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::no_nagle_simulate({{dm::algorithms::NagleEvent::Write,7}},5)); // [5, 2]
/// @endcode
/// @see nagle_simulate()
std::vector<int> no_nagle_simulate(const std::vector<NagleEvent>& events, int mss);
/// @brief Return actual terminal sender accounting from the ordinary producer.
/// No replay; same complexity and guards as nagle_simulate().
/// @param events Complete valid event stream, retained only in trace calls.
/// @param mss Strictly positive maximum payload size.
/// @param enabled True default enables coalescing; false uses immediate-write baseline.
/// @return Owned packets and actual final widened state.
/// @throws std::invalid_argument If any event or mss is invalid.
/// @throws std::overflow_error If cumulative counters overflow uint64.
/// @par Example
/// @code
/// auto a=dm::algorithms::nagle_analyze(dm::algorithms::nagle_demo_events(),500);
/// std::print("{}\n",a.unacked); // 300
/// @endcode
/// @see NagleAnalysis
NagleAnalysis nagle_analyze(const std::vector<NagleEvent>& events, int mss, bool enabled = true);
/// @brief Capture actual coalescing or baseline event/packet transitions.
/// Same producer and output as ordinary execution, without replay. Admit the
/// complete2+2*events base before copies and two units per actual emitted packet.
/// History quota failure returns no partial result. Zero disables optional cap.
/// @param events Complete validated event stream, copied into result.
/// @param mss Positive maximum payload size.
/// @param enabled True default coalesces; false captures immediate-write baseline.
/// @param max_entries Capture-unit cap, default1000000; zero disables.
/// @return Owned actual original events, states and packet decisions.
/// @throws std::invalid_argument If mss/event fields are invalid, before capture.
/// @throws std::overflow_error If byte counters overflow uint64.
/// @throws std::length_error If capture geometry or quota exceeds its limit.
/// @throws std::bad_alloc If allocation fails.
/// @par Example
/// @code
/// auto t=dm::algorithms::nagle_trace(dm::algorithms::nagle_demo_events(),500);
/// std::print("{}\n",t.steps.size()); // 7
/// @endcode
/// @see NagleTrace
NagleTrace nagle_trace(
    const std::vector<NagleEvent>& events,
    int                            mss,
    bool                           enabled     = true,
    std::size_t                    max_entries = 1000000);
/// @brief Generate the original six100-byte writes with one ACK after write four.
/// @param drain False default preserves original unsent tail; true appends ACK300 and ACK200 to drain/acknowledge it.
/// @return Owned seven original events, or nine with explicit terminal ACKs.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::nagle_demo_events().size()); // 7
/// @endcode
/// @see nagle_analyze()
std::vector<NagleEvent> nagle_demo_events(bool drain = false);
/// @brief Count assumed per-packet header bytes for the actually emitted packets only.
/// This excludes unsent buffered bytes and all protocol details beyond a fixed
/// supplied header size. O(1) work/storage, without modifying the analysis.
/// @param result Actual packetization result.
/// @param bytes_per_packet Assumed header bytes, default40, may be zero.
/// @return Checked packet count times assumed header size.
/// @throws std::overflow_error If product exceeds uint64.
/// @par Example
/// @code
/// auto a=dm::algorithms::nagle_analyze(dm::algorithms::nagle_demo_events(),500);
/// std::print("{}\n",dm::algorithms::nagle_header_bytes(a)); // 80
/// @endcode
/// @see NagleAnalysis
std::uint64_t nagle_header_bytes(const NagleAnalysis& result, std::uint64_t bytes_per_packet = 40);
} // namespace datamunge::algorithms
