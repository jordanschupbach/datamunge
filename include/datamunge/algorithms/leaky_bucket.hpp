#pragma once
/// @file leaky_bucket.hpp
/// @brief Checked lazy leaky-bucket meter with actual event decisions; no queue or output pacing.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief One proposed admission in caller time units and amount units.
/// @par Example
/// @code
/// const dm::algorithms::LeakyBucketArrival a{0,1};
/// std::print("{}\n",a.amount); // 1
/// @endcode
/// @see leaky_bucket_trace()
struct LeakyBucketArrival {
  double time{0};   ///< Finite nonnegative nondecreasing timestamp in caller units.
  double amount{1}; ///< Finite nonnegative amount, default1; zero is valid.
};
/// @brief Actual producing meter event with old state, lazy leak and final admission state.
/// The operation has no queue and schedules no release time. Rejection still
/// commits the elapsed leak and timestamp. Negative/backward/nonfinite inputs
/// throw before mutation. Unused excess drainage is not banked below level0.
/// @par Example
/// @code
/// dm::algorithms::LeakyBucket m(2,1);
/// const auto e=m.add_detail(0,3);
/// std::print("{} {}\n",e.accepted,e.after); // false 0
/// @endcode
/// @see LeakyBucket::add_detail()
struct LeakyBucketEvent {
  double time{0};          ///< Actual caller timestamp.
  double amount{0};        ///< Actual proposed nonnegative amount.
  double previous_time{0}; ///< Actual prior committed timestamp.
  double before{0};        ///< Actual stored level before elapsed leak.
  double leaked_level{0};  ///< Actual finite level after leak, before admission.
  double after{0};         ///< Actual committed level after acceptance/rejection.
  bool   accepted{false};  ///< Whether the proposed amount fits at this event.
  bool   widened{false};   ///< Whether ordinary leakage overflow used widened fallback.
};
/// @brief Lazy leaky-bucket meter, algebraically dual to a token bucket with complementary storage.
/// Starts empty at time0 and therefore allows an immediate burst up to capacity.
/// This is an admission meter, not a shaping queue; accepted arrivals have no
/// delayed release schedule. Capacity must be positive finite, rate nonnegative
/// finite; timestamps finite, nonnegative and nondecreasing, amounts finite and
/// nonnegative. Same-time arrivals preserve caller order. O(1) per event, no
/// borrowed input/shared global state; concurrent mutation requires synchronization.
/// @par Example
/// @code
/// dm::algorithms::LeakyBucket m(2,1);
/// const bool first=m.add(0,2), second=m.add(1,1);
/// std::print("{} {}\n",first,second); // true true
/// @endcode
/// @par References
/// - RFC2697(1999), https://www.rfc-editor.org/rfc/rfc2697; token meters are a related contract, not this implementation's exact parameterization.
/// @see add_detail(), leaky_bucket_trace()
class LeakyBucket {
 public:
  /// @brief Construct an empty meter at time0.
  /// @param capacity Positive finite maximum stored admitted amount.
  /// @param leak_rate Nonnegative finite amount drained per time unit;0 permits no drain.
  /// @throws std::invalid_argument On nonfinite/nonpositive capacity or negative/nonfinite rate.
  /// @par Example
  /// @code
  /// dm::algorithms::LeakyBucket m(5,1);
  /// std::print("{}\n",m.stored_level()); // 0
  /// @endcode
  /// @see LeakyBucket
  LeakyBucket(double capacity, double leak_rate);
  /// @brief Apply lazy leakage and try to admit the amount, returning the actual decision.
  /// @param now Finite nonnegative nondecreasing timestamp.
  /// @param amount Finite nonnegative amount, default1.
  /// @return True if leaked_level+amount fits, false otherwise; rejection still advances leakage/time.
  /// @throws std::invalid_argument On invalid event, before mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::LeakyBucket m(2,1);
  /// std::print("{}\n",m.add(0,3)); // false
  /// @endcode
  /// @see add_detail()
  bool add(double now, double amount = 1);
  /// @brief Return the actual same-execution leak/admission event while mutating the meter.
  /// Ordinary finite double order is preserved. Nonfinite elapsed-drain arithmetic
  /// uses a widened fallback clamped to zero; level always remains finite in[0,C].
  /// @param now Finite nonnegative nondecreasing timestamp.
  /// @param amount Finite nonnegative amount, default1.
  /// @return Owned actual old/leaked/new state and decision; no replay.
  /// @throws std::invalid_argument On invalid event, before mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::LeakyBucket m(2,1);
  /// m.add(0,2);
  /// const auto e=m.add_detail(1,1);
  /// std::print("{} {}\n",e.leaked_level,e.after); // 1 2
  /// @endcode
  /// @see add(), level()
  LeakyBucketEvent add_detail(double now, double amount = 1);
  /// @brief Apply lazy leakage and return the committed level without proposing an arrival.
  /// @param now Finite nonnegative nondecreasing timestamp.
  /// @return Actual committed level after leakage; this call mutates time/level.
  /// @throws std::invalid_argument On invalid time, before mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::LeakyBucket m(2,1);
  /// m.add(0,2);
  /// std::print("{}\n",m.level(1)); // 1
  /// @endcode
  /// @see stored_level()
  double level(double now);
  /// @brief Return capacity without mutation.
  /// @return Positive finite constructor capacity.
  double capacity() const noexcept { return capacity_; }
  /// @brief Return leak rate without mutation.
  /// @return Nonnegative finite amount per caller time unit.
  double leak_rate() const noexcept { return leak_rate_; }
  /// @brief Return stored level without advancing time.
  /// @return Actual most recently committed level.
  double stored_level() const noexcept { return level_; }
  /// @brief Return last committed time without leaking.
  /// @return Actual most recent event/level time, initially0.
  double last_time() const noexcept { return last_; }

 private:
  /// @brief Compute the next leaked level without mutation after complete event validation.
  /// @param now Validated next timestamp.
  /// @param widened Mutable actual fallback flag, set only when ordinary arithmetic is nonfinite.
  /// @return Finite next level clamped at0.
  double leak_value(double now, bool& widened) const;
  double capacity_;  ///< Positive finite maximum level.
  double leak_rate_; ///< Nonnegative finite drain rate.
  double level_{0};  ///< Actual committed level, initially empty.
  double last_{0};   ///< Actual committed timestamp, initially0.
};
/// @brief Owned complete actual meter event sequence and selected real decisions.
/// One captured logical record per arrival owns its time, amount and actual
/// transition. Field scalars/capacities/bytes/RAM are excluded from the quota.
/// @par Example
/// @code
/// const auto t=dm::algorithms::leaky_bucket_trace(2,1,{{0,2},{0,1}});
/// std::print("{} {}\n",t.accepted,t.rejected); // 1 1
/// @endcode
/// @see leaky_bucket_trace()
struct LeakyBucketTrace {
  double                        capacity{0}; ///< Actual constructor capacity.
  double                        rate{0};     ///< Actual constructor leak rate.
  std::vector<LeakyBucketEvent> events; ///< Owned actual producing event records, caller order.
  std::size_t accepted{0};         ///< Actual accepted event count, including fitting zero amounts.
  std::size_t rejected{0};         ///< Actual rejected event count.
  std::size_t captured_entries{0}; ///< Actual complete event-record count.
};
/// @brief Run a complete validated arrival sequence once and retain every actual meter decision.
/// Complete constructor/events/time-order/quota validation precedes copies and
/// simulation. O(n) time/capture, no prefix replay, no input mutation. A burst
/// can be admitted immediately; no pacing/shaping queue is modeled.
/// @param capacity Positive finite maximum amount.
/// @param rate Nonnegative finite amount per time unit.
/// @param arrivals Borrowed complete nondecreasing finite nonnegative times/amounts.
/// @param max_entries Complete event-record cap, default1000000;0 disables optional cap.
/// @return Owned actual complete producing records and real decision counts.
/// @throws std::invalid_argument On complete constructor/event contract violations.
/// @throws std::length_error On capture quota/container limits.
/// @throws std::bad_alloc On owned capture allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::leaky_bucket_trace(2,1,{{0,2},{1,1}});
/// std::print("{}\n",t.events.back().after); // 2
/// @endcode
/// @par References
/// - RFC2697(1999), https://www.rfc-editor.org/rfc/rfc2697; related token-meter model, not a queued shaper.
/// @see LeakyBucket
LeakyBucketTrace leaky_bucket_trace(
    double                                 capacity,
    double                                 rate,
    const std::vector<LeakyBucketArrival>& arrivals,
    std::size_t                            max_entries = 1000000);
} // namespace datamunge::algorithms
