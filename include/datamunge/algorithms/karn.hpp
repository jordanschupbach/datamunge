#pragma once
/// @file karn.hpp
/// @brief Checked educational RTT smoothing/Karn ambiguity rejection with actual event snapshots.
namespace datamunge::algorithms {
/// @brief Owned current RTT smoothing and timeout values in seconds.
/// Uninitialized state has SRTT/variation0 and the positive configured initial
/// RTO. A zero clean sample may produce zero RTO because this educational model
/// has no clock granularity or RFC minimum/maximum clamp. No caller references.
/// @par Example
/// @code
/// const auto s=dm::algorithms::KarnEstimator().state();std::print("{} {}\n",s.rto,s.initialized); // 1 false
/// @endcode
/// @see KarnEstimator
struct KarnState {
  double srtt{0};   ///< Actual smoothed RTT seconds, zero before initialization.
  double rttvar{0}; ///< Actual smoothed absolute deviation seconds, zero before initialization.
  double rto{1}; ///< Actual educational timeout seconds, possibly zero after a zero clean sample.
  bool   initialized{false}; ///< Whether a clean sample has initialized the smoother.
};
/// @brief Actual transition category for one accepted API event.
/// @par Example
/// @code
/// dm::algorithms::KarnEstimator k;std::print("{}\n",k.ack(.2).action==dm::algorithms::KarnAction::FirstSample); // true
/// @endcode
/// @see KarnEvent
enum class KarnAction {
  IgnoredAck,  ///< Retransmitted ACK ignored, even if its unused sample is nonfinite.
  FirstSample, ///< First clean RTT initializes mean and half-sample variation.
  CleanSample, ///< Later clean RTT performs original old-SRTT variation/mean updates.
  Timeout      ///< Actual RTO doubling without mean/variation updates.
};
/// @brief Owned actual pre/post estimator states from one producing transition.
/// Sample is caller-provided for ACK events, zero placeholder for Timeout. No
/// replay, retained input views or inference of network elapsed time.
/// @par Example
/// @code
/// dm::algorithms::KarnEstimator k;const auto e=k.timeout();std::print("{} {}\n",e.before.rto,e.after.rto); // 1 2
/// @endcode
/// @see KarnEstimator::ack(), KarnEstimator::timeout()
struct KarnEvent {
  KarnAction action{KarnAction::IgnoredAck}; ///< Actual branch or timeout operation.
  double     sample{0}; ///< Original RTT sample seconds, ignored for retransmitted ACKs.
  KarnState  before;    ///< Actual complete state before this event.
  KarnState  after;     ///< Actual complete committed state after this event.
};
/// @brief Educational Jacobson/Karels RTT smoother with Karn's ambiguity and backoff rules.
/// Clean first sample R sets SRTT=R,RTTVAR=R/2,RTO=3R. Subsequent clean samples
/// update RTTVAR=.75*oldvar+.25*abs(oldSRTT-R), then SRTT=.875*oldSRTT+.125*R,
/// RTO=SRTT+4*RTTVAR. Retransmitted ACKs do nothing; timeouts double current RTO.
/// No timestamp exception, clock granularity, RFC minimum/maximum, TCP sender,
/// packet sequence tracking or scheduling is modeled. Therefore not a complete
/// RFC6298 implementation. Preserves original ordinary-double operation order.
/// Local owned state, no allocation/shared state; mutation requires external
/// synchronization if one instance is used concurrently. Failed updates leave
/// all state unchanged; copied instances evolve independently.
/// @par Example
/// @code
/// dm::algorithms::KarnEstimator k;k.on_ack(.2,false);k.on_timeout();
/// std::print("{:.1f}\n",k.rto()); // 1.2
/// @endcode
/// @par References
/// - RFC6298 sections2-5, https://www.rfc-editor.org/rfc/rfc6298.html;
///   this teaching variant omits granularity and recommended RTO clamps.
/// @see KarnState, KarnEvent
class KarnEstimator {
 public:
  /// @brief Start an uninitialized owned estimator with a positive finite initial timeout.
  /// @param initial_rto Initial timeout seconds, default1.
  /// @throws std::invalid_argument On nonpositive or nonfinite initial_rto.
  /// @par Example
  /// @code
  /// std::print("{}\n",dm::algorithms::KarnEstimator(2).rto()); // 2
  /// @endcode
  explicit KarnEstimator(double initial_rto = 1);
  /// @brief Apply an ACK through the same producing transition as ack(), discarding its snapshot.
  /// Retransmitted samples are ignored before validation because no RTT is used.
  /// @param rtt_sample RTT seconds, finite nonnegative for clean samples.
  /// @param was_retransmitted Whether the ACK's RTT is ambiguous.
  /// @throws std::invalid_argument On a negative/nonfinite clean sample.
  /// @throws std::overflow_error On produced estimate overflow, before state commit.
  /// @par Example
  /// @code
  /// dm::algorithms::KarnEstimator k;k.on_ack(.2,false);k.on_ack(99,true);std::print("{:.1f}\n",k.srtt()); // 0.2
  /// @endcode
  void on_ack(double rtt_sample, bool was_retransmitted);
  /// @brief Apply timeout doubling through timeout(), discarding its actual snapshot.
  /// @throws std::overflow_error Before doubling would become nonfinite; state unchanged.
  /// @par Example
  /// @code
  /// dm::algorithms::KarnEstimator k;k.on_timeout();std::print("{}\n",k.rto()); // 2
  /// @endcode
  void on_timeout();
  /// @brief Apply one ACK and own its actual before/after state.
  /// Ignored ACK preserves every field and does not validate its unused sample.
  /// Clean validation/calculation is complete before committing any field.
  /// @param rtt_sample RTT seconds, finite nonnegative if clean.
  /// @param was_retransmitted True to ignore ambiguity, defaultfalse.
  /// @return Actual branch, input sample and owned pre/post state snapshots.
  /// @throws std::invalid_argument On invalid clean RTT.
  /// @throws std::overflow_error On produced mean/variation/timeout overflow before commit.
  /// @par Example
  /// @code
  /// dm::algorithms::KarnEstimator k;std::print("{:.1f}\n",k.ack(.2).after.rto); // 0.6
  /// @endcode
  /// @see on_ack(), timeout()
  KarnEvent ack(double rtt_sample, bool was_retransmitted = false);
  /// @brief Double the actual current timeout, retaining mean/variation/initialization.
  /// @return Owned actual pre/post timeout event with sample0 placeholder.
  /// @throws std::overflow_error On nonfinite doubling, with no partial mutation.
  /// @par Example
  /// @code
  /// dm::algorithms::KarnEstimator k;std::print("{}\n",k.timeout().after.rto); // 2
  /// @endcode
  /// @see ack(), on_timeout()
  KarnEvent timeout();
  /// @brief Read an owned snapshot of all actual current fields.
  /// @return Current scalar state by value, no references.
  KarnState state() const { return state_; }
  /// @brief Read the current educational timeout seconds.
  /// @return Current finite RTO, possibly zero after clean zero RTT.
  double rto() const { return state_.rto; }
  /// @brief Read the current smoothed RTT seconds.
  /// @return Actual SRTT, zero before the first clean sample.
  double srtt() const { return state_.srtt; }
  /// @brief Read the current absolute-deviation smoother seconds.
  /// @return Actual RTTVAR, zero before initialization.
  double rttvar() const { return state_.rttvar; }
  /// @brief Report whether a clean RTT has initialized the smoother.
  /// @return True after a clean sample, unaffected by timeout/ignored ACK.
  bool initialized() const { return state_.initialized; }

 private:
  KarnState
      state_; ///< Owned complete estimator state, atomically committed after validation/calculation.
};
} // namespace datamunge::algorithms
