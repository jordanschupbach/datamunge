#pragma once
/// @file cristian.hpp
/// @brief Checked ideal-rate Cristian delay intervals, controlled exchanges and probe selection.
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace datamunge::algorithms {
/// @brief One ideal-rate request/reply observation using one consistent time unit.
/// server_time is stamped at reply departure, after server_processing has
/// elapsed. RTT includes outbound delay, processing and return delay. The
/// common one-way lower bound applies to both network legs, not processing.
/// All quantities are finite; durations are nonnegative. The server is assumed
/// accurate and elapsed client durations calibrated to the same ideal rate.
/// @par Example
/// @code
/// const dm::algorithms::CristianProbe p{1000,40,5,0};
/// std::print("{}\n",dm::algorithms::cristian_analyze(p).estimate); // 1020
/// @endcode
/// @see cristian_analyze(), CristianAnalysis
struct CristianProbe {
  double server_time=0; ///< Server clock at reply departure; any finite epoch is allowed.
  double round_trip_time=0; ///< Nonnegative elapsed request-to-receipt duration, including processing.
  double min_one_way_latency=0; ///< Assumed common lower bound on each network leg.
  double server_processing=0; ///< Known nonnegative server residence duration, default zero.
};

/// @brief Owned nominal receipt-time estimate and admissible return-delay interval.
/// network_round_trip=RTT-processing; return delay lies in [minimum,network-minimum]
/// in the ideal model. The interval midpoint is minimax for absolute error.
/// All fields are double evaluations, not outward-rounded certificates. Large
/// epochs can collapse absolute endpoints; the separate delay fields retain
/// information that an absolute timestamp addition may round away. No clock
/// is read/set and no references or shared mutable state are retained.
/// @par Example
/// @code
/// const auto a=dm::algorithms::cristian_analyze({1000,40,5,0});
/// std::print("{} {} {}\n",a.lower_time,a.estimate,a.upper_time); // 1005 1020 1035
/// @endcode
/// @par References
/// - F. Cristian (1989), Probabilistic clock synchronization, Distributed Computing 3, 146-158. https://doi.org/10.1007/BF01784024
/// @see cristian_analyze(), CristianProbe
struct CristianAnalysis {
  CristianProbe probe; ///< Owned original observation and assumptions.
  double network_round_trip=0; ///< RTT minus known server processing.
  double lower_delay=0; ///< Minimum admissible return delay from the assumed lower bound.
  double estimate_delay=0; ///< Half the network-only round trip.
  double upper_delay=0; ///< Network-only round trip minus the common minimum delay.
  double error_bound=0; ///< Nominal half-width, estimate_delay minus minimum delay.
  double lower_time=0; ///< Rounded server_time plus lower_delay.
  double estimate=0; ///< Rounded server_time plus estimate_delay, at this probe's receipt.
  double upper_time=0; ///< Rounded server_time plus upper_delay.
};

/// @brief Owned controlled exchange separating actual path asymmetry from an uncertainty bound.
/// The generated RTT includes both network legs and known processing. The server
/// timestamp is at reply departure. model_error=(outbound-return)/2 is calculated
/// from durations, rather than subtracting rounded epoch timestamps. It describes
/// ideal midpoint error; fields are floating approximations, not certificates.
/// @par Example
/// @code
/// const auto x=dm::algorithms::cristian_exchange(1000,5,0,35,5);
/// std::print("{} {}\n",x.analysis.error_bound,x.model_error); // 15 -15
/// @endcode
/// @see cristian_exchange(), CristianAnalysis
struct CristianExchange {
  double outbound_delay=0; ///< Controlled request network delay.
  double return_delay=0; ///< Controlled reply network delay.
  CristianAnalysis analysis; ///< Owned interval generated from this exchange's RTT/processing.
  double receipt_time=0; ///< Rounded actual server_time plus return_delay.
  double model_error=0; ///< Ideal estimate minus actual receipt-time value, from path durations.
};

/// @brief Owned complete per-probe analyses and the first smallest stored error radius.
/// Empty input has has_probe=false and index zero as an unused placeholder.
/// Selection is by stored error_bound, not raw RTT when assumptions/processing
/// differ. The chosen estimate refers to that probe's own receipt instant; it
/// is not extrapolated to a later present and does not set an operating-system clock.
/// @par Example
/// @code
/// const auto s=dm::algorithms::cristian_select({{1000,40,5,0},{2000,20,5,0}});
/// std::print("{} {}\n",s.index,s.probes[s.index].error_bound); // 1 5
/// @endcode
/// @see cristian_select(), CristianAnalysis
struct CristianSelection {
  bool has_probe=false; ///< Whether a selected observation exists.
  std::size_t index=0; ///< First index attaining the smallest stored radius, or unused zero when empty.
  std::vector<CristianAnalysis> probes; ///< Complete owned analyses in original order.
};

/// @brief Estimate receipt time from a reply-departure timestamp and an uncorrected RTT.
/// Same server_time+RTT/2 arithmetic as the original helper for accepted inputs.
/// Assumes negligible server processing or treats it conservatively as part of
/// the outbound leg. O(1) time/storage; no clock mutation or shared state.
/// A symmetric network alone does not remove bias from nonzero processing.
/// @param server_time Finite accurate server timestamp at reply departure, in the chosen unit.
/// @param round_trip_time Finite nonnegative request-to-receipt duration in that same unit.
/// @return Rounded timestamp estimate at receipt; not an extrapolated present time.
/// @throws std::invalid_argument For a nonfinite timestamp/duration or negative RTT.
/// @throws std::overflow_error If a positive half-duration underflows or the estimate overflows.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cristian_estimate(1000,40)); // 1020
/// @endcode
/// @par References
/// - F. Cristian (1989), Probabilistic clock synchronization. https://doi.org/10.1007/BF01784024
/// @see cristian_analyze(), cristian_error_bound()
inline double cristian_estimate(double server_time,double round_trip_time) {
  if(!std::isfinite(server_time) || !std::isfinite(round_trip_time) || round_trip_time<0)
    throw std::invalid_argument("Cristian: invalid timestamp or RTT");
  const double half=round_trip_time/2;
  if(round_trip_time>0 && half==0) throw std::overflow_error("Cristian: half-duration underflow");
  const double result=server_time+half;
  if(!std::isfinite(result)) throw std::overflow_error("Cristian: timestamp overflow");
  return result;
}

/// @brief Return the ideal-rate midpoint uncertainty RTT/2 minus a common delay minimum.
/// This is a model interval radius, not a statistical confidence interval or a
/// floating-point enclosure. If known server processing is available, call
/// cristian_analyze() to remove it first. Contradictory RTT<2*minimum now throws
/// instead of silently returning a misleading zero. O(1) time/storage, no state.
/// @param round_trip_time Finite nonnegative elapsed duration in the chosen time unit.
/// @param min_one_way_latency Finite nonnegative common network lower bound, default zero.
/// @return Nonnegative rounded nominal half-width; zero only at an accepted boundary.
/// @throws std::invalid_argument For nonfinite/negative durations or an infeasible lower bound.
/// @throws std::overflow_error If a positive half-duration or radius collapses to zero in double storage.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::cristian_error_bound(40,5)); // 15
/// @endcode
/// @par References
/// - Cristian (1989), https://doi.org/10.1007/BF01784024
/// @see cristian_analyze(), cristian_estimate()
inline double cristian_error_bound(double round_trip_time,double min_one_way_latency=0) {
  if(!std::isfinite(round_trip_time) || round_trip_time<0 || !std::isfinite(min_one_way_latency) || min_one_way_latency<0)
    throw std::invalid_argument("Cristian: invalid delay parameters");
  // Near the feasibility boundary, RTT-minimum is exact by Sterbenz's lemma.
  // This comparison avoids 2*minimum overflow and wider-long-double assumptions.
  const double other=round_trip_time-min_one_way_latency;
  if(min_one_way_latency>other) throw std::invalid_argument("Cristian: infeasible minimum latency");
  const double half=round_trip_time/2;
  if(round_trip_time>0 && half==0) throw std::overflow_error("Cristian: half-duration underflow");
  const double radius=half-min_one_way_latency;
  if(min_one_way_latency<other && radius==0) throw std::overflow_error("Cristian: positive radius underflow");
  return radius;
}

/// @brief Analyze the admissible return delay after removing known server processing.
/// All durations share one unit. Requires RTT-processing>=2*minimum. Reply
/// departure is the timestamp point; request-arrival stamps need a different
/// processing correction. Ideal-rate/server-accuracy assumptions are inputs,
/// not measurements certified by this calculator. O(1) time/storage; owned result.
/// Absolute values are rounded independently; use relative delay fields when
/// epoch resolution is coarse. No outward rounding or drift/error propagation.
/// @param probe Finite timestamp and nonnegative RTT/minimum/known residence duration.
/// @return Owned observation, relative admissible interval and nominal absolute receipt values.
/// @throws std::invalid_argument For nonfinite/negative durations, processing>RTT or infeasible bounds.
/// @throws std::overflow_error For a positive half-duration/radius collapsing to zero or nonfinite absolute outputs.
/// @par Example
/// @code
/// const auto a=dm::algorithms::cristian_analyze({1000,40,5,10});
/// std::print("{} {}\n",a.estimate,a.error_bound); // 1015 10
/// @endcode
/// @par References
/// - Cristian (1989), https://doi.org/10.1007/BF01784024 (ideal-rate specialization with known residence).
/// @see CristianAnalysis, cristian_exchange(), cristian_select()
CristianAnalysis cristian_analyze(const CristianProbe& probe);

/// @brief Generate a controlled asymmetric request/reply exchange in the ideal timing model.
/// Both network legs must meet the supplied common lower bound. RTT is their
/// sum plus processing; the timestamp is at reply departure. Known processing
/// is removed by cristian_analyze(). If rounding hides all positive network
/// time behind processing or invalidates the generated delay interval, the
/// exchange is rejected. O(1), no I/O/clock mutation.
/// @param server_time Finite server timestamp at reply departure.
/// @param outbound_delay Finite nonnegative request network delay.
/// @param server_processing Finite nonnegative known server residence duration.
/// @param return_delay Finite nonnegative reply network delay.
/// @param min_one_way_latency Finite common network lower bound, default zero.
/// @return Owned analysis, rounded actual receipt time and duration-based ideal midpoint error.
/// @throws std::invalid_argument For invalid scalar domains or either leg below the assumed minimum.
/// @throws std::overflow_error If RTT/time outputs overflow, a nonzero half/error collapses, or rounding hides network time/invalidates the interval.
/// @par Example
/// @code
/// const auto x=dm::algorithms::cristian_exchange(1000,20,10,20,5);
/// std::print("{} {}\n",x.analysis.probe.round_trip_time,x.model_error); // 50 0
/// @endcode
/// @par References
/// - Ideal-rate specialization of Cristian (1989), https://doi.org/10.1007/BF01784024
/// @see CristianExchange, cristian_analyze()
CristianExchange cristian_exchange(double server_time,double outbound_delay,double server_processing,double return_delay,double min_one_way_latency=0);

/// @brief Select the first probe with the smallest stored uncertainty after complete validation.
/// All probes must use the same duration unit. Validates every probe including
/// all derived output arithmetic before the
/// optional count cap and allocation, then owns complete analyses in input
/// order. Two linear passes, O(n) time/storage; no partial selection/history.
/// A smaller radius can select a different probe than minimum raw RTT when
/// residence/minimum assumptions differ. Estimates remain at their own receipt
/// instants: no aging, drift compensation or clock-setting policy is supplied.
/// @param probes Read-only observations, including an allowed empty sequence.
/// @param max_probes Complete observation cap, default 100000; zero disables this optional cap.
/// @return Owned complete analyses, existence flag and first minimum-radius index.
/// @throws std::invalid_argument If any probe is invalid, including ones beyond the cap.
/// @throws std::overflow_error If any probe has unrepresentable derived output.
/// @throws std::length_error If the complete cap or container capacity is exceeded.
/// @throws std::bad_alloc If owned storage cannot be allocated.
/// @par Example
/// @code
/// const auto s=dm::algorithms::cristian_select({{1000,20,5,0},{2000,40,5,30}},2);
/// std::print("{} {}\n",s.index,s.probes[s.index].error_bound); // 1 0
/// @endcode
/// @par References
/// - Cristian (1989), https://doi.org/10.1007/BF01784024 (retry motivation; first stored-radius selection is this API policy).
/// @see CristianSelection, cristian_analyze()
CristianSelection cristian_select(const std::vector<CristianProbe>& probes,std::size_t max_probes=100'000);
} // namespace datamunge::algorithms
