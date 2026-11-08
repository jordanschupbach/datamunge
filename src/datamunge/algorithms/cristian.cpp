#include <datamunge/algorithms/cristian.hpp>
/// @file cristian.cpp
/// @brief Ideal-rate Cristian observation validation, controlled exchanges and owned selection.
#include <limits>

namespace datamunge::algorithms {
/// @brief Private finite-conversion guard for controlled duration/time sums.
namespace {
/// @brief Convert a widened scalar to finite double storage.
/// @param value Arithmetic result, possibly outside the double range.
/// @return Rounded finite double, including zero.
/// @throws std::overflow_error If finite double storage cannot represent the result.
double cristian_finite(long double value) {
  if(!std::isfinite(value) || std::fabs(value)>std::numeric_limits<double>::max())
    throw std::overflow_error("Cristian: unrepresentable duration or timestamp");
  return static_cast<double>(value);
}
} // namespace
CristianAnalysis cristian_analyze(const CristianProbe& probe) {
  if(!std::isfinite(probe.server_time) || !std::isfinite(probe.round_trip_time) || probe.round_trip_time<0 ||
     !std::isfinite(probe.min_one_way_latency) || probe.min_one_way_latency<0 ||
     !std::isfinite(probe.server_processing) || probe.server_processing<0 || probe.server_processing>probe.round_trip_time)
    throw std::invalid_argument("Cristian: invalid observation");
  CristianAnalysis result;result.probe=probe;
  result.network_round_trip=probe.round_trip_time-probe.server_processing;
  result.error_bound=cristian_error_bound(result.network_round_trip,probe.min_one_way_latency);
  result.lower_delay=probe.min_one_way_latency;
  result.estimate_delay=result.network_round_trip/2;
  result.upper_delay=result.network_round_trip-probe.min_one_way_latency;
  result.estimate=cristian_estimate(probe.server_time,result.network_round_trip);
  // Check range in widened arithmetic, then use the same double addition as
  // the midpoint. Casting a long-double sum can double-round a zero-width
  // endpoint away from the correctly rounded legacy midpoint.
  (void)cristian_finite(static_cast<long double>(probe.server_time)+result.lower_delay);
  (void)cristian_finite(static_cast<long double>(probe.server_time)+result.upper_delay);
  result.lower_time=cristian_finite(probe.server_time+result.lower_delay);
  result.upper_time=cristian_finite(probe.server_time+result.upper_delay);
  return result;
}
CristianExchange cristian_exchange(double server_time,double outbound_delay,double server_processing,double return_delay,double min_one_way_latency) {
  if(!std::isfinite(server_time) || !std::isfinite(outbound_delay) || outbound_delay<0 ||
     !std::isfinite(server_processing) || server_processing<0 || !std::isfinite(return_delay) || return_delay<0 ||
     !std::isfinite(min_one_way_latency) || min_one_way_latency<0 || outbound_delay<min_one_way_latency || return_delay<min_one_way_latency)
    throw std::invalid_argument("Cristian: invalid controlled exchange");
  const auto rtt=cristian_finite(static_cast<long double>(outbound_delay)+server_processing+return_delay);
  if((outbound_delay>0 || return_delay>0) && rtt==server_processing)
    throw std::overflow_error("Cristian: residence hides network duration");
  if(min_one_way_latency>(rtt-server_processing)-min_one_way_latency)
    throw std::overflow_error("Cristian: rounding invalidates network interval");
  CristianExchange result;result.outbound_delay=outbound_delay;result.return_delay=return_delay;
  result.analysis=cristian_analyze({server_time,rtt,min_one_way_latency,server_processing});
  (void)cristian_finite(static_cast<long double>(server_time)+return_delay);
  result.receipt_time=cristian_finite(server_time+return_delay);
  result.model_error=cristian_finite((static_cast<long double>(outbound_delay)-return_delay)/2);
  if(outbound_delay!=return_delay && result.model_error==0)
    throw std::overflow_error("Cristian: nonzero model error underflow");
  return result;
}
CristianSelection cristian_select(const std::vector<CristianProbe>& probes,std::size_t max_probes) {
  for(const auto& probe:probes) (void)cristian_analyze(probe);
  if(max_probes && probes.size()>max_probes) throw std::length_error("Cristian: complete probe cap exceeded");
  CristianSelection result;result.probes.reserve(probes.size());
  for(const auto& probe:probes) {
    result.probes.push_back(cristian_analyze(probe));
    const auto index=result.probes.size()-1;
    if(!result.has_probe || result.probes[index].error_bound<result.probes[result.index].error_bound) {
      result.has_probe=true;result.index=index;
    }
  }
  return result;
}
} // namespace datamunge::algorithms
