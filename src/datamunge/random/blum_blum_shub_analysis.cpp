/// @file blum_blum_shub_analysis.cpp
/// @brief Actual BBS bit recording and bounded first-repeat analysis from the engine's shared transition.
#include <datamunge/random/blum_blum_shub_analysis.hpp>
#include <unordered_map>
namespace datamunge::random {
std::vector<BlumBlumShubStep> blum_blum_shub_trace(BlumBlumShub& engine,std::size_t count) {
  std::vector<BlumBlumShubStep> result;result.reserve(count);
  for (std::size_t i=0;i<count;++i) {
    const auto before=engine.state(),bit=engine.next_bit();
    result.push_back({before,engine.state(),bit});
  }
  return result;
}
BlumBlumShubOrbit blum_blum_shub_orbit(const BlumBlumShub& engine,std::size_t max_steps) {
  auto copy=engine;BlumBlumShubOrbit result;result.states.push_back(copy.state());
  std::unordered_map<std::uint64_t,std::size_t> first{{copy.state(),0}};
  for (std::size_t i=0;i<max_steps;++i) {
    copy.next_bit();const auto state=copy.state();result.states.push_back(state);
    const auto [entry,inserted]=first.emplace(state,i+1);
    if (!inserted) {result.closed=true;result.transient=entry->second;result.period=i+1-entry->second;break;}
  }
  return result;
}
} // namespace datamunge::random
