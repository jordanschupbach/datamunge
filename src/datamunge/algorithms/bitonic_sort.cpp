/// @file bitonic_sort.cpp
/// @brief Owned topology and integer-event recording from the canonical bitonic visitor.
#include <datamunge/algorithms/bitonic_sort.hpp>

namespace datamunge::algorithms {
BitonicNetwork bitonic_network(std::size_t wires) {
  BitonicNetwork result;
  result.wires=wires;
  /// @brief Store every generated comparator before grouping independent dependency layers.
  /// @param first First wire.
  /// @param second Second wire.
  /// @param ascending Whether the smaller value belongs at first.
  /// @param stage Dependency layer.
  auto collect=[&](std::size_t first,std::size_t second,bool ascending,std::size_t stage) {
    result.comparators.push_back({first,second,ascending,stage});
  };
  result.stages=detail::bitonic_sort_visit(0,wires,true,0,collect);
  std::vector<std::vector<BitonicComparator>> layers(result.stages);
  for (const auto& comparator:result.comparators) layers[comparator.stage].push_back(comparator);
  result.comparators.clear();
  for (const auto& layer:layers) result.comparators.insert(result.comparators.end(),layer.begin(),layer.end());
  return result;
}

BitonicTrace bitonic_sort_trace(std::vector<int> values) {
  BitonicTrace result;
  result.input=values;
  result.output=std::move(values);
  result.stages=detail::bitonic_apply(result.output,
      [&](BitonicComparator comparator,bool swapped,int first,int second) {
        result.steps.push_back({comparator,swapped?second:first,swapped?first:second,first,second,swapped});
      });
  return result;
}
} // namespace datamunge::algorithms
