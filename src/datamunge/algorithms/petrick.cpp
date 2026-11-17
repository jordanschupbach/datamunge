/// @file petrick.cpp
/// @brief Canonical producing implementation and actual-site diagnostics; public contracts are in the paired header.
#include <algorithm>
#include <datamunge/algorithms/petrick.hpp>
#include <limits>
#include <set>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief File-local validated kernels shared by ordinary and observed entry points.
namespace {
/// @brief Admit cumulative logical work/history before allocating its next record.
void admit(std::size_t& n, std::size_t add, std::size_t cap) {
  if (add > std::numeric_limits<std::size_t>::max() - n || (cap && (n > cap || add > cap - n)))
    throw std::length_error("Petrick work/history policy");
  n += add;
}
/// @brief Shared native expansion, optionally observing actual retained terms.
PetrickTrace solve(const std::vector<std::vector<int>>& chart, PetrickOptions o, bool capture) {
  for (const auto& c : chart)
    for (int id : c)
      if (id < 0)
        throw std::invalid_argument("Petrick ids must be nonnegative");
  PetrickTrace               r;
  std::size_t                retained = 0;
  std::vector<std::set<int>> terms    = {{}};
  for (std::size_t ci = 0; ci < chart.size(); ++ci) {
    const auto&                clause = chart[ci];
    PetrickStage               stage{ci, 0, 0, {}};
    std::vector<std::set<int>> next;
    for (const auto& term : terms)
      for (int pi : clause) {
        admit(r.generated, 1, o.max_candidates);
        ++stage.generated;
        auto t = term;
        t.insert(pi);
        next.push_back(std::move(t));
      }
    std::sort(
        next.begin(), next.end(), [](const auto& a, const auto& b) { return a.size() < b.size(); });
    std::vector<std::set<int>> reduced;
    for (const auto& t : next) {
      bool absorbed = false;
      for (const auto& s : reduced)
        if (std::includes(t.begin(), t.end(), s.begin(), s.end())) {
          absorbed = true;
          break;
        }
      if (absorbed)
        ++stage.absorbed;
      else
        reduced.push_back(t);
    }
    terms = std::move(reduced);
    if (capture) {
      admit(retained, 1, o.max_trace_entries);
      for (const auto& t : terms) {
        admit(retained, 1, o.max_trace_entries);
        admit(retained, t.size(), o.max_trace_entries);
        stage.terms.emplace_back(t.begin(), t.end());
      }
      r.stages.push_back(std::move(stage));
    }
    if (terms.empty())
      break;
  }
  auto best = std::numeric_limits<std::size_t>::max();
  for (const auto& t : terms)
    best = std::min(best, t.size());
  for (const auto& t : terms)
    if (t.size() == best)
      r.result.minimal_covers.emplace_back(t.begin(), t.end());
  return r;
}
} // namespace
PetrickResult petrick(const std::vector<std::vector<int>>& chart) {
  return solve(chart, {}, false).result;
}
PetrickTrace petrick_trace(const std::vector<std::vector<int>>& chart, PetrickOptions o) {
  return solve(chart, o, true);
}
} // namespace datamunge::algorithms
