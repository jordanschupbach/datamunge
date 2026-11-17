#pragma once
/// @file index_calculus_analysis.hpp
/// @brief Native index-calculus preparation diagram and explicitly illustrative cost models.
#include <cmath>
#include <datamunge/algorithms/index_calculus.hpp>
#include <datamunge/plot/diagram.hpp>
#include <format>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief One illustrative logarithmic asymptotic-model comparison, not measured work.
/// Bits denotes log2 of a hypothetical modulus, including sizes outside uint64.
/// These formula values omit hidden constants/lower-order terms and do not predict
/// this implementation's capped factor base, finite budgets or crossover timing.
/// @par Example
/// @code
/// const dm::algorithms::IndexCalculusModelPoint p;
/// std::print("{}\n",p.bits); // 0
/// @endcode
/// @see index_calculus_cost_models()
struct IndexCalculusModelPoint {
  double bits{0};          ///< Supplied hypothetical log2(p).
  double index_log10{0};   ///< c*sqrt(log p*log log p)/log(10), evaluated without forming exp().
  double generic_log10{0}; ///< log10(sqrt(p))=bits*log(2)/(2*log(10)).
};
/// @brief Evaluate illustrative L_p[1/2,c] and square-root models directly in log space.
/// Validates complete finite inputs before output allocation. O(N) time/storage;
/// empty input gives empty output after validating c. These are formulas, not
/// measured runtime, cryptographic security estimates or guarantees for the
/// bounded uint64 solver. No mutation, retained references or shared state.
/// @param bits Borrowed finite hypothetical bit lengths, each at least two.
/// @param c Positive finite illustrative L constant, default sqrt(2).
/// @return Owned formula values in supplied order.
/// @throws std::invalid_argument On nonfinite/too-small bit length or invalid c.
/// @throws std::overflow_error On a nonfinite computed model value.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto p=dm::algorithms::index_calculus_cost_models({30});
/// std::print("{:.2f} {:.2f}\n",p[0].index_log10,p[0].generic_log10); // 4.88 4.52
/// @endcode
/// @par References
/// - Menezes, van Oorschot and Vanstone (1996), Handbook of Applied Cryptography,
///   Note 3.71, https://cacr.uwaterloo.ca/hac/about/chap3.pdf. L-model notation;
///   sqrt(2) preserves the original report's illustrative constant, not a fitted value.
/// @see index_calculus_prepare()
inline std::vector<IndexCalculusModelPoint> index_calculus_cost_models(
    const std::vector<double>& bits,
    double                     c = std::sqrt(2.)) {
  if (!std::isfinite(c) || c <= 0)
    throw std::invalid_argument("index model positive finite constant");
  for (auto b : bits)
    if (!std::isfinite(b) || b < 2)
      throw std::invalid_argument("index model finite bits >=2");
  std::vector<IndexCalculusModelPoint> out;
  out.reserve(bits.size());
  for (auto b : bits) {
    const double ln      = b * std::log(2.);
    const double ic      = c * std::sqrt(ln) * std::sqrt(std::log(ln)) / std::log(10.),
                 generic = b * (std::log(2.) / (2 * std::log(10.)));
    if (!std::isfinite(ic) || !std::isfinite(generic))
      throw std::overflow_error("index model range");
    out.push_back({b, ic, generic});
  }
  return out;
}
/// @brief Draw actual relation/stage/table summaries without replaying preparation.
/// The first relation is explicitly labelled as a single selected example; all
/// accepted/trial counts and all stages/base logs are shown. Complete display
/// quotas precede layout and table verification. Querying target one validates
/// the public table without random trials; this is extra diagnostic validation,
/// not producing work. Historical record dimensions/scalars are checked, but
/// an edited trace's complete mathematical provenance is not certified.
/// @param trace Borrowed actual preparation records/table.
/// @param max_variables Complete factor-base display cap, default12; zero disables.
/// @param max_stages Complete linear-stage display cap, default16; zero disables.
/// @return Owned fitted native Diagram, including explicit not-ready status.
/// @throws std::length_error On exceeded nonzero complete display quotas.
/// @throws std::invalid_argument On invalid table or historical shapes/indices/scalars.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare_trace(2,5);
/// std::print("{}\n",dm::algorithms::index_calculus_preparation_diagram(t).to_svg().empty()); // false
/// @endcode
/// @see index_calculus_prepare_trace(), index_calculus_query()
inline graphics::Diagram index_calculus_preparation_diagram(
    const IndexCalculusPreparationTrace& trace,
    std::size_t                          max_variables = 12,
    std::size_t                          max_stages    = 16) {
  const auto& t = trace.table;
  const auto  k = t.factor_base.size();
  if ((max_variables && k > max_variables) || (max_stages && trace.stages.size() > max_stages))
    throw std::length_error("index complete diagram cap");
  (void)index_calculus_query(t, 1, 0);
  if (trace.relations.size() != t.work.accepted_relations)
    throw std::invalid_argument("index diagram relation count");
  std::size_t previous = 0;
  for (const auto& r : trace.relations) {
    if (r.powers.size() != k || r.attempt <= previous || r.attempt > t.work.relation_attempts ||
        !r.exponent || r.exponent >= t.p - 1 || !r.residue || r.residue >= t.p)
      throw std::invalid_argument("index diagram relation shape");
    previous = r.attempt;
  }
  std::string relations =
      std::format("Accepted {} / {} trials", trace.relations.size(), t.work.relation_attempts);
  if (!trace.relations.empty()) {
    const auto& r = trace.relations.front();
    relations += std::format("\nFirst only:\n{}^{} mod {} = {}", t.g, r.exponent, t.p, r.residue);
  }
  std::string stages = "Actual linear stages";
  for (const auto& s : trace.stages) {
    if (s.prime < 2 || s.modulus < s.prime || s.rank > k ||
        (!s.solution.empty() && s.solution.size() != k))
      throw std::invalid_argument("index diagram stage shape");
    stages += std::format("\nmod {}: rank {} / {}", s.modulus, s.rank, k);
  }
  std::string logs =
      t.status == IndexCalculusStatus::ready ? "Verified CRT logs" : "Table not ready";
  if (t.status == IndexCalculusStatus::ready)
    for (std::size_t i = 0; i < k; ++i)
      logs += std::format("\nlog({}) = {}", t.factor_base[i], t.logarithms[i]);
  graphics::Diagram          d;
  graphics::DiagramNodeStyle style;
  style.width = 4.6;
  d.node("relations", {0, 0}, relations, style)
      .node("stages", {7, 0}, stages, style)
      .node("logs", {14, 0}, logs, style)
      .edge("relations", "stages")
      .edge("stages", "logs");
  return d;
}
} // namespace datamunge::algorithms
