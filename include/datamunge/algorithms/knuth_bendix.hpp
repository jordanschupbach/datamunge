#pragma once
/// @file knuth_bendix.hpp
/// @brief Shortlex string completion and checked same-execution rewriting observations.
#include <cstddef>
#include <string>
#include <utility>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned byte-string rewrite lhs->rhs, required to strictly decrease shortlex for reduction.
/// Shortlex orders length first then std::string lexical order. Equal-length
/// lexical decreases are valid; length need not decrease at every step.
/// @par Example
/// @code
/// const dm::algorithms::RewriteRule r{"ba","ab"};
/// std::print("{}\n",dm::algorithms::kb_normal_form("baba",{r})); // aabb
/// @endcode
/// @see kb_normal_form()
struct RewriteRule {
  std::string lhs; ///< Owned nonempty pattern on oriented valid rules.
  std::string rhs; ///< Owned strictly smaller replacement, possibly empty for identity.
};
/// @brief Owned generated shortlex rules and actual finite completion status.
/// complete=true means the final exhaustive ordered overlap/containment scan
/// joined every critical pair. With strict terminating shortlex rules this gives
/// confluence and a canonical form for the supplied presentation. false means a
/// rule/round ceiling stopped the procedure: equality of reductions is sound,
/// but unequal reductions do not decide inequivalence. Initial rule-limit exits
/// may have processed only a prefix of the supplied equations. No group/monoid
/// finiteness or convergence promise follows from a finite test of words.
/// @par Example
/// @code
/// const auto r=dm::algorithms::knuth_bendix({{"ba","ab"}});
/// std::print("{} {}\n",r.complete,r.rules.size()); // true 1
/// @endcode
/// @see knuth_bendix()
struct KnuthBendixResult {
  std::vector<RewriteRule> rules; ///< Owned actual oriented rules, deterministic producer order.
  bool complete{false};           ///< True only after a complete no-new-rule critical-pair scan.
};
/// @brief Owned actual rule application from deterministic first-rule/leftmost reduction.
/// Rule index is in the original supplied rule order, not an inferred canonical
/// system. before/after strings are actual producing values, not prefix replay.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kb_rewrite_trace("ba",{{"ba","ab"}});
/// std::print("{} {}\n",t.steps[0].position,t.steps[0].after); // 0 ab
/// @endcode
/// @see KBRewriteTrace
struct KBRewriteStep {
  std::size_t rule_index{0}; ///< Actual first applicable rule in supplied order.
  std::size_t position{0};   ///< Actual leftmost byte occurrence of that rule's lhs.
  std::string before;        ///< Owned complete actual word before replacement.
  std::string after;         ///< Owned complete actual word after replacement.
};
/// @brief Owned actual finite reduction and complete original word/rules.
/// captured_entries counts one initial word, one per input rule and one per
/// application; excludes individual bytes/inner strings/capacities/runtime copies.
/// A trace step stores two whole words, so this record quota is not a byte or
/// peak-memory bound. Rules are validated before copies, and applications admitted
/// before replacement. No second reduction or partial returned trace.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kb_rewrite_trace("baba",{{"ba","ab"}});
/// std::print("{} {}\n",t.normal_form,t.steps.size()); // aabb 3
/// @endcode
/// @see kb_rewrite_trace()
struct KBRewriteTrace {
  std::string                start;       ///< Owned complete original input bytes.
  std::vector<RewriteRule>   rules;       ///< Owned complete validated oriented input rules.
  std::vector<KBRewriteStep> steps;       ///< Actual chronological producing replacements.
  std::string                normal_form; ///< Actual irreducible final byte word under these rules.
  std::size_t captured_entries{0};        ///< Logical initial-word+rule+application record count.
};
/// @brief Attempt finite shortlex Knuth–Bendix completion of byte-string equations.
/// Orient nontrivial equations by length then std::string lexical order. Empty
/// strings represent identity; empty-equals-empty is ignored. Interreduce rules,
/// then examine ordered suffix/prefix overlaps and contained lhs occurrences on
/// a stable round snapshot. Normalize each pair under the actual growing rules,
/// adding an oriented equation when unequal. Stops complete only after a full
/// no-addition scan; rule ceiling and fixed2000-round guard return incomplete.
/// max_rules must be positive; no result contains more rules than that ceiling.
/// Resource work may be very large: completion is not guaranteed to terminate
/// without ceilings, and these rule/round caps do not bound pairs/word bytes/time.
/// Inputs unchanged, result owns rules, no shared mutable state. Rules strictly
/// decrease shortlex, not necessarily length. Normal forms are canonical only
/// when complete, not merely because finitely many words appeared to agree.
/// @param equations Borrowed byte-string equality pairs, possibly empty/identity.
/// @param max_rules Positive complete retained-rule ceiling, default400.
/// @return Owned actual rules and honest complete/incomplete status.
/// @throws std::invalid_argument On nonpositive max_rules.
/// @throws std::length_error On string/container representability limits.
/// @throws std::bad_alloc On rule/pair/word allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::knuth_bendix({{"aa",""}});
/// std::print("{} {}\n",r.complete,dm::algorithms::kb_normal_form("aaa",r.rules)); // true a
/// @endcode
/// @par References
/// - Knuth and Bendix(1970), Simple Word Problems in Universal Algebras, in Computational Problems in Abstract Algebra,263–297, https://doi.org/10.1016/B978-0-08-012975-4.50028-X.
/// @see kb_normal_form(), kb_rewrite_trace()
KnuthBendixResult knuth_bendix(
    const std::vector<std::pair<std::string, std::string>>& equations,
    int                                                     max_rules = 400);
/// @brief Reduce a byte word using the first applicable rule and its leftmost occurrence.
/// Validate every rule first: lhs must be nonempty and strictly greater than rhs
/// in shortlex. This rejects identity/increasing/cyclic rules that could hang the
/// historical unchecked reducer. Termination follows from strict well-founded
/// shortlex descent, including equal-length lexical steps. For arbitrary valid
/// nonconfluent rules the result depends on rule order and is not canonical.
/// Work depends on actual replacements and std::string::find/replace costs, not
/// simply original length. O(max word length) runtime copies; no trace or quota.
/// Caller word/rules unchanged; returned string owns bytes, embedded NUL allowed.
/// @param word Borrowed arbitrary finite byte word, possibly empty.
/// @param rules Borrowed complete ordered strict-shortlex rules, possibly empty.
/// @return Owned actual irreducible result under this deterministic rule order.
/// @throws std::invalid_argument On any empty-lhs or nondecreasing rule, even if unused.
/// @throws std::length_error On string/container representability limits.
/// @throws std::bad_alloc On word allocation failure.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::kb_normal_form("bbaa",{{"ba","ab"}})); // aabb
/// @endcode
/// @par References
/// - Knuth and Bendix(1970), Simple Word Problems in Universal Algebras, https://doi.org/10.1016/B978-0-08-012975-4.50028-X.
/// @see kb_rewrite_trace(), knuth_bendix()
std::string kb_normal_form(const std::string& word, const std::vector<RewriteRule>& rules);
/// @brief Capture every actual first-rule/leftmost shortlex reduction application once.
/// Same complete rule validation, byte order and termination as kb_normal_form().
/// Complete initial word/rule count admitted before copies; each application
/// admitted before replacement. Adds owned before/after whole-word histories;
/// record quota excludes their bytes and is not a resource/time guarantee.
/// @param word Borrowed complete original byte word.
/// @param rules Borrowed complete ordered strict-shortlex rules.
/// @param max_entries Initial-word+rule+application ceiling, default1000000, zero off.
/// @return Owned complete original inputs, actual applications and irreducible result.
/// @throws std::invalid_argument On any invalid input rule.
/// @throws std::length_error On enabled capture ceiling or count/string/container limits.
/// @throws std::bad_alloc On word/rule/history allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::kb_rewrite_trace("ba",{{"ba","ab"}},3);
/// std::print("{} {}\n",t.normal_form,t.captured_entries); // ab 3
/// @endcode
/// @par References
/// - Knuth and Bendix(1970), Simple Word Problems in Universal Algebras, https://doi.org/10.1016/B978-0-08-012975-4.50028-X.
/// @see KBRewriteTrace, kb_normal_form()
KBRewriteTrace kb_rewrite_trace(
    const std::string&              word,
    const std::vector<RewriteRule>& rules,
    std::size_t                     max_entries = 1000000);
} // namespace datamunge::algorithms
