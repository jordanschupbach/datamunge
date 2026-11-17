/// @file knuth_bendix.cpp
/// @brief Strict-shortlex completion and actual deterministic rewrite capture.
#include <datamunge/algorithms/knuth_bendix.hpp>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

/// @brief Local shortlex, deterministic reduction and critical-pair helpers.
namespace {

/// @brief Compare finite byte words by length then std::string lexical order.
/// @param a First word.
/// @param b Second word.
/// @return True exactly when a is strictly larger in shortlex.
bool shortlex_gt(const std::string& a, const std::string& b) {
  if (a.size() != b.size())
    return a.size() > b.size();
  return a > b;
}

/// @brief Validate every supplied reducer rule before any word mutation/copies.
/// @param rules Complete borrowed ordered rules.
/// @throws std::invalid_argument On empty lhs or a nondecreasing shortlex rule.
void validate_rules(const std::vector<RewriteRule>& rules) {
  for (const auto& r : rules)
    if (r.lhs.empty() || !shortlex_gt(r.lhs, r.rhs))
      throw std::invalid_argument("KB requires strict shortlex rules");
}
/// @brief Admit capture records before original-input copying or actual replacement.
/// @param trace Borrowed local optional actual output.
/// @param amount Initial-word/rule/application records.
/// @param cap Optional complete record ceiling, zero off.
/// @throws std::length_error On count overflow or enabled quota.
void admit(KBRewriteTrace* trace, std::size_t amount, std::size_t cap) {
  if (!trace)
    return;
  if (amount > std::numeric_limits<std::size_t>::max() - trace->captured_entries)
    throw std::length_error("KB trace count");
  const auto next = trace->captured_entries + amount;
  if (cap && next > cap)
    throw std::length_error("KB trace quota");
  trace->captured_entries = next;
}
/// @brief Perform the first applicable rule's leftmost actual replacement once.
/// @param w Mutable local producing word.
/// @param rules Complete valid strict-shortlex rules, input order.
/// @param trace Optional producing actual capture, default null.
/// @param cap Optional capture-record ceiling, zero off.
/// @return True exactly if one replacement happened.
/// @throws std::length_error On capture/string/container limits.
/// @throws std::bad_alloc On word/history allocation failure.
bool rewrite_once(
    std::string&                    w,
    const std::vector<RewriteRule>& rules,
    KBRewriteTrace*                 trace = nullptr,
    std::size_t                     cap   = 0) {
  for (std::size_t i = 0; i < rules.size(); ++i) {
    const auto& r   = rules[i];
    const auto  pos = w.find(r.lhs);
    if (pos != std::string::npos) {
      admit(trace, 1, cap);
      std::string before;
      if (trace)
        before = w;
      w.replace(pos, r.lhs.size(), r.rhs);
      if (trace)
        trace->steps.push_back({i, pos, std::move(before), w});
      return true;
    }
  }
  return false;
}
/// @brief Reduce a local owned word under valid terminating rules, optionally observing actual steps.
/// @param w Owned local original word, mutated during reduction.
/// @param rules Borrowed valid strict-shortlex rules.
/// @param trace Optional local actual capture, default null.
/// @param cap Optional capture ceiling, zero off.
/// @return Owned actual irreducible word.
/// @throws std::length_error On capture/string/container limits.
/// @throws std::bad_alloc On word/history allocation failure.
std::string normal_form(
    std::string                     w,
    const std::vector<RewriteRule>& rules,
    KBRewriteTrace*                 trace = nullptr,
    std::size_t                     cap   = 0) {
  while (rewrite_once(w, rules, trace, cap)) {
  }
  return w;
}
/// @brief Test whether an exact oriented pair already exists in actual rule order.
/// @param rules Borrowed actual rules.
/// @param l Proposed left side.
/// @param r Proposed right side.
/// @return True for an exact existing pair.
bool rule_exists(
    const std::vector<RewriteRule>& rules,
    const std::string&              l,
    const std::string&              r) {
  for (const auto& x : rules)
    if (x.lhs == l && x.rhs == r)
      return true;
  return false;
}

/// @brief Orient and append one distinct nontrivial equation.
/// @param rules Mutable actual rule sequence.
/// @param u First equality side.
/// @param v Second equality side.
/// @return True exactly if one new rule was appended.
/// @throws std::length_error On string/container size limits.
/// @throws std::bad_alloc On rule allocation failure.
bool add_equation(std::vector<RewriteRule>& rules, const std::string& u, const std::string& v) {
  if (u == v)
    return false;
  const std::string& big   = shortlex_gt(u, v) ? u : v;
  const std::string& small = shortlex_gt(u, v) ? v : u;
  if (rule_exists(rules, big, small))
    return false;
  rules.push_back({big, small});
  return true;
}

/// @brief Append actual ordered suffix/prefix and containment fork reducts.
/// @param r1 First snapshot rule.
/// @param r2 Second snapshot rule.
/// @param pairs Mutable complete ordered round fork list.
/// @throws std::length_error On word/container representability limits.
/// @throws std::bad_alloc On fork allocation failure.
void critical_pairs(
    const RewriteRule&                                r1,
    const RewriteRule&                                r2,
    std::vector<std::pair<std::string, std::string>>& pairs) {
  const std::string& L1 = r1.lhs;
  const std::string& L2 = r2.lhs;

  // Overlap: a proper suffix of L1 equals a proper prefix of L2.
  const std::size_t maxk = L1.size() < L2.size() ? L1.size() : L2.size();
  for (std::size_t k = 1; k <= maxk; ++k) {
    if (k == L1.size() && k == L2.size())
      continue; // identical, no genuine overlap
    if (L1.compare(L1.size() - k, k, L2, 0, k) == 0) {
      // overlap word w = L1 + tail(L2)
      const std::string u = r1.rhs + L2.substr(k);                // rewrite L1 prefix
      const std::string v = L1.substr(0, L1.size() - k) + r2.rhs; // rewrite L2 suffix
      pairs.emplace_back(u, v);
    }
  }

  // Containment: L2 occurs inside L1 as L1 = a L2 b.
  if (L2.size() <= L1.size()) {
    for (std::size_t p = 0; p + L2.size() <= L1.size(); ++p) {
      if (L1.compare(p, L2.size(), L2) == 0) {
        const std::string u = r1.rhs;
        const std::string v = L1.substr(0, p) + r2.rhs + L1.substr(p + L2.size());
        pairs.emplace_back(u, v);
      }
    }
  }
}

/// @brief Preserve equations while reducing every side against the other actual rules.
/// Reducible lhs is removed and its normalized equality reoriented; no confluence
/// claim follows from this interreduction alone.
/// @param rules Mutable actual strict-shortlex rule sequence.
/// @throws std::length_error On string/container limits.
/// @throws std::bad_alloc On temporary rule/word allocation failure.
void interreduce(std::vector<RewriteRule>& rules) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < rules.size();) {
      std::vector<RewriteRule> others;
      for (std::size_t j = 0; j < rules.size(); ++j)
        if (j != i)
          others.push_back(rules[j]);
      const std::string lhsNF = normal_form(rules[i].lhs, others);
      if (lhsNF != rules[i].lhs) {
        // lhs reducible by others: this rule is redundant; re-add as an equation.
        const std::string rhsNF = normal_form(rules[i].rhs, others);
        rules.erase(rules.begin() + i);
        if (add_equation(rules, lhsNF, rhsNF))
          changed = true;
        changed = true;
        continue;
      }
      const std::string rhsNF = normal_form(rules[i].rhs, others);
      if (rhsNF != rules[i].rhs) {
        rules[i].rhs = rhsNF;
        changed      = true;
      }
      ++i;
    }
  }
}

} // namespace

KnuthBendixResult knuth_bendix(
    const std::vector<std::pair<std::string, std::string>>& equations,
    int                                                     max_rules) {
  if (max_rules <= 0)
    throw std::invalid_argument("KB max_rules must be positive");
  std::vector<RewriteRule> rules;
  for (const auto& [l, r] : equations) {
    if (l == r)
      continue;
    const auto& big   = shortlex_gt(l, r) ? l : r;
    const auto& small = shortlex_gt(l, r) ? r : l;
    if (rule_exists(rules, big, small))
      continue;
    if (rules.size() >= static_cast<std::size_t>(max_rules))
      return {rules, false};
    add_equation(rules, l, r);
  }

  int  guard    = 0;
  bool complete = false;
  while (guard++ < 2000) {
    interreduce(rules);
    bool added = false;
    // Snapshot to iterate over a stable set this round.
    const std::vector<RewriteRule>                   snap = rules;
    std::vector<std::pair<std::string, std::string>> pairs;
    for (const auto& a : snap)
      for (const auto& b : snap)
        critical_pairs(a, b, pairs);

    for (const auto& [u, v] : pairs) {
      const std::string nu = normal_form(u, rules);
      const std::string nv = normal_form(v, rules);
      if (nu != nv) {
        const auto& big   = shortlex_gt(nu, nv) ? nu : nv;
        const auto& small = shortlex_gt(nu, nv) ? nv : nu;
        if (!rule_exists(rules, big, small) && rules.size() >= static_cast<std::size_t>(max_rules))
          return {rules, false};
        if (add_equation(rules, nu, nv))
          added = true;
      }
    }
    if (!added) {
      complete = true;
      break;
    }
  }
  interreduce(rules);
  return {rules, complete};
}

std::string kb_normal_form(const std::string& word, const std::vector<RewriteRule>& rules) {
  validate_rules(rules);
  return normal_form(word, rules);
}

KBRewriteTrace kb_rewrite_trace(
    const std::string&              word,
    const std::vector<RewriteRule>& rules,
    std::size_t                     max_entries) {
  validate_rules(rules);
  KBRewriteTrace out;
  admit(&out, 1, max_entries);
  admit(&out, rules.size(), max_entries);
  out.start       = word;
  out.rules       = rules;
  out.normal_form = normal_form(word, rules, &out, max_entries);
  return out;
}
} // namespace datamunge::algorithms
