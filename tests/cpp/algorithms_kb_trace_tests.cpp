#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <datamunge/algorithms/knuth_bendix.hpp>
#include <map>
#include <set>
using namespace datamunge::algorithms;
namespace {
bool gt(const std::string& a, const std::string& b) {
  return a.size() > b.size() || (a.size() == b.size() && a > b);
}
std::array<int, 3> permutation(const std::string& w) {
  std::array<int, 3> p{0, 1, 2};
  for (char c : w) {
    const std::array<int, 3> g =
        c == 'a' ? std::array<int, 3>{1, 0, 2} : std::array<int, 3>{1, 2, 0};
    auto old = p;
    for (int i = 0; i < 3; ++i)
      p[i] = old[g[i]];
  }
  return p;
}
void verify(const KBRewriteTrace& t) {
  std::string w = t.start;
  for (const auto& e : t.steps) {
    ASSERT_EQ(w, e.before);
    ASSERT_LT(e.rule_index, t.rules.size());
    for (std::size_t i = 0; i < e.rule_index; ++i)
      EXPECT_EQ(w.find(t.rules[i].lhs), std::string::npos);
    const auto& r = t.rules[e.rule_index];
    EXPECT_EQ(w.find(r.lhs), e.position);
    w.replace(e.position, r.lhs.size(), r.rhs);
    EXPECT_EQ(w, e.after);
    EXPECT_TRUE(gt(e.before, e.after));
  }
  EXPECT_EQ(w, t.normal_form);
  for (const auto& r : t.rules)
    EXPECT_EQ(w.find(r.lhs), std::string::npos);
  EXPECT_EQ(t.captured_entries, 1 + t.rules.size() + t.steps.size());
  EXPECT_EQ(w, kb_normal_form(t.start, t.rules));
}
void critical_certificate(const std::vector<RewriteRule>& rules) {
  for (const auto& a : rules)
    for (const auto& b : rules)
      for (int shift = 1 - int(b.lhs.size()); shift < int(a.lhs.size()); ++shift) {
        const int lo = std::min(0, shift),
                  hi = std::max(int(a.lhs.size()), shift + int(b.lhs.size()));
        std::string w(hi - lo, '?');
        bool        compatible = true;
        for (int i = 0; i < int(a.lhs.size()); ++i)
          w[i - lo] = a.lhs[i];
        for (int i = 0; i < int(b.lhs.size()); ++i) {
          auto& c = w[shift + i - lo];
          if (c != '?' && c != b.lhs[i])
            compatible = false;
          c = b.lhs[i];
        }
        if (!compatible)
          continue;
        auto u = w, v = w;
        u.replace(-lo, a.lhs.size(), a.rhs);
        v.replace(shift - lo, b.lhs.size(), b.rhs);
        EXPECT_EQ(kb_normal_form(u, rules), kb_normal_form(v, rules));
      }
}
} // namespace
TEST(KBTrace, OriginalS3AndIndependentPermutationModel) {
  const auto c = knuth_bendix({{"aa", ""}, {"bbb", ""}, {"abab", ""}});
  ASSERT_TRUE(c.complete);
  critical_certificate(c.rules);
  std::map<std::array<int, 3>, std::string> seen;
  std::set<std::string>                     forms;
  for (int n = 0; n <= 10; ++n)
    for (int word = 0; word < (1 << n); ++word) {
      std::string w;
      for (int i = 0; i < n; ++i)
        w += ((word >> i) & 1) ? 'b' : 'a';
      const auto t = kb_rewrite_trace(w, c.rules);
      verify(t);
      const auto p = permutation(w);
      EXPECT_EQ(permutation(t.normal_form), p);
      if (seen.contains(p))
        EXPECT_EQ(seen[p], t.normal_form);
      seen[p] = t.normal_form;
      forms.insert(t.normal_form);
    }
  EXPECT_EQ(forms.size(), 6U);
  EXPECT_EQ(seen.size(), 6U);
  verify(kb_rewrite_trace("bbababbaab", c.rules));
}
TEST(KBTrace, EqualLengthDescentAndNonconfluentRuleOrder) {
  const auto t = kb_rewrite_trace("bbaa", {{"ba", "ab"}});
  verify(t);
  ASSERT_EQ(t.normal_form, "aabb");
  EXPECT_EQ(t.steps.size(), 4U);
  for (const auto& e : t.steps)
    EXPECT_EQ(e.before.size(), e.after.size());
  EXPECT_EQ(kb_normal_form("ab", {{"ab", "a"}, {"ab", "b"}}), "a");
  EXPECT_EQ(kb_normal_form("ab", {{"ab", "b"}, {"ab", "a"}}), "b");
}
TEST(KBTrace, CompleteCriticalPairsAndCappedIncompletePresentation) {
  for (const auto& eq : std::vector<std::vector<std::pair<std::string, std::string>>>{
           {}, {{"ab", "ba"}}, {{"aa", ""}}, {{"aa", ""}, {"bb", ""}, {"abab", ""}}}) {
    const auto c = knuth_bendix(eq);
    ASSERT_TRUE(c.complete);
    critical_certificate(c.rules);
  }
  const auto initial = knuth_bendix({{"aa", ""}, {"bbb", ""}}, 1);
  EXPECT_FALSE(initial.complete);
  EXPECT_EQ(initial.rules.size(), 1U);
  const auto later = knuth_bendix({{"aa", ""}, {"bbb", ""}, {"abab", ""}}, 3);
  EXPECT_FALSE(later.complete);
  EXPECT_LE(later.rules.size(), 3U);
  EXPECT_THROW((void)knuth_bendix({}, 0), std::invalid_argument);
  EXPECT_THROW((void)knuth_bendix({}, -1), std::invalid_argument);
}
TEST(KBTrace, RejectInvalidUnusedRulesInsteadOfLooping) {
  for (const auto& rules : std::vector<std::vector<RewriteRule>>{
           {{"", "a"}}, {{"a", "a"}}, {{"a", "aa"}}, {{"a", "b"}}, {{"b", "a"}, {"a", "b"}}}) {
    EXPECT_THROW((void)kb_normal_form("", rules), std::invalid_argument);
    EXPECT_THROW((void)kb_rewrite_trace("unused", rules), std::invalid_argument);
  }
  const std::string hi(1, char(255)), lo(1, char(128));
  EXPECT_EQ(kb_normal_form(hi, {{hi, lo}}), lo);
}
TEST(KBTrace, CompleteAdmissionAndOwnedSnapshots) {
  const auto t = kb_rewrite_trace("bbaa", {{"ba", "ab"}});
  EXPECT_NO_THROW((void)kb_rewrite_trace(t.start, t.rules, t.captured_entries));
  EXPECT_THROW((void)kb_rewrite_trace(t.start, t.rules, t.captured_entries - 1), std::length_error);
  EXPECT_THROW((void)kb_rewrite_trace(t.start, t.rules, 1), std::length_error);
  EXPECT_NO_THROW((void)kb_rewrite_trace(t.start, t.rules, 0));
  auto c = t;
  c.start.clear();
  c.rules[0].lhs.clear();
  c.steps[0].before.clear();
  c.normal_form.clear();
  EXPECT_EQ(t.start, "bbaa");
  EXPECT_EQ(t.rules[0].lhs, "ba");
  EXPECT_FALSE(t.steps[0].before.empty());
  EXPECT_EQ(t.normal_form, "aabb");
}
