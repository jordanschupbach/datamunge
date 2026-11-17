/// @file nysiis.cpp
/// @brief Legacy native NYSIIS byte-rule producer with actual owned stage/token observations.
#include <cctype>
#include <datamunge/algorithms/nysiis.hpp>
#include <datamunge/algorithms/phonetic.hpp>
#include <limits>
#include <map>
#include <stdexcept>
#include <string_view>
namespace datamunge::algorithms {
/// @brief Private complete logical-byte admission and shared legacy rule producer.
namespace {
/// @brief Add actual logical counts without wrapping.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error If the count overflows.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("NYSIIS: capture count overflow");
  return a + b;
}
/// @brief Test membership in the five actual native uppercase vowel bytes.
/// @param c Actual rewritten byte.
/// @return True only for A,E,I,O,U.
bool vowel(char c) {
  return c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U';
}
/// @brief Execute original native prefix/suffix/token/cleanup rules with optional actual capture.
/// @param name Complete original bytes, all strings accepted.
/// @param out Optional owned actual-history sink.
/// @param cap Actual logical-byte/record policy, zero disables.
/// @return Actual native key and producing token counters.
/// @throws std::length_error If string/count geometry or actual quota is exceeded.
NysiisAnalysis run(const std::string& name, NysiisTrace* out, std::size_t cap) {
  /// @brief Admit actual retained byte/record units before their copy or append.
  auto admit = [&](std::size_t count) {
    if (!out)
      return;
    auto next = add(out->captured_entries, count);
    if (cap && next > cap)
      throw std::length_error("NYSIIS: actual capture quota");
    out->captured_entries = next;
  };
  NysiisAnalysis result;
  result.input_bytes = name.size();
  if (out) {
    admit(add(name.size(), name.size()));
    out->input = name;
  }
  if (name.empty())
    return result;
  std::string s = name;
  for (auto& c : s)
    c = char(std::toupper(static_cast<unsigned char>(c)));
  if (out)
    out->uppercase = s;
  /// @brief Begin one actual record, admitting each preexisting string before retaining it.
  auto begin = [&](std::string_view   stage,
                   std::string_view   rule,
                   std::size_t        first,
                   std::size_t        end,
                   const std::string& before,
                   std::string_view   token,
                   bool               appended) {
    NysiisStep step;
    if (!out)
      return step;
    std::size_t count =
        add(1, add(stage.size(), add(rule.size(), add(before.size(), token.size()))));
    admit(count);
    step.stage    = stage;
    step.rule     = rule;
    step.begin    = first;
    step.end      = end;
    step.before   = before;
    step.token    = token;
    step.appended = appended;
    return step;
  };
  /// @brief Finish an actual admitted record with its actual producer state, before retaining the after copy.
  auto finish = [&](NysiisStep step, const std::string& after) {
    if (!out)
      return;
    admit(after.size());
    step.after = after;
    out->steps.push_back(std::move(step));
  };
  std::string_view prefix   = "none", replacement;
  std::size_t      consumed = 0;
  if (s.starts_with("MAC")) {
    prefix      = "MAC";
    replacement = "MCC";
    consumed    = 3;
  } else if (s.starts_with("KN")) {
    prefix      = "KN";
    replacement = "N";
    consumed    = 2;
  } else if (s.starts_with("K")) {
    prefix      = "K";
    replacement = "C";
    consumed    = 1;
  } else if (s.starts_with("PH")) {
    prefix      = "PH";
    replacement = "FF";
    consumed    = 2;
  } else if (s.starts_with("PF")) {
    prefix      = "PF";
    replacement = "FF";
    consumed    = 2;
  } else if (s.starts_with("SCH")) {
    prefix      = "SCH";
    replacement = "SSS";
    consumed    = 3;
  }
  auto prefix_record = begin("prefix", prefix, 0, 0, s, replacement, false);
  if (consumed)
    s = std::string(replacement) + s.substr(consumed);
  finish(std::move(prefix_record), s);
  std::string_view suffix       = "none", suffix_token;
  std::size_t      suffix_bytes = 0;
  for (auto ending : {"IE", "EE"})
    if (s.ends_with(ending)) {
      suffix       = ending;
      suffix_token = "Y";
      suffix_bytes = 2;
      break;
    }
  if (!suffix_bytes)
    for (auto ending : {"DT", "RT", "RD", "NT", "ND"})
      if (s.ends_with(ending)) {
        suffix       = ending;
        suffix_token = "D";
        suffix_bytes = 2;
        break;
      }
  auto suffix_record = begin("suffix", suffix, 0, 0, s, suffix_token, false);
  if (suffix_bytes)
    s = s.substr(0, s.size() - suffix_bytes) + std::string(suffix_token);
  finish(std::move(suffix_record), s);
  if (out) {
    admit(s.size());
    out->rewritten = s;
  }
  std::string key;
  auto first_record = begin("first", "first-byte", 0, 1, key, std::string_view(s.data(), 1), true);
  key.assign(1, s[0]);
  finish(std::move(first_record), key);
  for (std::size_t i = 1; i < s.size(); ++i) {
    const auto       first = i;
    char             c = s[i], next = i + 1 < s.size() ? s[i + 1] : '\0';
    std::string      token(1, c);
    std::string_view rule = "literal";
    if (c == 'E' && next == 'V') {
      token = "AF";
      rule  = "EV";
      ++i;
    } else if (vowel(c)) {
      token = "A";
      rule  = "vowel";
    } else if (c == 'Q') {
      token = "G";
      rule  = "Q";
    } else if (c == 'Z') {
      token = "S";
      rule  = "Z";
    } else if (c == 'M') {
      token = "N";
      rule  = "M";
    } else if (c == 'K') {
      token = next == 'N' ? "N" : "C";
      rule  = next == 'N' ? "KN" : "K";
    } else if (c == 'S' && next == 'C' && i + 2 < s.size() && s[i + 2] == 'H') {
      token = "SS";
      rule  = "SCH";
      i += 2;
    } else if (c == 'P' && next == 'H') {
      token = "F";
      rule  = "PH";
      ++i;
    } else if (
        c == 'H' &&
        (!vowel(s[i - 1]) || (i + 1 < s.size() && !vowel(s[i + 1])) || i + 1 == s.size())) {
      token = vowel(s[i - 1]) ? "A" : std::string(1, s[i - 1]);
      rule  = "H-neighbor";
    } else if (c == 'W' && vowel(s[i - 1])) {
      token = std::string(1, s[i - 1]);
      rule  = "W-previous";
    }
    const bool appended = token.back() != key.back();
    ++result.interior_tokens;
    if (!appended)
      ++result.suppressed_tokens;
    auto step = begin("interior", rule, first, i + 1, key, token, appended);
    if (appended)
      key += token;
    finish(std::move(step), key);
  }
  auto trim_s = begin(
      "cleanup", key.size() > 1 && key.back() == 'S' ? "trim-S" : "none", 0, 0, key, "", false);
  if (key.size() > 1 && key.back() == 'S')
    key.pop_back();
  finish(std::move(trim_s), key);
  auto fold = begin(
      "cleanup", key.size() >= 2 && key.ends_with("AY") ? "fold-AY" : "none", 0, 0, key, "", false);
  if (key.size() >= 2 && key.ends_with("AY"))
    key = key.substr(0, key.size() - 2) + "Y";
  finish(std::move(fold), key);
  auto trim_a = begin(
      "cleanup", key.size() > 1 && key.back() == 'A' ? "trim-A" : "none", 0, 0, key, "", false);
  if (key.size() > 1 && key.back() == 'A')
    key.pop_back();
  finish(std::move(trim_a), key);
  if (out)
    admit(key.size());
  result.key = std::move(key);
  return result;
}
} // namespace
std::string nysiis(const std::string& s) {
  return run(s, nullptr, 0).key;
}
NysiisAnalysis nysiis_analyze(const std::string& s) {
  return run(s, nullptr, 0);
}
NysiisTrace nysiis_trace(const std::string& s, std::size_t cap) {
  NysiisTrace t;
  t.analysis = run(s, &t, cap);
  return t;
}
NysiisComparison nysiis_compare_names(const std::vector<std::string>& names, std::size_t cap) {
  if (cap && names.size() > cap)
    throw std::length_error("NYSIIS cohort: row cap");
  NysiisComparison c;
  c.names = names;
  std::map<std::string, std::uint64_t> sx, ny;
  /// @brief Count equal unordered prior-row pairs before incrementing an actual key frequency.
  auto count = [&](auto& frequencies, const std::string& key, std::uint64_t& pairs) {
    auto& n = frequencies[key];
    if (n > std::numeric_limits<std::uint64_t>::max() - pairs ||
        n == std::numeric_limits<std::uint64_t>::max())
      throw std::length_error("NYSIIS cohort: pair geometry");
    pairs += n;
    ++n;
  };
  for (auto& name : names) {
    auto a = soundex(name), b = nysiis(name);
    count(sx, a, c.equal_soundex_pairs);
    count(ny, b, c.equal_nysiis_pairs);
    c.soundex_keys.push_back(std::move(a));
    c.nysiis_keys.push_back(std::move(b));
  }
  c.distinct_soundex = sx.size();
  c.distinct_nysiis  = ny.size();
  return c;
}
std::vector<std::string> nysiis_demo_names() {
  return {"Anderson", "Andrews",  "Ashcraft",   "Ashcroft", "Baker",  "Barker",   "Brown",
          "Braun",    "Carter",   "Cartwright", "Davidson", "Davis",  "Edwards",  "Evans",
          "Fisher",   "Fischer",  "Gauss",      "Gaus",     "Harris", "Harrison", "Jackson",
          "Johnson",  "Johnston", "Kelly",      "Kelley",   "Miller", "Mueller",  "Nelson",
          "Nielsen",  "Peterson", "Pearson",    "Robinson"};
}
std::vector<std::string> nysiis_demo_examples() {
  return {"MacDonald", "Catherine", "Ashcraft", "Ashcroft", "Mueller", "Miller"};
}
} // namespace datamunge::algorithms
