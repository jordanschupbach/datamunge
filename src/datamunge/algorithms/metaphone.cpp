/// @file metaphone.cpp
/// @brief Canonical metaphone API, producing histories and checked native contracts.

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <datamunge/algorithms/metaphone.hpp>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical metaphone production and capture.
namespace {
/// @brief Shared complete logical record/character admission for one actual capture.
struct Budget {
  std::size_t count{0}; ///< Actual admitted count.
  std::size_t limit{0}; ///< Optional cap, zero off.

  /// @brief Admit actual next capture contributions before copies.
  /// @param n Additional record/character count.
  /// @throws std::length_error On overflow or exceeded complete optional cap.
  void admit(std::size_t n) {
    if (n > SIZE_MAX - count || (limit && (count > limit || n > limit - count)))
      throw std::length_error("Metaphone complete capture");
    count += n;
  }
};
/// @brief Safely lowercase one unsigned byte under the current C locale.
/// @param c Original byte.
/// @return Single lowercase byte.
char lo(char c) {
  return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
}
/// @brief Safely uppercase every actual output byte under the current C locale.
/// @param input Actual working key.
/// @return Same-length uppercase key.
std::string upper(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (unsigned char c : input)
    out.push_back(static_cast<char>(std::toupper(c)));
  return out;
}
/// @brief Capture actual already-produced branch spans after complete record/suffix admission.
/// @param trace Owned capture target.
/// @param budget Complete logical admission.
/// @param begin Actual normalized source start.
/// @param end Actual normalized past-last consumed source.
/// @param c Actual current byte.
/// @param nx Actual first lookahead value.
/// @param nnx Actual second lookahead value.
/// @param has_next Actual first lookahead position exists.
/// @param has_next2 Actual second lookahead position exists.
/// @param rule Actual entered top-level branch.
/// @param out Actual working output after branch.
/// @param before Actual pre-branch output length.
/// @throws std::length_error On complete actual capture admission failure.
void record(
    MetaphoneTrace&    trace,
    Budget&            budget,
    std::size_t        begin,
    std::size_t        end,
    char               c,
    char               nx,
    char               nnx,
    bool               has_next,
    bool               has_next2,
    const char*        rule,
    const std::string& out,
    std::size_t        before) {
  budget.admit(1);
  budget.admit(out.size() - before);
  trace.steps.push_back(
      {begin,
       end,
       static_cast<unsigned char>(c),
       static_cast<unsigned char>(nx),
       static_cast<unsigned char>(nnx),
       has_next,
       has_next2,
       rule,
       out.substr(before),
       before,
       out.size()});
}
/// @brief Canonical retained native context-rule encoder and optional actual producer capture.
/// @param name Original arbitrary bytes.
/// @param trace Optional owned actual history.
/// @param budget Shared complete capture admission, nonnull exactly when capturing.
/// @return Actual normalized input, key and selected work.
/// @throws std::length_error On complete actual capture admission failure.
MetaphoneAnalysis produce(const std::string& name, MetaphoneTrace* trace, Budget* budget) {
  MetaphoneAnalysis analysis;
  if (trace) {
    budget->admit(name.size());
    trace->input = name;
  }
  std::string s(name.size(), '\0');
  std::transform(name.begin(), name.end(), s.begin(), lo);

  // drop the first letter of a silent-onset cluster
  /// @brief Actual recognized two-byte initial silent-onset clusters in testing order.
  static const char* onsets[] = {"kn", "gn", "pn", "wr", "ae"};
  for (const char* o : onsets) {
    if (s.rfind(o, 0) == 0) {
      s                      = s.substr(1);
      analysis.onset_removed = 1;
      break;
    }
  }

  std::string       out;
  const std::size_t n = s.size();
  if (trace)
    budget->admit(n);
  /// @brief Test actual non-NUL byte membership in a fixed lowercase rule set.
  /// @param set NUL-terminated lowercase branch letters.
  /// @param ch Actual context byte, zero never matches.
  /// @return Whether ch occurs in set.
  auto lower_in = [](const char* set, char ch) {
    return ch != '\0' && std::strchr(set, ch) != nullptr;
  };

  for (std::size_t i = 0; i < n; ++i) {
    const auto  start = i, before = out.size();
    const char* rule = "unhandled byte";
    const char  c    = s[i];
    const char  nx   = (i + 1 < n) ? s[i + 1] : '\0';
    const char  nnx  = (i + 2 < n) ? s[i + 2] : '\0';

    // skip doubled letters, except "cc"
    if (c == nx && c != 'c') {
      ++analysis.iterations;
      ++analysis.consumed;
      if (trace)
        record(
            *trace,
            *budget,
            start,
            i + 1,
            c,
            nx,
            nnx,
            i + 1 < n,
            i + 2 < n,
            "duplicate",
            out,
            before);
      continue;
    }

    if (lower_in("aeiou", c)) {
      rule = "vowel context";
      if (i == 0 || s[i - 1] == ' ')
        out.push_back(c);
    } else if (c == 'b') {
      rule = "letter b context";
      // silent b in a word-final "mb" (comb, dumb) -- but kept mid-word (amber, climber)
      if (!(i != 0 && s[i - 1] == 'm' && nx == '\0'))
        out.push_back('b');
    } else if (c == 'c') {
      rule = "letter c context";
      if ((nx == 'i' && nnx == 'a') || nx == 'h') {
        out.push_back('x');
        ++i;
      } else if (lower_in("iey", nx)) {
        out.push_back('s');
        ++i;
      } else {
        out.push_back('k');
      }
    } else if (c == 'd') {
      rule = "letter d context";
      if (nx == 'g' && lower_in("iey", nnx)) {
        out.push_back('j');
        i += 2;
      } else {
        out.push_back('t');
      }
    } else if (lower_in("fjlmnr", c)) {
      rule = "direct consonant";
      out.push_back(c);
    } else if (c == 'g') {
      rule = "letter g context";
      if (lower_in("iey", nx)) {
        out.push_back('j');
      } else if (nx == 'h' && nnx != '\0' && !lower_in("aeiou", nnx)) {
        ++i; // silent gh before a consonant (light, night, eight)
      } else if (nx == 'n' && nnx == '\0') {
        ++i; // silent gn ending a word (sign, align, reign) -- consumes the n
      } else {
        out.push_back('k');
      }
    } else if (c == 'h') {
      rule = "letter h context";
      if (i == 0 || lower_in("aeiou", nx) || !lower_in("aeiou", s[i - 1]))
        out.push_back('h');
    } else if (c == 'k') {
      rule = "letter k context";
      if (i == 0 || s[i - 1] != 'c')
        out.push_back('k');
    } else if (c == 'p') {
      rule = "letter p context";
      if (nx == 'h') {
        out.push_back('f');
        ++i;
      } else {
        out.push_back('p');
      }
    } else if (c == 'q') {
      rule = "letter q context";
      out.push_back('k');
    } else if (c == 's') {
      rule = "letter s context";
      if (nx == 'h') {
        out.push_back('x');
        ++i;
      } else if (nx == 'i' && (nnx == 'o' || nnx == 'a')) {
        out.push_back('x');
        i += 2;
      } else {
        out.push_back('s');
      }
    } else if (c == 't') {
      rule = "letter t context";
      if (nx == 'i' && (nnx == 'o' || nnx == 'a')) {
        out.push_back('x');
      } else if (nx == 'h') {
        out.push_back('0');
        ++i;
      } else if (nx != 'c' || nnx != 'h') {
        out.push_back('t');
      }
    } else if (c == 'v') {
      rule = "letter v context";
      out.push_back('f');
    } else if (c == 'w') {
      rule = "letter w context";
      if (i == 0 && nx == 'h') {
        ++i;
        out.push_back('w');
      } else if (lower_in("aeiou", nx)) {
        out.push_back('w');
      }
    } else if (c == 'x') {
      rule = "letter x context";
      if (i == 0) {
        if (nx == 'h' || (nx == 'i' && (nnx == 'o' || nnx == 'a')))
          out.push_back('x');
        else
          out.push_back('s');
      } else {
        out.push_back('k');
        out.push_back('s');
      }
    } else if (c == 'y') {
      rule = "letter y context";
      if (lower_in("aeiou", nx))
        out.push_back('y');
    } else if (c == 'z') {
      rule = "letter z context";
      out.push_back('s');
    } else if (c == ' ') {
      rule = "space context";
      if (!out.empty() && out.back() != ' ')
        out.push_back(' ');
    }
    ++analysis.iterations;
    analysis.consumed += i + 1 - start;
    if (trace)
      record(
          *trace,
          *budget,
          start,
          i + 1,
          c,
          nx,
          nnx,
          start + 1 < n,
          start + 2 < n,
          rule,
          out,
          before);
  }
  if (trace)
    budget->admit(out.size());
  analysis.key        = upper(out);
  analysis.normalized = std::move(s);
  analysis.emitted    = out.size();
  return analysis;
}

} // namespace
std::string metaphone(const std::string& name) {
  return produce(name, nullptr, nullptr).key;
}
MetaphoneAnalysis metaphone_analyze(const std::string& name) {
  return produce(name, nullptr, nullptr);
}
MetaphoneTrace metaphone_trace(const std::string& name, std::size_t limit) {
  Budget         budget{0, limit};
  MetaphoneTrace out;
  out.analysis         = produce(name, &out, &budget);
  out.captured_entries = budget.count;
  return out;
}
std::vector<std::string> metaphone_demo_words() {
  return {
      "Knight",
      "Wright",
      "Thompson",
      "Xavier",
      "sign",
      "eight",
      "comb",
      "Pneumonia",
      "psychology",
      "Wednesday",
      "Gnome",
      "phlegm"};
}
} // namespace datamunge::algorithms
