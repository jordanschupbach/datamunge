/// @file soundex.cpp
/// @brief One invocation-local ASCII Soundex producer and actual branch capture.
#include <datamunge/algorithms/soundex.hpp>
#include <limits>
#include <stdexcept>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked producing helpers.
namespace {
/// @brief Uppercase ASCII letters only, without locale or Unicode assumptions.
/// @param c Actual input byte.
/// @return ASCII uppercase or unchanged byte.
char upper(unsigned char c) {
  return c >= 'a' && c <= 'z' ? static_cast<char>(c - 'a' + 'A') : static_cast<char>(c);
}
/// @brief Actual American-Soundex consonant group.
/// @param c Actual uppercase byte.
/// @return Digit class or0 for uncoded.
char group(char c) {
  switch (c) {
    case 'B':
    case 'F':
    case 'P':
    case 'V':
      return '1';
    case 'C':
    case 'G':
    case 'J':
    case 'K':
    case 'Q':
    case 'S':
    case 'X':
    case 'Z':
      return '2';
    case 'D':
    case 'T':
      return '3';
    case 'L':
      return '4';
    case 'M':
    case 'N':
      return '5';
    case 'R':
      return '6';
    default:
      return '0';
  }
}
/// @brief Encode actual prefix transitions once.
/// @param name Raw bytes.
/// @param o Policies.
/// @param capture Retain transitions.
/// @return Actual key and work.
SoundexRun run(const std::string& name, const SoundexOptions& o, bool capture) {
  if (o.max_bytes && name.size() > o.max_bytes)
    throw std::length_error("soundex: bytes");
  SoundexRun r;
  if (name.empty())
    return r;
  std::size_t retained = 0;
  /// @brief Retain the actual current producing transition with checked copy geometry.
  auto record = [&](std::size_t        offset,
                    unsigned char      raw,
                    char               c,
                    char               digit,
                    char               previous,
                    const std::string& action) {
    if (capture) {
      for (const auto extra : {std::size_t{1}, r.code.size(), action.size()}) {
        if (extra > std::numeric_limits<std::size_t>::max() - retained)
          throw std::length_error("soundex: history capacity");
        retained += extra;
        if (o.max_history_values && retained > o.max_history_values)
          throw std::length_error("soundex: history policy");
      }
      r.history.push_back({offset, raw, c, digit, previous, action, r.code});
    }
  };
  const char first = upper(static_cast<unsigned char>(name[0]));
  r.code           = std::string(1, first);
  char last        = group(first);
  r.scanned_bytes  = 1;
  record(0, static_cast<unsigned char>(name[0]), first, last, '0', "retain-first");
  for (std::size_t i = 1; i < name.size() && r.code.size() < 4; ++i) {
    const auto  raw = static_cast<unsigned char>(name[i]);
    const char  c = upper(raw), digit = group(c), previous = last;
    std::string action;
    if (digit != '0') {
      if (digit != last) {
        r.code += digit;
        ++r.emitted_digits;
        action = "append";
      } else
        action = "suppress";
      last = digit;
    } else if (c == 'H' || c == 'W')
      action = "transparent";
    else {
      last   = '0';
      action = "reset";
    }
    ++r.scanned_bytes;
    record(i, raw, c, digit, previous, action);
  }
  r.padding = 4 - r.code.size();
  r.code.append(r.padding, '0');
  record(r.scanned_bytes, 0, 0, '0', last, "pad");
  return r;
}
} // namespace
SoundexRun soundex_run(const std::string& name, const SoundexOptions& o) {
  return run(name, o, false);
}
SoundexRun soundex_trace(const std::string& name, const SoundexOptions& o) {
  return run(name, o, true);
}
std::string soundex(const std::string& name) {
  SoundexOptions o;
  o.max_bytes = 0;
  return soundex_run(name, o).code;
}
std::vector<std::string> soundex_example_names() {
  return {"Robert",    "Rupert",  "Rubin",   "Smith",    "Smyth",    "Smythe",  "Catherine",
          "Katherine", "Kathryn", "Johnson", "Johnston", "Gauss",    "Gaus",    "Sean",
          "Shawn",     "Shaun",   "Lee",     "Leigh",    "Ashcraft", "Ashcroft"};
}
std::map<std::string, std::vector<std::string>> soundex_groups(
    const std::vector<std::string>& names) {
  if (names.size() > 1000000)
    throw std::length_error("soundex: rows");
  std::map<std::string, std::vector<std::string>> groups;
  for (const auto& name : names)
    groups[soundex(name)].push_back(name);
  return groups;
}
} // namespace datamunge::algorithms
