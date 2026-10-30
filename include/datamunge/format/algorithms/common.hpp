#pragma once

/// @file common.hpp
/// @brief Shared value-to-text helpers for the datamunge::algorithms std::formatter specializations.
/// Not part of the public API; included by <datamunge/format/algorithms.hpp>.

#include <datamunge/format/text_formatter.hpp>

#include <algorithm>
#include <cstddef>
#include <format>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge::format_detail {

/// @brief Helpers that turn the public fields of datamunge::algorithms types into short, readable text.
namespace algorithms_fmt {

/// @brief Longest string field printed before it is cut with "...".
inline constexpr std::size_t max_string_chars = 60;

/// @brief Elements shown per nesting level of a range before eliding the middle.
inline constexpr std::size_t max_items = 6;

/// @brief Elements shown when a range holds records (structs), which print long.
inline constexpr std::size_t max_record_items = 3;

/// @brief True for std::function specializations, which print as "<function>".
/// @tparam T The candidate type.
template <typename T>
struct is_function_object : std::false_type {};

/// @brief Matches std::function<Signature>.
/// @tparam Signature The wrapped call signature.
template <typename Signature>
struct is_function_object<std::function<Signature>> : std::true_type {};

/// @brief True for std::optional specializations.
/// @tparam T The candidate type.
template <typename T>
struct is_optional : std::false_type {};

/// @brief Matches std::optional<T>.
/// @tparam T The contained type.
template <typename T>
struct is_optional<std::optional<T>> : std::true_type {};

/// @brief Satisfied by tuple-like types (std::pair, std::tuple; std::array is treated as a range).
/// @tparam T The candidate type.
template <typename T>
concept tuple_like = !std::ranges::range<T> && requires { std::tuple_size<T>::value; };

/// @brief Satisfied by associative containers (std::map, std::unordered_map, ...).
/// @tparam T The candidate type.
template <typename T>
concept map_like = std::ranges::range<T> && requires {
  typename T::key_type;
  typename T::mapped_type;
};

/// @brief Satisfied by set-like containers (std::set, std::unordered_set, ...).
/// @tparam T The candidate type.
template <typename T>
concept set_like = std::ranges::range<T> && !map_like<T> && requires { typename T::key_type; };

/// @brief Satisfied by unordered containers, whose iteration order is unspecified (their entries are sorted).
/// @tparam T The candidate type.
template <typename T>
concept unordered_container = requires { typename T::hasher; };

/// @brief Prints a signed 128-bit integer in decimal (std::format has no __int128 support).
/// @param v The value.
/// @return The decimal digits, with a leading '-' when negative.
inline std::string int128_text(__int128 v) {
  if (v == 0) return "0";
  const bool negative = v < 0;
  unsigned __int128 u = negative ? static_cast<unsigned __int128>(-(v + 1)) + 1 : static_cast<unsigned __int128>(v);
  std::string digits;
  while (u != 0) {
    digits.push_back(static_cast<char>('0' + static_cast<int>(u % 10)));
    u /= 10;
  }
  if (negative) digits.push_back('-');
  std::reverse(digits.begin(), digits.end());
  return digits;
}

/// @brief Quotes a string, truncating it past max_string_chars.
/// @param s The string.
/// @return The text wrapped in double quotes.
inline std::string quoted(std::string_view s) {
  if (s.size() > max_string_chars) return std::format("\"{}...\" (n={})", s.substr(0, max_string_chars - 3), s.size());
  return std::format("\"{}\"", s);
}

/// @brief Joins pre-rendered items as "[a, b, ..., z] (n=N)", eliding the middle past `cap` items.
/// @param items The rendered elements.
/// @param open The opening bracket.
/// @param close The closing bracket.
/// @param cap How many elements to show before eliding.
/// @return The bracketed list.
inline std::string join_items(const std::vector<std::string>& items, std::string_view open, std::string_view close,
                              std::size_t cap) {
  std::string out(open);
  const std::size_t n = items.size();
  const bool elide = n > cap;
  for (std::size_t i = 0; i < n; ++i) {
    if (elide && i >= cap - 1 && i != n - 1) continue;
    if (i > 0) out += ", ";
    if (elide && i == n - 1) out += "..., ";
    out += items[i];
  }
  out += close;
  if (elide) out += std::format(" (n={})", n);
  return out;
}

/// @brief Forward declaration of value() so nested containers and tuples can recurse into it.
/// @tparam T The value's type.
/// @param v The value.
/// @return Its text.
template <typename T>
std::string value(const T& v);

/// @brief Renders any value a datamunge::algorithms record can hold: numbers, strings, chars, enums,
///        optionals, pairs/tuples, ranges (nested ranges recurse), maps, sets and other formattable types.
/// @tparam T The value's type.
/// @param v The value.
/// @return Its text; types with no text print as "<?>".
template <typename T>
std::string value(const T& v) {
  using U = std::remove_cvref_t<T>;
  if constexpr (std::is_same_v<U, bool>) {
    return v ? "true" : "false";
  } else if constexpr (std::is_same_v<U, char>) {
    const auto c = static_cast<unsigned char>(v);
    if (c >= 0x20 && c < 0x7f) return std::format("'{}'", v);
    return std::format("{}", static_cast<int>(c));
  } else if constexpr (std::is_same_v<U, __int128>) {
    return int128_text(v);
  } else if constexpr (std::is_arithmetic_v<U>) {
    return std::format("{}", v);
  } else if constexpr (std::is_convertible_v<const U&, std::string_view>) {
    return quoted(std::string_view(v));
  } else if constexpr (std::is_enum_v<U>) {
    if constexpr (std::formattable<U, char>) return std::format("{}", v);
    else return std::format("{}", static_cast<std::underlying_type_t<U>>(v));
  } else if constexpr (is_function_object<U>::value) {
    return v ? "<function>" : "<empty function>";
  } else if constexpr (is_optional<U>::value) {
    return v ? value(*v) : std::string("none");
  } else if constexpr (tuple_like<U>) {
    return std::apply(
        [](const auto&... parts) {
          std::vector<std::string> items{value(parts)...};
          return join_items(items, "(", ")", items.size());
        },
        v);
  } else if constexpr (map_like<U>) {
    std::vector<std::string> items;
    for (const auto& [key, mapped] : v) items.push_back(value(key) + ": " + value(mapped));
    if constexpr (unordered_container<U>) std::sort(items.begin(), items.end());
    return join_items(items, "{", "}", max_items);
  } else if constexpr (std::ranges::range<U>) {
    using E = std::remove_cvref_t<std::ranges::range_value_t<U>>;
    std::vector<std::string> items;
    for (const auto& e : v) items.push_back(value(e));
    if constexpr (unordered_container<U>) std::sort(items.begin(), items.end());
    const bool records = std::is_class_v<E> && !std::ranges::range<E> && !tuple_like<E> &&
                         !std::is_convertible_v<const E&, std::string_view>;
    const std::size_t cap = records ? max_record_items : max_items;
    return set_like<U> ? join_items(items, "{", "}", cap) : join_items(items, "[", "]", cap);
  } else if constexpr (std::formattable<U, char>) {
    return std::format("{}", v);
  } else {
    return "<?>";
  }
}

/// @brief One rendered "name=value" field of a record.
using field = std::pair<std::string_view, std::string>;

/// @brief Prints a record as "Name(a=1, b=[2, 3])".
/// @param name The short type name.
/// @param fields The rendered fields, in declaration order.
/// @return The one-line text.
inline std::string record(std::string_view name, std::initializer_list<field> fields) {
  std::string out(name);
  out += "(";
  bool first = true;
  for (const auto& [key, text] : fields) {
    if (!first) out += ", ";
    first = false;
    out += key;
    out += "=";
    out += text;
  }
  out += ")";
  return out;
}

}  // namespace algorithms_fmt
}  // namespace datamunge::format_detail
