#pragma once

/// @file text_formatter.hpp
/// @brief Shared building blocks for the library's std::format / std::print support.
///
/// Every library type that prints is given a `std::formatter` specialization
/// that derives from `datamunge::format_detail::text_formatter<T>` and supplies
/// one static function, `text(const T&)`, returning the printed text:
///
/// @code
/// template <>
/// struct std::formatter<datamunge::geometry::Point2D>
///     : datamunge::format_detail::text_formatter<datamunge::geometry::Point2D> {
///   static std::string text(const datamunge::geometry::Point2D& p) {
///     return std::format("Point2D({}, {})", p.x, p.y);
///   }
/// };
/// @endcode
///
/// The base handles the format spec: it accepts everything a string accepts
/// (fill, alignment, width, precision-as-truncation), so `{:>40}` pads the
/// whole text like it would a std::string.

#include <cstddef>
#include <format>
#include <iterator>
#include <string>
#include <string_view>

/// @brief Core library APIs and their algorithm, data, and numerical submodules.
namespace datamunge {

/// @brief Implementation helpers for the std::formatter specializations; not part of the public API.
namespace format_detail {

/// @brief Base for library formatters: parses a string format spec and prints `std::formatter<T>::text(value)`.
///
/// Wraps (rather than inherits) the string formatter so it does not expose
/// `set_debug_format()`: the standard range formatter calls that when present,
/// which would quote every element of a `std::vector<LibraryType>`.
/// @tparam T The library type being formatted; its `std::formatter<T>` must define `static std::string text(const T&)`.
template <typename T>
struct text_formatter {
  /// @brief Parses a string format spec (fill, alignment, width, precision).
  /// @tparam ParseContext A standard format parse context.
  /// @param context The spec to parse, positioned after the `:`.
  /// @return The iterator at the spec's closing `}`.
  template <typename ParseContext>
  constexpr auto parse(ParseContext& context) {
    return text_.parse(context);
  }

  /// @brief Writes the value's text, honouring fill, alignment and width.
  /// @tparam FormatContext A standard output formatting context.
  /// @param value The library value to print.
  /// @param context The output context the text is written to.
  /// @return The output iterator just past the written text.
  template <typename FormatContext>
  auto format(const T& value, FormatContext& context) const {
    const std::string text = std::formatter<T, char>::text(value);
    return text_.format(std::string_view(text), context);
  }

 private:
  /// @brief The string formatter that applies the parsed spec to the text.
  std::formatter<std::string_view, char> text_;
};

/// @brief Formats a range's elements as "[a, b, c]", eliding the middle past `max_items` ("[a, b, ..., z] (n=100)").
/// @tparam Range A sized range whose elements are std::formattable.
/// @param values The elements to show.
/// @param max_items How many elements to show before eliding (0 shows all).
/// @return The bracketed preview.
///
/// @par Example
/// @code
/// std::print("{}\n", datamunge::format_detail::preview(std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8}));
/// // [1, 2, 3, 4, 5, ..., 8] (n=8)
/// @endcode
template <typename Range>
std::string preview(const Range& values, std::size_t max_items = 6) {
  const auto count = static_cast<std::size_t>(std::ranges::distance(values));
  std::string out = "[";
  std::size_t i = 0;
  const bool elide = max_items != 0 && count > max_items;
  const std::size_t head = elide ? max_items - 1 : count;
  for (const auto& value : values) {
    if (i < head || (elide && i == count - 1)) {
      if (i > 0) out += ", ";
      if (elide && i == count - 1) out += "..., ";
      out += std::format("{}", value);
    }
    i++;
  }
  out += "]";
  if (elide) out += std::format(" (n={})", count);
  return out;
}

}  // namespace format_detail
}  // namespace datamunge
