#pragma once

/// @file tex.hpp
/// @brief std::format/std::print support for owned TeX fragments.

#include <datamunge/tex.hpp>
#include <format>
#include <string_view>

/// @brief Format a Tex fragment as raw text, with standard string alignment and precision.
/// Delimiters are supplied explicitly by Tex::display() or Tex::inline_math().
/// @par Example
/// @code
/// std::print("{}\n", dm::Tex("x+y").underbrace("s"));
/// @endcode
template<>
struct std::formatter<datamunge::Tex, char> : std::formatter<std::string_view, char> {
  /// @brief Write raw TeX using the parsed standard string format specification.
  /// @tparam Context Standard output-format context.
  /// @param value Fragment to format without mutation.
  /// @param context Destination format context.
  /// @return Iterator past the formatted bytes.
  template<class Context>
  auto format(const datamunge::Tex& value, Context& context) const {
    return std::formatter<std::string_view, char>::format(value.str(), context);
  }
};
