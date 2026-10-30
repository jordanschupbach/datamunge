#pragma once
/// @file text.hpp
/// @brief Private scalar UTF-8 and bounded native scene-name validation.
#include <stdexcept>
#include <string_view>
/// @brief Private text and scene serialization helpers.
namespace datamunge::game::detail {
/// @brief Validate scalar UTF-8, rejecting overlong encodings and surrogate code points.
/// @param text Raw or decoded UTF-8 bytes.
inline void utf8(std::string_view text) {
  for (std::size_t i = 0; i < text.size();) {
    const auto first = static_cast<unsigned char>(text[i++]);
    if (first < 128)
      continue;
    unsigned    code;
    std::size_t count;
    if (first >= 0xc2 && first <= 0xdf) {
      code  = first & 31;
      count = 1;
    } else if (first >= 0xe0 && first <= 0xef) {
      code  = first & 15;
      count = 2;
    } else if (first >= 0xf0 && first <= 0xf4) {
      code  = first & 7;
      count = 3;
    } else
      throw std::invalid_argument("invalid UTF-8 in scene string");
    if (count > text.size() - i)
      throw std::invalid_argument("truncated UTF-8 in scene string");
    for (std::size_t j = 0; j < count; ++j) {
      const auto c = static_cast<unsigned char>(text[i++]);
      if ((c & 0xc0) != 0x80)
        throw std::invalid_argument("invalid UTF-8 continuation");
      code = (code << 6) | (c & 63);
    }
    if ((count == 1 && code < 128) || (count == 2 && code < 2048) || (count == 3 && code < 65536) ||
        code > 0x10ffff || (code >= 0xd800 && code <= 0xdfff))
      throw std::invalid_argument("invalid UTF-8 scalar");
  }
}

/// @brief Validate bounded scalar UTF-8 authoring and factory names.
/// @param name UTF-8 text, at most 128 bytes and without NUL.
/// @param allow_empty Whether the empty string denotes no optional behavior.
inline void validate_name(std::string_view name, bool allow_empty = false) {
  if ((name.empty() && !allow_empty) || name.size() > 128 ||
      name.find('\0') != std::string_view::npos)
    throw std::invalid_argument("game names require 1..128 UTF-8 bytes without NUL");
  utf8(name);
}
} // namespace datamunge::game::detail
