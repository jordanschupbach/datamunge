#pragma once
/// @file json.hpp
/// @brief Private bounded RFC 8259 scene JSON parser with exact integer tokens.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "text.hpp"
namespace datamunge::game::detail {
/// @brief Minimal owned JSON value used only during scene import.
struct Json {
  /// @brief JSON grammar alternatives; numbers retain their original exact tokens.
  enum class Kind {
    Null,    ///< JSON null literal.
    Boolean, ///< Exact true/false literal.
    Number,  ///< Exact lexical number token.
    String,  ///< Decoded scalar UTF-8 string.
    Array,   ///< Ordered sequence of values.
    Object   ///< Duplicate-free string-keyed members.
  };
  Kind                        kind{Kind::Null}; ///< Active grammar alternative.
  bool                        boolean{false};   ///< Boolean payload.
  std::string                 text;             ///< String payload or exact numeric token.
  std::vector<Json>           array;            ///< Ordered array payload.
  std::map<std::string, Json> object;           ///< Duplicate-free object payload.

  /// @brief Require a particular JSON alternative before conversion.
  /// @param expected Required alternative.
  void require(Kind expected) const {
    if (kind != expected)
      throw std::invalid_argument("unexpected scene JSON type");
  }
  /// @brief Convert a numeric token to a finite native double.
  /// @return Finite numeric value.
  double number() const {
    require(Kind::Number);
    double     result;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() ||
        !std::isfinite(result))
      throw std::invalid_argument("scene number outside finite range");
    return result;
  }
  /// @brief Convert an exact nonnegative integer token without floating-point rounding.
  /// @return Unsigned 64-bit integer.
  std::uint64_t integer() const {
    require(Kind::Number);
    std::uint64_t result;
    const auto    parsed = std::from_chars(text.data(), text.data() + text.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
      throw std::invalid_argument("scene ID must be an exact nonnegative integer token");
    return result;
  }
  /// @brief Borrow a required object member.
  /// @param key Required key.
  /// @return Reference retained by this value.
  const Json& at(const std::string& key) const {
    require(Kind::Object);
    const auto it = object.find(key);
    if (it == object.end())
      throw std::invalid_argument("missing scene key: " + key);
    return it->second;
  }
  /// @brief Look up an optional object member without conversion.
  /// @param key Optional key.
  /// @return Borrowed member or null.
  const Json* find(const std::string& key) const {
    require(Kind::Object);
    const auto it = object.find(key);
    return it == object.end() ? nullptr : &it->second;
  }
  /// @brief Reject object members outside a schema's allowed set.
  /// @param allowed Accepted exact UTF-8 keys.
  void keys(std::initializer_list<std::string_view> allowed) const {
    require(Kind::Object);
    for (const auto& item : object)
      if (std::find(allowed.begin(), allowed.end(), item.first) == allowed.end())
        throw std::invalid_argument("unknown scene key: " + item.first);
  }
};
/// @brief Bounded recursive-descent reader; at most 64 nested containers and 1M values.
class JsonParser {
  std::string_view source_;    ///< Borrowed complete bounded document.
  std::size_t      offset_{0}; ///< Next unread byte.
  std::size_t      nodes_{0};  ///< Successfully started JSON values.

  /// @brief Throw a syntax error with a byte offset.
  [[noreturn]] void fail() const {
    throw std::invalid_argument("invalid scene JSON at byte " + std::to_string(offset_));
  }
  /// @brief Consume only the four whitespace bytes permitted by JSON.
  void space() {
    while (offset_ < source_.size() && (source_[offset_] == ' ' || source_[offset_] == '\n' ||
                                        source_[offset_] == '\r' || source_[offset_] == '\t'))
      ++offset_;
  }
  /// @brief Consume one exact byte when present.
  /// @param c Expected byte.
  /// @return Whether it was consumed.
  bool take(char c) {
    if (offset_ < source_.size() && source_[offset_] == c) {
      ++offset_;
      return true;
    }
    return false;
  }
  /// @brief Read one escaped UTF-16 code unit from exactly four hexadecimal digits.
  /// @return Code unit without surrogate combination.
  unsigned hex() {
    unsigned result = 0;
    for (int i = 0; i < 4; ++i) {
      if (offset_ == source_.size())
        fail();
      const char c = source_[offset_++];
      unsigned   v;
      if (c >= '0' && c <= '9')
        v = c - '0';
      else if (c >= 'a' && c <= 'f')
        v = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F')
        v = c - 'A' + 10;
      else
        fail();
      result = (result << 4) | v;
    }
    return result;
  }
  /// @brief Append one validated Unicode scalar as canonical UTF-8.
  /// @param out Owned decoded string.
  /// @param code Non-surrogate scalar at most U+10FFFF.
  static void append(std::string& out, unsigned code) {
    if (code < 0x80)
      out += char(code);
    else if (code < 0x800) {
      out += char(0xc0 | (code >> 6));
      out += char(0x80 | (code & 63));
    } else if (code < 0x10000) {
      out += char(0xe0 | (code >> 12));
      out += char(0x80 | ((code >> 6) & 63));
      out += char(0x80 | (code & 63));
    } else {
      out += char(0xf0 | (code >> 18));
      out += char(0x80 | ((code >> 12) & 63));
      out += char(0x80 | ((code >> 6) & 63));
      out += char(0x80 | (code & 63));
    }
  }
  /// @brief Decode and validate a quoted JSON string, including UTF-16 surrogate pairs.
  /// @return Owned UTF-8 string.
  std::string string() {
    if (!take('"'))
      fail();
    std::string out;
    while (offset_ < source_.size()) {
      const auto c = static_cast<unsigned char>(source_[offset_++]);
      if (c == '"') {
        utf8(out);
        return out;
      }
      if (c < 32)
        fail();
      if (c != '\\') {
        out += char(c);
        continue;
      }
      if (offset_ == source_.size())
        fail();
      const char escape = source_[offset_++];
      switch (escape) {
        case '"':
        case '\\':
        case '/':
          out += escape;
          break;
        case 'b':
          out += '\b';
          break;
        case 'f':
          out += '\f';
          break;
        case 'n':
          out += '\n';
          break;
        case 'r':
          out += '\r';
          break;
        case 't':
          out += '\t';
          break;
        case 'u': {
          unsigned code = hex();
          if (code >= 0xd800 && code <= 0xdbff) {
            if (!take('\\') || !take('u'))
              fail();
            const unsigned low = hex();
            if (low < 0xdc00 || low > 0xdfff)
              fail();
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
          } else if (code >= 0xdc00 && code <= 0xdfff)
            fail();
          append(out, code);
          break;
        }
        default:
          fail();
      }
    }
    fail();
  }
  /// @brief Parse one grammar value and reject malformed/excessively nested input.
  /// @param depth Enclosing container count.
  /// @return Owned JSON value.
  Json value(std::size_t depth) {
    if (depth > 64 || ++nodes_ > 1000000)
      throw std::length_error("scene JSON nesting or value limit exceeded");
    space();
    if (offset_ == source_.size())
      fail();
    Json       out;
    const char c = source_[offset_];
    if (c == '"') {
      out.kind = Json::Kind::String;
      out.text = string();
      return out;
    }
    if (take('{')) {
      out.kind = Json::Kind::Object;
      space();
      if (take('}'))
        return out;
      do {
        space();
        auto key = string();
        space();
        if (!take(':'))
          fail();
        auto child = value(depth + 1);
        if (!out.object.emplace(std::move(key), std::move(child)).second)
          throw std::invalid_argument("duplicate scene JSON key");
        space();
        if (take('}'))
          return out;
      } while (take(','));
      fail();
    }
    if (take('[')) {
      out.kind = Json::Kind::Array;
      space();
      if (take(']'))
        return out;
      do {
        out.array.push_back(value(depth + 1));
        space();
        if (take(']'))
          return out;
      } while (take(','));
      fail();
    }
    if (c == 't' || c == 'f' || c == 'n') {
      const auto token = c == 't'   ? std::string_view("true")
                         : c == 'f' ? std::string_view("false")
                                    : std::string_view("null");
      if (source_.substr(offset_, token.size()) != token)
        fail();
      offset_ += token.size();
      out.kind    = c == 'n' ? Json::Kind::Null : Json::Kind::Boolean;
      out.boolean = c == 't';
      return out;
    }
    out.kind         = Json::Kind::Number;
    const auto first = offset_;
    take('-');
    if (!take('0')) {
      if (offset_ == source_.size() || source_[offset_] < '1' || source_[offset_] > '9')
        fail();
      do {
        ++offset_;
      } while (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9');
    }
    if (take('.')) {
      const auto begin = offset_;
      while (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9')
        ++offset_;
      if (begin == offset_)
        fail();
    }
    if (take('e') || take('E')) {
      if (!take('+'))
        take('-');
      const auto begin = offset_;
      while (offset_ < source_.size() && source_[offset_] >= '0' && source_[offset_] <= '9')
        ++offset_;
      if (begin == offset_)
        fail();
    }
    out.text = source_.substr(first, offset_ - first);
    return out;
  }

 public:
  /// @brief Retain a bounded JSON document for a single parse.
  /// @param source Borrowed bytes, at most 16 MiB.
  explicit JsonParser(std::string_view source) : source_(source) {
    if (source.size() > 16 * 1024 * 1024)
      throw std::length_error("scene JSON exceeds 16 MiB");
  }
  /// @brief Parse exactly one value with no trailing non-whitespace bytes.
  /// @return Owned strict JSON syntax tree.
  Json parse() {
    auto result = value(0);
    space();
    if (offset_ != source_.size())
      fail();
    return result;
  }
};
/// @brief Encode an exact UTF-8 string with JSON control-character escaping.
/// @param text Valid scalar UTF-8 bytes.
/// @return Quoted JSON string.
inline std::string quote(std::string_view text) {
  utf8(text);
  std::string out = "\"";
  const char* hex = "0123456789abcdef";
  for (unsigned char c : text) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += char(c);
    } else if (c < 32) {
      out += "\\u00";
      out += hex[c >> 4];
      out += hex[c & 15];
    } else
      out += char(c);
  }
  out += '"';
  return out;
}
} // namespace datamunge::game::detail
