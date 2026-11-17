#pragma once
/// @file decimal_uint.hpp
/// @brief Private nonnegative arbitrary-precision arithmetic for exact decimal reports.
#include <algorithm>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
/// @brief Invocation-local implementation-only decimal arithmetic; not an installed public API.
namespace datamunge::algorithms::detail {
/// @brief Owned nonnegative base-billion integer with exact schoolbook products/division.
/// Zero has no words. Storage is linear in decimal digits. Addition/subtraction and
/// small products are linear in word count, full products quadratic. Long division
/// scans dividend words and binary-searches each base-billion quotient digit; it
/// has quadratic word work times log(base). No shared state or external dependency.
class DecimalUInt {
  /// @brief Little-endian base-billion digits, normalized after every operation.
  std::vector<std::uint32_t> words_;
  /// @brief Radix chosen so each product/accumulation fits uint64.
  static constexpr std::uint64_t base = 1000000000;
  /// @brief Remove unused high zero words; zero becomes empty.
  void normalize() {
    while (!words_.empty() && !words_.back())
      words_.pop_back();
  }

 public:
  /// @brief Construct an owned exact unsigned64-bit integer, default zero.
  /// @param value Complete nonnegative input.
  DecimalUInt(std::uint64_t value = 0) {
    while (value) {
      words_.push_back(static_cast<std::uint32_t>(value % base));
      value /= base;
    }
  }
  /// @brief Parse complete nonempty decimal digits, without octal interpretation.
  /// @param text Borrowed nonempty ASCII decimal string.
  /// @throws std::invalid_argument Empty or nondecimal input.
  explicit DecimalUInt(const std::string& text) {
    if (text.empty())
      throw std::invalid_argument("decimal uint: empty");
    for (char c : text) {
      if (c < '0' || c > '9')
        throw std::invalid_argument("decimal uint: digit");
      *this *= 10;
      *this += DecimalUInt(static_cast<std::uint64_t>(c - '0'));
    }
  }
  /// @brief Compare exact owned values.
  /// @param other Borrowed second value.
  /// @return Numeric ordering.
  std::strong_ordering operator<=>(const DecimalUInt& other) const {
    if (words_.size() != other.words_.size())
      return words_.size() <=> other.words_.size();
    for (std::size_t i = words_.size(); i > 0; --i)
      if (words_[i - 1] != other.words_[i - 1])
        return words_[i - 1] <=> other.words_[i - 1];
    return std::strong_ordering::equal;
  }
  /// @brief Compare normalized exact values for equality.
  /// @param other Borrowed second value.
  /// @return Same integer value.
  bool operator==(const DecimalUInt& other) const { return words_ == other.words_; }
  /// @brief Add an exact nonnegative value in place.
  /// @param other Borrowed addend, aliasing this is permitted.
  /// @return Mutable receiver.
  DecimalUInt& operator+=(const DecimalUInt& other) {
    const auto n = std::max(words_.size(), other.words_.size());
    words_.resize(n);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t value =
          std::uint64_t(words_[i]) + (i < other.words_.size() ? other.words_[i] : 0) + carry;
      words_[i] = static_cast<std::uint32_t>(value % base);
      carry     = value / base;
    }
    if (carry)
      words_.push_back(static_cast<std::uint32_t>(carry));
    return *this;
  }
  /// @brief Subtract a no-larger exact value in place.
  /// @param other Borrowed subtrahend, aliasing this is permitted.
  /// @return Mutable receiver.
  /// @throws std::invalid_argument Subtrahend exceeds receiver.
  DecimalUInt& operator-=(const DecimalUInt& other) {
    if (*this < other)
      throw std::invalid_argument("decimal uint: negative subtraction");
    std::uint64_t borrow = 0;
    for (std::size_t i = 0; i < words_.size(); ++i) {
      const std::uint64_t sub = (i < other.words_.size() ? other.words_[i] : 0) + borrow;
      if (words_[i] < sub) {
        words_[i] = static_cast<std::uint32_t>(base + words_[i] - sub);
        borrow    = 1;
      } else {
        words_[i] = static_cast<std::uint32_t>(words_[i] - sub);
        borrow    = 0;
      }
    }
    normalize();
    return *this;
  }
  /// @brief Multiply in place by an exact uint32 factor.
  /// @param factor Nonnegative factor, all uint32 values are supported.
  /// @return Mutable receiver.
  DecimalUInt& operator*=(std::uint32_t factor) {
    if (!factor) {
      words_.clear();
      return *this;
    }
    std::uint64_t carry = 0;
    for (auto& word : words_) {
      const std::uint64_t value = std::uint64_t(word) * factor + carry;
      word                      = static_cast<std::uint32_t>(value % base);
      carry                     = value / base;
    }
    while (carry) {
      words_.push_back(static_cast<std::uint32_t>(carry % base));
      carry /= base;
    }
    return *this;
  }
  /// @brief Add two owned exact values.
  /// @param a Owned first operand.
  /// @param b Borrowed second operand.
  /// @return Owned exact sum.
  friend DecimalUInt operator+(DecimalUInt a, const DecimalUInt& b) { return a += b; }
  /// @brief Subtract a no-larger value from an owned operand.
  /// @param a Owned minuend.
  /// @param b Borrowed subtrahend.
  /// @return Owned nonnegative difference.
  /// @throws std::invalid_argument Negative result.
  friend DecimalUInt operator-(DecimalUInt a, const DecimalUInt& b) { return a -= b; }
  /// @brief Multiply an owned integer by an exact small factor.
  /// @param a Owned integer.
  /// @param factor Complete uint32 factor.
  /// @return Owned exact product.
  friend DecimalUInt operator*(DecimalUInt a, std::uint32_t factor) { return a *= factor; }
  /// @brief Multiply a small factor by an owned integer.
  /// @param factor Complete uint32 factor.
  /// @param a Owned integer.
  /// @return Owned exact product.
  friend DecimalUInt operator*(std::uint32_t factor, DecimalUInt a) { return a *= factor; }
  /// @brief Multiply two complete arbitrary-precision integers.
  /// @param a Borrowed first factor.
  /// @param b Borrowed second factor.
  /// @return Owned exact schoolbook product.
  friend DecimalUInt operator*(const DecimalUInt& a, const DecimalUInt& b) {
    DecimalUInt result;
    if (a.words_.empty() || b.words_.empty())
      return result;
    if (a.words_.size() > result.words_.max_size() - b.words_.size())
      throw std::length_error("decimal uint: product capacity");
    result.words_.assign(a.words_.size() + b.words_.size(), 0);
    for (std::size_t i = 0; i < a.words_.size(); ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < b.words_.size(); ++j) {
        const std::uint64_t value =
            std::uint64_t(a.words_[i]) * b.words_[j] + result.words_[i + j] + carry;
        result.words_[i + j] = static_cast<std::uint32_t>(value % base);
        carry                = value / base;
      }
      result.words_[i + b.words_.size()] = static_cast<std::uint32_t>(carry);
    }
    result.normalize();
    return result;
  }
  /// @brief Divide exact nonnegative integers and return the floor quotient.
  /// @param numerator Borrowed complete dividend.
  /// @param denominator Borrowed nonzero divisor.
  /// @return Owned exact floor quotient.
  /// @throws std::invalid_argument Zero divisor.
  friend DecimalUInt operator/(const DecimalUInt& numerator, const DecimalUInt& denominator) {
    if (denominator.words_.empty())
      throw std::invalid_argument("decimal uint: zero division");
    DecimalUInt quotient, remainder;
    quotient.words_.assign(numerator.words_.size(), 0);
    for (std::size_t i = numerator.words_.size(); i > 0; --i) {
      remainder.words_.insert(remainder.words_.begin(), numerator.words_[i - 1]);
      remainder.normalize();
      std::uint32_t low = 0, high = static_cast<std::uint32_t>(base - 1), best = 0;
      while (low <= high) {
        const auto mid     = low + (high - low) / 2;
        const auto product = denominator * mid;
        if (product <= remainder) {
          best = mid;
          if (mid == base - 1)
            break;
          low = mid + 1;
        } else {
          if (mid == 0)
            break;
          high = mid - 1;
        }
      }
      quotient.words_[i - 1] = best;
      remainder -= denominator * best;
    }
    quotient.normalize();
    return quotient;
  }
  /// @brief Format the exact integer as canonical decimal digits.
  /// @return Owned decimal string with no leading zeros except zero itself.
  std::string str() const {
    if (words_.empty())
      return "0";
    std::string text = std::to_string(words_.back());
    for (std::size_t i = words_.size() - 1; i > 0; --i) {
      auto word = std::to_string(words_[i - 1]);
      text.append(9 - word.size(), '0');
      text += word;
    }
    return text;
  }
  /// @brief Return canonical decimal text through the internal conversion spelling.
  /// @tparam T Only std::string is supported.
  /// @return Owned canonical decimal string.
  template<class T>
  T convert_to() const {
    static_assert(std::is_same_v<T, std::string>);
    return str();
  }
};
} // namespace datamunge::algorithms::detail
