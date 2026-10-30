#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

namespace datamunge::dstruct {
template <typename First, typename Second> struct ProductType { First first; Second second; };
template <typename... Fields> class Tuple { public: explicit Tuple(Fields... fields) : values_(std::move(fields)...) {} template <std::size_t I> auto& get() { return std::get<I>(values_); } template <std::size_t I> const auto& get() const { return std::get<I>(values_); } private: std::tuple<Fields...> values_; };
template <typename... Alternatives> class TaggedUnion { public: template <typename T> void set(T value) { value_ = std::move(value); } template <typename T> bool is() const { return std::holds_alternative<T>(value_); } template <typename T> T& get() { return std::get<T>(value_); }
  /// @brief Read-only access to the held alternative.
  /// @tparam T The alternative to read.
  /// @return The held value.
  /// @throws std::bad_variant_access If T is not the held alternative.
  template <typename T> [[nodiscard]] const T& get() const { return std::get<T>(value_); }
  /// @brief Zero-based position of the held alternative in the Alternatives pack.
  /// @return The index (std::variant_npos only if valueless by exception).
  [[nodiscard]] std::size_t index() const noexcept { return value_.index(); }
  /// @brief Read-only access to the underlying variant, e.g. for std::visit.
  /// @return The stored std::variant.
  [[nodiscard]] const std::variant<Alternatives...>& value() const noexcept { return value_; }
 private: std::variant<Alternatives...> value_; };
template <typename T> class Union { public: void set(T value) { value_ = std::move(value); } const std::optional<T>& value() const { return value_; } private: std::optional<T> value_; };
}  // namespace datamunge::dstruct
