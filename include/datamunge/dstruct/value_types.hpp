#pragma once

#include <cstddef>
#include <optional>
#include <tuple>
#include <utility>
#include <variant>

namespace datamunge::dstruct {
template <typename First, typename Second> struct ProductType { First first; Second second; };
template <typename... Fields> class Tuple { public: explicit Tuple(Fields... fields) : values_(std::move(fields)...) {} template <std::size_t I> auto& get() { return std::get<I>(values_); } template <std::size_t I> const auto& get() const { return std::get<I>(values_); } private: std::tuple<Fields...> values_; };
template <typename... Alternatives> class TaggedUnion { public: template <typename T> void set(T value) { value_ = std::move(value); } template <typename T> bool is() const { return std::holds_alternative<T>(value_); } template <typename T> T& get() { return std::get<T>(value_); } private: std::variant<Alternatives...> value_; };
template <typename T> class Union { public: void set(T value) { value_ = std::move(value); } const std::optional<T>& value() const { return value_; } private: std::optional<T> value_; };
}  // namespace datamunge::dstruct
