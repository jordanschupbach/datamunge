#pragma once

#include <cstddef>
#include <utility>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {
template <typename T>
class VariableLengthArray {
 public:
  explicit VariableLengthArray(std::size_t length = 0) : values_(length) {}
  std::size_t size() const noexcept { return values_.size(); }
  void resize(std::size_t length) { DynamicArray<T> replacement(length); const auto common = length < values_.size() ? length : values_.size(); for (std::size_t i = 0; i < common; ++i) replacement[i] = std::move(values_[i]); values_ = std::move(replacement); }
  T& at(std::size_t index) { return values_.at(index); }
  const T& at(std::size_t index) const { return values_.at(index); }
 private: DynamicArray<T> values_;
};
}  // namespace datamunge::dstruct
