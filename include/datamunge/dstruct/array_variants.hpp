#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename T, typename Compare = std::less<T>>
class SortedArray {
 public:
  bool insert(T value) { std::size_t index = 0; while (index < values_.size() && compare_(values_[index], value)) ++index; if (index < values_.size() && !compare_(value, values_[index])) return false; values_.push_back(std::move(value)); for (std::size_t i = values_.size() - 1; i > index; --i) std::swap(values_[i], values_[i - 1]); return true; }
  bool contains(const T& value) const { std::size_t low = 0, high = values_.size(); while (low < high) { const auto mid = low + (high - low) / 2; if (compare_(values_[mid], value)) low = mid + 1; else high = mid; } return low < values_.size() && !compare_(value, values_[low]); }
  const T& at(std::size_t index) const { return values_.at(index); }
  std::size_t size() const noexcept { return values_.size(); }
 private: DynamicArray<T> values_; Compare compare_{};
};

template <typename T>
class LookupTable {
 public:
  explicit LookupTable(std::size_t size) : values_(size), present_(size) {}
  void set(std::size_t index, T value) { check(index); values_[index] = std::move(value); present_.set(index); }
  bool contains(std::size_t index) const { check(index); return present_.at(index); }
  const T& at(std::size_t index) const { if (!contains(index)) throw std::out_of_range("LookupTable entry is absent"); return values_[index]; }
  std::size_t size() const noexcept { return values_.size(); }
 private: void check(std::size_t index) const { if (index >= values_.size()) throw std::out_of_range("LookupTable index out of range"); } DynamicArray<T> values_; BitArray present_;
};

class BitField {
 public:
  explicit BitField(std::size_t width) : bits_(width) {}
  void set(std::size_t offset, std::size_t width, std::uint64_t value) { if (width > 64 || offset + width > bits_.size()) throw std::out_of_range("BitField range out of range"); for (std::size_t i = 0; i < width; ++i) bits_.set(offset + i, (value >> i) & 1U); }
  std::uint64_t get(std::size_t offset, std::size_t width) const { if (width > 64 || offset + width > bits_.size()) throw std::out_of_range("BitField range out of range"); std::uint64_t value = 0; for (std::size_t i = 0; i < width; ++i) if (bits_.at(offset + i)) value |= std::uint64_t{1} << i; return value; }
  /// @brief Total number of bits, fixed at construction.
  /// @return The field width in bits.
  [[nodiscard]] std::size_t width() const noexcept { return bits_.size(); }
 private: BitArray bits_;
};

class Bitboard {
 public:
  bool test(std::size_t square) const { check(square); return (bits_ >> square) & 1U; }
  void set(std::size_t square) { check(square); bits_ |= std::uint64_t{1} << square; }
  void clear(std::size_t square) { check(square); bits_ &= ~(std::uint64_t{1} << square); }
  std::uint64_t value() const noexcept { return bits_; }
 private: static void check(std::size_t square) { if (square >= 64) throw std::out_of_range("Bitboard square out of range"); } std::uint64_t bits_{0};
};

}  // namespace datamunge::dstruct
