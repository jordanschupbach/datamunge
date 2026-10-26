#pragma once

#include <cstddef>
#include <functional>
#include <stdexcept>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename T, typename Hash = std::hash<T>>
class BloomFilter {
 public:
  BloomFilter(std::size_t bits, std::size_t hash_count) : bits_(bits), hash_count_(hash_count) { if (bits == 0 || hash_count == 0) throw std::invalid_argument("BloomFilter requires positive dimensions"); }
  void insert(const T& value) { for (std::size_t i = 0; i < hash_count_; ++i) bits_.set(index(value, i)); }
  bool may_contain(const T& value) const { for (std::size_t i = 0; i < hash_count_; ++i) if (!bits_.at(index(value, i))) return false; return true; }
  std::size_t bit_count() const noexcept { return bits_.size(); }
 private:
  std::size_t index(const T& value, std::size_t salt) const { const auto base = hash_(value); const auto mixed = base ^ (salt + 0x9e3779b97f4a7c15ULL + (base << 6U) + (base >> 2U)); return mixed % bits_.size(); }
  BitArray bits_; std::size_t hash_count_; Hash hash_{};
};

}  // namespace datamunge::dstruct
