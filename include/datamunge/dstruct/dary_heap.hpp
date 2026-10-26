#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename T, std::size_t Arity, typename Compare = std::less<T>>
class DaryHeap {
  static_assert(Arity >= 2, "DaryHeap requires arity of at least two");
 public:
  bool empty() const noexcept { return values_.empty(); }
  std::size_t size() const noexcept { return values_.size(); }
  const T& top() const { if (empty()) throw std::out_of_range("DaryHeap is empty"); return values_[0]; }
  void push(T value) { values_.push_back(std::move(value)); sift_up(values_.size() - 1); }
  std::optional<T> pop() { if (empty()) return std::nullopt; T result = std::move(values_[0]); auto last = values_.pop_back(); if (!values_.empty()) { values_[0] = std::move(*last); sift_down(0); } return result; }
 private:
  void sift_up(std::size_t index) { while (index > 0) { const auto parent = (index - 1) / Arity; if (!compare_(values_[parent], values_[index])) return; std::swap(values_[parent], values_[index]); index = parent; } }
  void sift_down(std::size_t index) { for (;;) { auto best = index; const auto first = index * Arity + 1; for (std::size_t child = first; child < first + Arity && child < values_.size(); ++child) if (compare_(values_[best], values_[child])) best = child; if (best == index) return; std::swap(values_[index], values_[best]); index = best; } }
  DynamicArray<T> values_; Compare compare_{};
};

template <typename T, typename Compare = std::less<T>>
using TernaryHeap = DaryHeap<T, 3, Compare>;

}  // namespace datamunge::dstruct
