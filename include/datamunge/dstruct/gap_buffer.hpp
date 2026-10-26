#pragma once
#include <cstddef>
#include <utility>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct {
template <typename T> class GapBuffer { public: std::size_t cursor() const noexcept { return left_.size(); } void move_cursor(std::size_t position) { while (left_.size() > position) right_.push_back(*left_.pop_back()); while (left_.size() < position && !right_.empty()) left_.push_back(*right_.pop_back()); } void insert(T value) { left_.push_back(std::move(value)); } bool erase_before() { return left_.pop_back().has_value(); } DynamicArray<T> values() const { DynamicArray<T> result; for (const auto& item : left_) result.push_back(item); for (std::size_t i = right_.size(); i; --i) result.push_back(right_[i - 1]); return result; } private: DynamicArray<T> left_, right_; };
}  // namespace datamunge::dstruct
