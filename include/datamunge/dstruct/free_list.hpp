#pragma once
#include <cstddef>
#include <optional>
#include <utility>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct {
template <typename T> class FreeList { public: void release(T value) { available_.push_front(std::move(value)); } std::optional<T> acquire() { return available_.pop_front(); } std::size_t size() const noexcept { return available_.size(); } private: LinkedList<T> available_; };
}  // namespace datamunge::dstruct
