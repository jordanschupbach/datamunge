#pragma once
#include <cstddef>
#include <optional>
#include <utility>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct {
template <typename Key, typename Value>
class AssociationList {
 public:
  void prepend(Key key, Value value) { entries_.push_front({std::move(key), std::move(value)}); }
  std::optional<Value> find(const Key& key) const { std::optional<Value> result; entries_.for_each([&](const auto& entry) { if (!result && entry.first == key) result = entry.second; }); return result; }
  std::size_t size() const noexcept { return entries_.size(); }
 private: LinkedList<std::pair<Key, Value>> entries_;
};
}  // namespace datamunge::dstruct
