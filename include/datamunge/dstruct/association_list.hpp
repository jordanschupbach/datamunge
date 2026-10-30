#pragma once
#include <vector>
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
  /// @brief Copies the entries, most recently prepended first, without modifying the list.
  /// @return The key/value pairs in lookup order (shadowed duplicates included).
  [[nodiscard]] std::vector<std::pair<Key, Value>> to_vector() const {
    std::vector<std::pair<Key, Value>> entries;
    entries.reserve(entries_.size());
    entries_.for_each([&](const std::pair<Key, Value>& entry) { entries.push_back(entry); });
    return entries;
  }
 private: LinkedList<std::pair<Key, Value>> entries_;
};
}  // namespace datamunge::dstruct
