#pragma once
#include <vector>
#include <optional>
#include <utility>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { template <typename T> class SelfOrganizingList { public: void push_front(T value) { values_.push_front(std::move(value)); } /// @brief Looks up a value and moves it to the front (move-to-front heuristic).
  ///
  /// The elements that were ahead of it keep their relative order behind it; a
  /// miss leaves the list unchanged. Linear in the list's length.
  /// @param value  The value to search for (compared with `==`).
  /// @return A copy of the found value, or std::nullopt.
  std::optional<T> find(const T& value) {
    // Move-to-front: pop until the match, then rebuild the front as
    // [match, skipped elements in their original order, rest...]. A miss
    // pops everything and restores it unchanged.
    std::vector<T> skipped;
    std::optional<T> found;
    while (auto item = values_.pop_front()) {
      if (*item == value) {
        found = std::move(item);
        break;
      }
      skipped.push_back(std::move(*item));
    }
    for (auto it = skipped.rbegin(); it != skipped.rend(); ++it) values_.push_front(std::move(*it));
    if (found) values_.push_front(*found);
    return found;
  } T& front() { return values_.front(); }
  /// @brief Number of stored elements.
  /// @return The element count.
  [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
  /// @brief Copies the elements front to back without reorganizing the list (unlike find()).
  /// @return The elements, most recently found or pushed first.
  [[nodiscard]] std::vector<T> to_vector() const {
    std::vector<T> values;
    values.reserve(values_.size());
    values_.for_each([&](const T& value) { values.push_back(value); });
    return values;
  }
  private: LinkedList<T> values_; }; }
