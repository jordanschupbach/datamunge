#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
class AssociativeArray {
  struct Node {
    Key key;
    Value value;
    std::unique_ptr<Node> next;
    Node(Key key, Value value) : key(std::move(key)), value(std::move(value)) {}
  };
 public:
  explicit AssociativeArray(std::size_t bucket_count = 17) : buckets_(bucket_count ? std::make_unique<std::unique_ptr<Node>[]>(bucket_count) : throw std::invalid_argument("bucket_count must be positive")), bucket_count_(bucket_count) {}
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  bool contains(const Key& key) const { return find_node(key) != nullptr; }
  Value* find(const Key& key) { Node* node = find_node(key); return node ? &node->value : nullptr; }
  const Value* find(const Key& key) const { const Node* node = find_node(key); return node ? &node->value : nullptr; }
  Value& at(const Key& key) { auto* value = find(key); if (!value) throw std::out_of_range("key not found"); return *value; }
  const Value& at(const Key& key) const { const auto* value = find(key); if (!value) throw std::out_of_range("key not found"); return *value; }
  void insert_or_assign(Key key, Value value) {
    const auto index = bucket(key);
    for (Node* node = buckets_[index].get(); node; node = node->next.get()) if (equal_(node->key, key)) { node->value = std::move(value); return; }
    auto node = std::make_unique<Node>(std::move(key), std::move(value)); node->next = std::move(buckets_[index]); buckets_[index] = std::move(node); ++size_;
  }
  bool erase(const Key& key) {
    auto* link = &buckets_[bucket(key)];
    while (*link) { if (equal_((*link)->key, key)) { *link = std::move((*link)->next); --size_; return true; } link = &(*link)->next; }
    return false;
  }
 private:
  std::size_t bucket(const Key& key) const { return hash_(key) % bucket_count_; }
  Node* find_node(const Key& key) { for (Node* node = buckets_[bucket(key)].get(); node; node = node->next.get()) if (equal_(node->key, key)) return node; return nullptr; }
  const Node* find_node(const Key& key) const { for (const Node* node = buckets_[bucket(key)].get(); node; node = node->next.get()) if (equal_(node->key, key)) return node; return nullptr; }
  std::unique_ptr<std::unique_ptr<Node>[]> buckets_;
  std::size_t bucket_count_;
  std::size_t size_{0};
  Hash hash_{};
  Equal equal_{};
};

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
class Multimap {
  struct Node { Key key; Value value; std::unique_ptr<Node> next; Node(Key key, Value value) : key(std::move(key)), value(std::move(value)) {} };
 public:
  explicit Multimap(std::size_t bucket_count = 17) : buckets_(bucket_count ? std::make_unique<std::unique_ptr<Node>[]>(bucket_count) : throw std::invalid_argument("bucket_count must be positive")), bucket_count_(bucket_count) {}
  void insert(Key key, Value value) { const auto index = bucket(key); auto node = std::make_unique<Node>(std::move(key), std::move(value)); node->next = std::move(buckets_[index]); buckets_[index] = std::move(node); ++size_; }
  std::size_t count(const Key& key) const { std::size_t result = 0; for (auto* node = buckets_[bucket(key)].get(); node; node = node->next.get()) if (equal_(node->key, key)) ++result; return result; }
  std::size_t erase(const Key& key) { std::size_t removed = 0; auto* link = &buckets_[bucket(key)]; while (*link) { if (equal_((*link)->key, key)) { *link = std::move((*link)->next); ++removed; --size_; } else link = &(*link)->next; } return removed; }
  std::size_t size() const noexcept { return size_; }
 private:
  std::size_t bucket(const Key& key) const { return hash_(key) % bucket_count_; }
  std::unique_ptr<std::unique_ptr<Node>[]> buckets_; std::size_t bucket_count_; std::size_t size_{0}; Hash hash_{}; Equal equal_{};
};

template <typename T, typename Hash = std::hash<T>, typename Equal = std::equal_to<T>>
class Set {
 public:
  bool insert(T value) { if (values_.contains(value)) return false; values_.insert_or_assign(std::move(value), true); return true; }
  bool contains(const T& value) const { return values_.contains(value); }
  bool erase(const T& value) { return values_.erase(value); }
  std::size_t size() const noexcept { return values_.size(); }
 private: AssociativeArray<T, bool, Hash, Equal> values_;
};

template <typename T, typename Hash = std::hash<T>, typename Equal = std::equal_to<T>>
class Multiset {
 public:
  void insert(T value) { values_.insert(std::move(value), true); }
  std::size_t count(const T& value) const { return values_.count(value); }
  std::size_t erase(const T& value) { return values_.erase(value); }
  std::size_t size() const noexcept { return values_.size(); }
 private: Multimap<T, bool, Hash, Equal> values_;
};

template <typename T, typename Compare = std::less<T>>
class BinaryHeap {
 public:
  BinaryHeap() = default;
  BinaryHeap(std::initializer_list<T> values) { for (const auto& value : values) push(value); }
  bool empty() const noexcept { return values_.empty(); }
  std::size_t size() const noexcept { return values_.size(); }
  const T& top() const { if (empty()) throw std::out_of_range("BinaryHeap is empty"); return values_[0]; }
  void push(T value) { values_.push_back(std::move(value)); sift_up(values_.size() - 1); }
  std::optional<T> pop() { if (empty()) return std::nullopt; T result = std::move(values_[0]); auto last = values_.pop_back(); if (!values_.empty()) { values_[0] = std::move(*last); sift_down(0); } return result; }
 private:
  void sift_up(std::size_t index) { while (index && compare_(values_[(index - 1) / 2], values_[index])) { using std::swap; swap(values_[(index - 1) / 2], values_[index]); index = (index - 1) / 2; } }
  void sift_down(std::size_t index) { for (;;) { const auto left = index * 2 + 1; const auto right = left + 1; auto best = index; if (left < values_.size() && compare_(values_[best], values_[left])) best = left; if (right < values_.size() && compare_(values_[best], values_[right])) best = right; if (best == index) return; using std::swap; swap(values_[index], values_[best]); index = best; } }
  DynamicArray<T> values_; Compare compare_{};
};

class DisjointSet {
 public:
  explicit DisjointSet(std::size_t size) : parent_(size), rank_(size) { for (std::size_t i = 0; i < size; ++i) parent_[i] = i; }
  std::size_t find(std::size_t item) { check(item); if (parent_[item] != item) parent_[item] = find(parent_[item]); return parent_[item]; }
  bool unite(std::size_t first, std::size_t second) { auto a = find(first), b = find(second); if (a == b) return false; if (rank_[a] < rank_[b]) std::swap(a, b); parent_[b] = a; if (rank_[a] == rank_[b]) ++rank_[a]; return true; }
  bool connected(std::size_t first, std::size_t second) { return find(first) == find(second); }
 private:
  void check(std::size_t item) const { if (item >= parent_.size()) throw std::out_of_range("DisjointSet item out of range"); }
  DynamicArray<std::size_t> parent_; DynamicArray<std::size_t> rank_;
};

}  // namespace datamunge::dstruct
