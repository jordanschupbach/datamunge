#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename T, typename Compare = std::less<T>>
class BinarySearchTree {
  struct Node { T value; std::unique_ptr<Node> left, right; explicit Node(T value) : value(std::move(value)) {} };
 public:
  bool insert(T value) { return insert(root_, std::move(value)); }
  bool contains(const T& value) const { const Node* node = root_.get(); while (node) { if (equal(value, node->value)) return true; node = compare_(value, node->value) ? node->left.get() : node->right.get(); } return false; }
  bool erase(const T& value) { return erase(root_, value); }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return !root_; }
  std::optional<T> minimum() const { const Node* node = root_.get(); if (!node) return std::nullopt; while (node->left) node = node->left.get(); return node->value; }
 private:
  bool equal(const T& a, const T& b) const { return !compare_(a, b) && !compare_(b, a); }
  bool insert(std::unique_ptr<Node>& node, T value) { if (!node) { node = std::make_unique<Node>(std::move(value)); ++size_; return true; } if (equal(value, node->value)) return false; return insert(compare_(value, node->value) ? node->left : node->right, std::move(value)); }
  bool erase(std::unique_ptr<Node>& node, const T& value) { if (!node) return false; if (compare_(value, node->value)) return erase(node->left, value); if (compare_(node->value, value)) return erase(node->right, value); if (!node->left) node = std::move(node->right); else if (!node->right) node = std::move(node->left); else { auto* successor = node->right.get(); while (successor->left) successor = successor->left.get(); node->value = successor->value; return erase(node->right, successor->value); } --size_; return true; }
  std::unique_ptr<Node> root_; std::size_t size_{0}; Compare compare_{};
};

class Trie {
  struct Node { bool terminal{false}; std::unique_ptr<Node> children[256]; };
 public:
  void insert(const std::string& word) { Node* node = &root_; for (unsigned char letter : word) { if (!node->children[letter]) node->children[letter] = std::make_unique<Node>(); node = node->children[letter].get(); } if (!node->terminal) { node->terminal = true; ++size_; } }
  bool contains(const std::string& word) const { const Node* node = descend(word); return node && node->terminal; }
  bool starts_with(const std::string& prefix) const { return descend(prefix) != nullptr; }
  std::size_t size() const noexcept { return size_; }
 private:
  const Node* descend(const std::string& word) const { const Node* node = &root_; for (unsigned char letter : word) { node = node->children[letter].get(); if (!node) return nullptr; } return node; }
  Node root_; std::size_t size_{0};
};

template <typename T>
class FenwickTree {
 public:
  explicit FenwickTree(std::size_t size) : values_(size + 1) {}
  void add(std::size_t index, T delta) { if (index >= size()) throw std::out_of_range("FenwickTree index out of range"); for (++index; index < values_.size(); index += index & -index) values_[index] += delta; }
  T prefix_sum(std::size_t end) const { if (end > size()) throw std::out_of_range("FenwickTree end out of range"); T total{}; for (; end; end -= end & -end) total += values_[end]; return total; }
  T range_sum(std::size_t begin, std::size_t end) const { if (begin > end) throw std::invalid_argument("invalid range"); return prefix_sum(end) - prefix_sum(begin); }
  std::size_t size() const noexcept { return values_.size() - 1; }
 private: DynamicArray<T> values_;
};

template <typename T, typename Combine = std::plus<T>>
class SegmentTree {
 public:
  SegmentTree(std::size_t size, T identity = T{}) : size_(size), identity_(std::move(identity)), values_(storage_size(size), identity_) {}
  void set(std::size_t index, T value) { if (index >= size_) throw std::out_of_range("SegmentTree index out of range"); set(1, 0, size_, index, std::move(value)); }
  T query(std::size_t begin, std::size_t end) const { if (begin > end || end > size_) throw std::out_of_range("SegmentTree range out of range"); return query(1, 0, size_, begin, end); }
 private:
  static std::size_t storage_size(std::size_t size) { return size ? size * 4 : 1; }
  void set(std::size_t node, std::size_t left, std::size_t right, std::size_t index, T value) { if (right - left == 1) { values_[node] = std::move(value); return; } const auto middle = (left + right) / 2; if (index < middle) set(node * 2, left, middle, index, std::move(value)); else set(node * 2 + 1, middle, right, index, std::move(value)); values_[node] = combine_(values_[node * 2], values_[node * 2 + 1]); }
  T query(std::size_t node, std::size_t left, std::size_t right, std::size_t begin, std::size_t end) const { if (end <= left || right <= begin) return identity_; if (begin <= left && right <= end) return values_[node]; const auto middle = (left + right) / 2; return combine_(query(node * 2, left, middle, begin, end), query(node * 2 + 1, middle, right, begin, end)); }
  std::size_t size_; T identity_; DynamicArray<T> values_; Combine combine_{};
};

}  // namespace datamunge::dstruct
