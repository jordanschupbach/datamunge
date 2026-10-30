#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename T>
class TernaryTree {
 public:
  struct Node { T value; std::unique_ptr<Node> left, middle, right; explicit Node(T value) : value(std::move(value)) {} };
  explicit TernaryTree(T root) : root_(std::make_unique<Node>(std::move(root))) {}
  Node& root() { return *root_; }
  /// @brief Read-only access to the root node (the tree always has one).
  /// @return The root node; walk its child pointers to traverse the tree.
  [[nodiscard]] const Node& root() const noexcept { return *root_; }
  Node& set_left(Node& parent, T value) { parent.left = std::make_unique<Node>(std::move(value)); return *parent.left; }
  Node& set_middle(Node& parent, T value) { parent.middle = std::make_unique<Node>(std::move(value)); return *parent.middle; }
  Node& set_right(Node& parent, T value) { parent.right = std::make_unique<Node>(std::move(value)); return *parent.right; }
 private: std::unique_ptr<Node> root_;
};

template <typename T, std::size_t Arity>
class KaryTree {
  static_assert(Arity > 0, "KaryTree requires positive arity");
 public:
  struct Node { T value; std::unique_ptr<Node> children[Arity]; explicit Node(T value) : value(std::move(value)) {} };
  explicit KaryTree(T root) : root_(std::make_unique<Node>(std::move(root))) {}
  Node& root() { return *root_; }
  /// @brief Read-only access to the root node (the tree always has one).
  /// @return The root node; walk its child pointers to traverse the tree.
  [[nodiscard]] const Node& root() const noexcept { return *root_; }
  Node& set_child(Node& parent, std::size_t index, T value) { if (index >= Arity) throw std::out_of_range("KaryTree child index out of range"); parent.children[index] = std::make_unique<Node>(std::move(value)); return *parent.children[index]; }
 private: std::unique_ptr<Node> root_;
};

template <typename T>
class RoseTree {
 public:
  struct Node { T value; DynamicArray<std::unique_ptr<Node>> children; explicit Node(T value) : value(std::move(value)) {} };
  explicit RoseTree(T root) : root_(std::make_unique<Node>(std::move(root))) {}
  Node& root() { return *root_; }
  /// @brief Read-only access to the root node (the tree always has one).
  /// @return The root node; walk its child pointers to traverse the tree.
  [[nodiscard]] const Node& root() const noexcept { return *root_; }
  Node& add_child(Node& parent, T value) { parent.children.push_back(std::make_unique<Node>(std::move(value))); return *parent.children[parent.children.size() - 1]; }
 private: std::unique_ptr<Node> root_;
};

}  // namespace datamunge::dstruct
