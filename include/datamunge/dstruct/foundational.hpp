#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::dstruct {

template <typename T, std::size_t N>
class Array {
 public:
  constexpr std::size_t size() const noexcept { return N; }
  constexpr bool empty() const noexcept { return N == 0; }
  T& at(std::size_t index) {
    if (index >= N) throw std::out_of_range("Array index out of range");
    return data_[index];
  }
  const T& at(std::size_t index) const {
    if (index >= N) throw std::out_of_range("Array index out of range");
    return data_[index];
  }
  T& operator[](std::size_t index) noexcept { return data_[index]; }
  const T& operator[](std::size_t index) const noexcept { return data_[index]; }
  T* begin() noexcept { return data_; }
  T* end() noexcept { return data_ + N; }
  const T* begin() const noexcept { return data_; }
  const T* end() const noexcept { return data_ + N; }

 private:
  T data_[N == 0 ? 1 : N]{};
};

template <typename T>
class DynamicArray {
 public:
  DynamicArray() = default;
  explicit DynamicArray(std::size_t size) { reserve(size); size_ = size; }
  DynamicArray(std::size_t size, const T& value) { reserve(size); for (std::size_t i = 0; i < size; ++i) data_[i] = value; size_ = size; }
  DynamicArray(std::initializer_list<T> values) { for (const auto& value : values) push_back(value); }
  DynamicArray(const DynamicArray& other) { for (const auto& value : other) push_back(value); }
  DynamicArray(DynamicArray&&) noexcept = default;
  DynamicArray& operator=(DynamicArray other) noexcept { swap(other); return *this; }

  void swap(DynamicArray& other) noexcept {
    using std::swap;
    swap(data_, other.data_); swap(size_, other.size_); swap(capacity_, other.capacity_);
  }
  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }
  T& at(std::size_t index) { if (index >= size_) throw std::out_of_range("DynamicArray index out of range"); return data_[index]; }
  const T& at(std::size_t index) const { if (index >= size_) throw std::out_of_range("DynamicArray index out of range"); return data_[index]; }
  T& operator[](std::size_t index) noexcept { return data_[index]; }
  const T& operator[](std::size_t index) const noexcept { return data_[index]; }
  T* begin() noexcept { return data_.get(); }
  T* end() noexcept { return data_.get() + size_; }
  const T* begin() const noexcept { return data_.get(); }
  const T* end() const noexcept { return data_.get() + size_; }
  void reserve(std::size_t requested) {
    if (requested <= capacity_) return;
    auto replacement = std::make_unique<T[]>(requested);
    for (std::size_t i = 0; i < size_; ++i) replacement[i] = std::move(data_[i]);
    data_ = std::move(replacement); capacity_ = requested;
  }
  void push_back(const T& value) { grow(); data_[size_++] = value; }
  void push_back(T&& value) { grow(); data_[size_++] = std::move(value); }
  template <typename... Args> T& emplace_back(Args&&... args) {
    grow(); data_[size_] = T(std::forward<Args>(args)...); return data_[size_++];
  }
  std::optional<T> pop_back() { if (empty()) return std::nullopt; return std::move(data_[--size_]); }
  void clear() noexcept { size_ = 0; }

 private:
  void grow() { if (size_ == capacity_) reserve(capacity_ == 0 ? 4 : capacity_ * 2); }
  std::unique_ptr<T[]> data_;
  std::size_t size_{0};
  std::size_t capacity_{0};
};

template <typename T>
using ArrayList = DynamicArray<T>;

template <typename T>
class LinkedList {
  struct Node { T value; std::unique_ptr<Node> next; explicit Node(T value) : value(std::move(value)) {} };
 public:
  bool empty() const noexcept { return head_ == nullptr; }
  std::size_t size() const noexcept { return size_; }
  void push_front(T value) { auto node = std::make_unique<Node>(std::move(value)); node->next = std::move(head_); head_ = std::move(node); ++size_; }
  void push_back(T value) {
    auto node = std::make_unique<Node>(std::move(value));
    if (!head_) head_ = std::move(node); else { Node* cursor = head_.get(); while (cursor->next) cursor = cursor->next.get(); cursor->next = std::move(node); }
    ++size_;
  }
  std::optional<T> pop_front() { if (!head_) return std::nullopt; auto node = std::move(head_); head_ = std::move(node->next); --size_; return std::move(node->value); }
  T& front() { if (!head_) throw std::out_of_range("LinkedList is empty"); return head_->value; }
  const T& front() const { if (!head_) throw std::out_of_range("LinkedList is empty"); return head_->value; }
  template <typename Function> void for_each(Function&& function) const { for (const Node* node = head_.get(); node; node = node->next.get()) function(node->value); }
 private:
  std::unique_ptr<Node> head_;
  std::size_t size_{0};
};

template <typename T>
class DoublyLinkedList {
  struct Node { T value; Node* previous{nullptr}; std::unique_ptr<Node> next; explicit Node(T value) : value(std::move(value)) {} };
 public:
  bool empty() const noexcept { return head_ == nullptr; }
  std::size_t size() const noexcept { return size_; }
  void push_front(T value) { auto node = std::make_unique<Node>(std::move(value)); node->next = std::move(head_); if (node->next) node->next->previous = node.get(); else tail_ = node.get(); head_ = std::move(node); ++size_; }
  void push_back(T value) { auto node = std::make_unique<Node>(std::move(value)); node->previous = tail_; Node* raw = node.get(); if (tail_) tail_->next = std::move(node); else head_ = std::move(node); tail_ = raw; ++size_; }
  std::optional<T> pop_front() { if (!head_) return std::nullopt; auto node = std::move(head_); head_ = std::move(node->next); if (head_) head_->previous = nullptr; else tail_ = nullptr; --size_; return std::move(node->value); }
  std::optional<T> pop_back() { if (!tail_) return std::nullopt; if (tail_ == head_.get()) return pop_front(); Node* previous = tail_->previous; auto node = std::move(previous->next); tail_ = previous; --size_; return std::move(node->value); }
  T& front() { if (!head_) throw std::out_of_range("DoublyLinkedList is empty"); return head_->value; }
  T& back() { if (!tail_) throw std::out_of_range("DoublyLinkedList is empty"); return tail_->value; }
  /// @brief Copies the elements front to back without modifying the list.
  /// @return The elements in list order (empty when the list is empty).
  [[nodiscard]] std::vector<T> to_vector() const {
    std::vector<T> values;
    values.reserve(size_);
    for (const Node* node = head_.get(); node; node = node->next.get()) values.push_back(node->value);
    return values;
  }
 private:
  std::unique_ptr<Node> head_;
  Node* tail_{nullptr};
  std::size_t size_{0};
};

template <typename T> class Stack {
 public:
  void push(T value) { values_.push_front(std::move(value)); }
  std::optional<T> pop() { return values_.pop_front(); }
  T& top() { return values_.front(); }
  bool empty() const noexcept { return values_.empty(); }
  std::size_t size() const noexcept { return values_.size(); }
  /// @brief Copies the elements from top to bottom without popping them.
  /// @return The elements, top of the stack first.
  [[nodiscard]] std::vector<T> to_vector() const {
    std::vector<T> values;
    values.reserve(values_.size());
    values_.for_each([&](const T& value) { values.push_back(value); });
    return values;
  }
 private: LinkedList<T> values_;
};

template <typename T> class Queue {
 public:
  void push(T value) { values_.push_back(std::move(value)); }
  std::optional<T> pop() { return values_.pop_front(); }
  T& front() { return values_.front(); }
  bool empty() const noexcept { return values_.empty(); }
  std::size_t size() const noexcept { return values_.size(); }
  /// @brief Copies the elements from front to back without dequeuing them.
  /// @return The elements, next-to-pop first.
  [[nodiscard]] std::vector<T> to_vector() const { return values_.to_vector(); }
 private: DoublyLinkedList<T> values_;
};

template <typename T> class DoubleEndedQueue {
 public:
  void push_front(T value) { values_.push_front(std::move(value)); }
  void push_back(T value) { values_.push_back(std::move(value)); }
  std::optional<T> pop_front() { return values_.pop_front(); }
  std::optional<T> pop_back() { return values_.pop_back(); }
  T& front() { return values_.front(); }
  T& back() { return values_.back(); }
  bool empty() const noexcept { return values_.empty(); }
  std::size_t size() const noexcept { return values_.size(); }
  /// @brief Copies the elements from front to back without removing them.
  /// @return The elements in deque order.
  [[nodiscard]] std::vector<T> to_vector() const { return values_.to_vector(); }
 private: DoublyLinkedList<T> values_;
};

template <typename T>
class CircularBuffer {
 public:
  explicit CircularBuffer(std::size_t capacity) : data_(capacity ? std::make_unique<T[]>(capacity) : nullptr), capacity_(capacity) {}
  bool push(T value) { if (size_ == capacity_) return false; data_[(head_ + size_++) % capacity_] = std::move(value); return true; }
  std::optional<T> pop() { if (size_ == 0) return std::nullopt; T value = std::move(data_[head_]); head_ = (head_ + 1) % capacity_; --size_; return value; }
  bool empty() const noexcept { return size_ == 0; }
  bool full() const noexcept { return size_ == capacity_; }
  std::size_t size() const noexcept { return size_; }
  std::size_t capacity() const noexcept { return capacity_; }
 private:
  std::unique_ptr<T[]> data_; std::size_t capacity_{0}; std::size_t head_{0}; std::size_t size_{0};
};

class BitArray {
 public:
  explicit BitArray(std::size_t bits) : bits_(bits), bytes_((bits + 7) / 8 ? std::make_unique<unsigned char[]>((bits + 7) / 8) : nullptr) {}
  std::size_t size() const noexcept { return bits_; }
  bool at(std::size_t index) const { check(index); return (bytes_[index / 8] >> (index % 8)) & 1U; }
  void set(std::size_t index, bool value = true) { check(index); const unsigned char mask = static_cast<unsigned char>(1U << (index % 8)); if (value) bytes_[index / 8] |= mask; else bytes_[index / 8] &= static_cast<unsigned char>(~mask); }
  void flip(std::size_t index) { check(index); bytes_[index / 8] ^= static_cast<unsigned char>(1U << (index % 8)); }
 private:
  void check(std::size_t index) const { if (index >= bits_) throw std::out_of_range("BitArray index out of range"); }
  std::size_t bits_; std::unique_ptr<unsigned char[]> bytes_;
};

}  // namespace datamunge::dstruct
