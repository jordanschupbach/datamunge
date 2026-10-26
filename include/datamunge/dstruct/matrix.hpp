#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>

#include <datamunge/dstruct/associative.hpp>
#include <datamunge/dstruct/foundational.hpp>

namespace datamunge::dstruct {

template <typename T>
class Matrix {
 public:
  Matrix(std::size_t rows, std::size_t columns, const T& value = T{}) : rows_(rows), columns_(columns), values_(rows * columns, value) {}
  std::size_t rows() const noexcept { return rows_; }
  std::size_t columns() const noexcept { return columns_; }
  T& at(std::size_t row, std::size_t column) { return values_[offset(row, column)]; }
  const T& at(std::size_t row, std::size_t column) const { return values_[offset(row, column)]; }
  Matrix transpose() const { Matrix result(columns_, rows_); for (std::size_t r = 0; r < rows_; ++r) for (std::size_t c = 0; c < columns_; ++c) result.at(c, r) = at(r, c); return result; }
 private:
  std::size_t offset(std::size_t row, std::size_t column) const { if (row >= rows_ || column >= columns_) throw std::out_of_range("Matrix index out of range"); return row * columns_ + column; }
  std::size_t rows_, columns_; DynamicArray<T> values_;
};

template <typename T>
class SparseMatrix {
 public:
  SparseMatrix(std::size_t rows, std::size_t columns) : rows_(rows), columns_(columns) {}
  void set(std::size_t row, std::size_t column, T value) { const auto key = offset(row, column); if (value == T{}) values_.erase(key); else values_.insert_or_assign(key, std::move(value)); }
  T at(std::size_t row, std::size_t column) const { const auto* value = values_.find(offset(row, column)); return value ? *value : T{}; }
  std::size_t rows() const noexcept { return rows_; }
  std::size_t columns() const noexcept { return columns_; }
  std::size_t nonzero_count() const noexcept { return values_.size(); }
 private:
  std::size_t offset(std::size_t row, std::size_t column) const { if (row >= rows_ || column >= columns_) throw std::out_of_range("SparseMatrix index out of range"); return row * columns_ + column; }
  std::size_t rows_, columns_; AssociativeArray<std::size_t, T> values_;
};

template <typename Key, typename Value>
class ParallelArray {
 public:
  void push_back(Key key, Value value) { keys_.push_back(std::move(key)); values_.push_back(std::move(value)); }
  std::size_t size() const noexcept { return keys_.size(); }
  bool empty() const noexcept { return keys_.empty(); }
  const Key& key_at(std::size_t index) const { return keys_.at(index); }
  const Value& value_at(std::size_t index) const { return values_.at(index); }
  Value& value_at(std::size_t index) { return values_.at(index); }
 private: DynamicArray<Key> keys_; DynamicArray<Value> values_;
};

}  // namespace datamunge::dstruct
