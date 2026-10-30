#pragma once
#include <cstddef>
#include <stdexcept>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { template <typename T> class DopeVector { public: DopeVector(std::size_t r,std::size_t c):r_(r),c_(c),data_(r*c){} T& at(std::size_t r,std::size_t c){return data_[index(r,c)];}
  /// @brief Read-only element access.
  /// @param r Row index.
  /// @param c Column index.
  /// @return The element at (r, c).
  /// @throws std::out_of_range If r >= rows() or c >= columns().
  [[nodiscard]] const T& at(std::size_t r, std::size_t c) const { return data_[index(r, c)]; }
  /// @brief Number of rows, fixed at construction.
  /// @return The row count.
  [[nodiscard]] std::size_t rows() const noexcept { return r_; }
  /// @brief Number of columns, fixed at construction.
  /// @return The column count.
  [[nodiscard]] std::size_t columns() const noexcept { return c_; }
  private:std::size_t index(std::size_t r,std::size_t c)const{if(r>=r_||c>=c_)throw std::out_of_range("DopeVector index");return r*c_+c;}std::size_t r_,c_;DynamicArray<T> data_;};}
