#pragma once
#include <cstddef>
#include <stdexcept>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { template <typename T> class ControlTable { public: explicit ControlTable(std::size_t size):values_(size),enabled_(size){} void configure(std::size_t index,T value,bool enabled=true){check(index);values_[index]=std::move(value);enabled_.set(index,enabled);} void enable(std::size_t index,bool state=true){check(index);enabled_.set(index,state);} bool enabled(std::size_t index)const{check(index);return enabled_.at(index);} const T& at(std::size_t index)const{check(index);return values_[index];}
  /// @brief Number of entries, fixed at construction.
  /// @return The table size; valid indices are 0 .. size()-1.
  [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
  private:void check(std::size_t i)const{if(i>=values_.size())throw std::out_of_range("ControlTable index out of range");}DynamicArray<T> values_;BitArray enabled_;}; }
