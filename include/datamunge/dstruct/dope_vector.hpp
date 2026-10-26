#pragma once
#include <cstddef>
#include <stdexcept>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { template <typename T> class DopeVector { public: DopeVector(std::size_t r,std::size_t c):r_(r),c_(c),data_(r*c){} T& at(std::size_t r,std::size_t c){return data_[index(r,c)];}private:std::size_t index(std::size_t r,std::size_t c)const{if(r>=r_||c>=c_)throw std::out_of_range("DopeVector index");return r*c_+c;}std::size_t r_,c_;DynamicArray<T> data_;};}
