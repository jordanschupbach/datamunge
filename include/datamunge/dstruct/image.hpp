#pragma once
#include <cstddef>
#include <stdexcept>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { template <typename Pixel> class Image { public: Image(std::size_t width,std::size_t height,Pixel fill=Pixel{}):width_(width),height_(height),pixels_(width*height,fill){} Pixel& at(std::size_t x,std::size_t y){return pixels_[index(x,y)];}const Pixel& at(std::size_t x,std::size_t y)const{return pixels_[index(x,y)];}std::size_t width()const{return width_;}std::size_t height()const{return height_;}private:std::size_t index(std::size_t x,std::size_t y)const{if(x>=width_||y>=height_)throw std::out_of_range("Image index out of range");return y*width_+x;}std::size_t width_,height_;DynamicArray<Pixel> pixels_;};}
