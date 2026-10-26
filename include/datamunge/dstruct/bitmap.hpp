#pragma once
#include <cstddef>
#include <stdexcept>
#include <datamunge/dstruct/foundational.hpp>
namespace datamunge::dstruct { class Bitmap { public: Bitmap(std::size_t width,std::size_t height):width_(width),height_(height),bits_(width*height){} void set(std::size_t x,std::size_t y,bool value=true){bits_.set(index(x,y),value);} bool at(std::size_t x,std::size_t y)const{return bits_.at(index(x,y));} std::size_t width()const{return width_;} std::size_t height()const{return height_;} private: std::size_t index(std::size_t x,std::size_t y)const{if(x>=width_||y>=height_)throw std::out_of_range("Bitmap index out of range");return y*width_+x;} std::size_t width_,height_;BitArray bits_;}; }
