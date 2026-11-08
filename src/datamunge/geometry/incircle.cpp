/// @file
/// @brief Exact four-coordinate determinant monomials accumulated as bounded dyadic integers.
#include <datamunge/geometry/incircle.hpp>
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Binary64 decode and bounded unsigned arithmetic for the 48 expanded monomials.
namespace {
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53 && std::numeric_limits<double>::max_exponent==1024,"incircle requires IEEE binary64");
/// @brief Magnitude in units 2^-4296; 8448 bits exceed the 8398-bit 48-term bound.
using Magnitude=std::array<std::uint32_t,264>;
/// @brief At most 53 mantissa bits times a power of two, with an independent sign.
struct Dyadic {
  std::uint64_t mantissa; ///< At most 53 bits, zero for either signed zero.
  int exponent; ///< Binary exponent in [-1074,971].
  bool negative; ///< Stored sign bit.
};
/// @brief Decode a previously validated finite double exactly.
/// @param v Finite binary64 coordinate.
/// @return Sign, integer mantissa and binary exponent.
Dyadic decode(double v) {
  std::uint64_t bits;std::memcpy(&bits,&v,8);const auto e=static_cast<unsigned>((bits>>52)&2047);
  return {(bits&0xfffffffffffffULL)|(e?0x10000000000000ULL:0),e?static_cast<int>(e)-1075:-1074,bool(bits>>63)};
}
/// @brief Add one signed product of four decoded coordinates to the appropriate sum.
/// Each product has at most 212 mantissa bits and shift at most 8180, hence
/// 8392 bits; adding at most 48 terms needs at most 8398, below 8448.
/// @param positive Positive algebraic magnitude accumulator.
/// @param negative Negative algebraic magnitude accumulator.
/// @param factors Four decoded factors, including two copies of one squared coordinate.
/// @param subtract Algebraic determinant coefficient is negative when true.
void term(Magnitude& positive,Magnitude& negative,const std::array<Dyadic,4>& factors,bool subtract) {
  std::array<std::uint32_t,8> product{};product[0]=1;std::size_t length=1;int shift=4296;bool sign=subtract;
  for(const auto& f:factors) {
    if(!f.mantissa)return;shift+=f.exponent;sign^=f.negative;
    const std::array<std::uint32_t,2> words{static_cast<std::uint32_t>(f.mantissa),static_cast<std::uint32_t>(f.mantissa>>32)};
    std::array<std::uint32_t,8> next{};
    for(std::size_t i=0;i<length;++i) {
      std::uint64_t carry=0;
      for(std::size_t j=0;j<2;++j) {
        assert(i+j<8);const auto v=std::uint64_t(product[i])*words[j]+next[i+j]+carry;next[i+j]=static_cast<std::uint32_t>(v);carry=v>>32;
      }
      if(carry) {assert(i+2<8);next[i+2]=static_cast<std::uint32_t>(carry);}
    }
    product=next;length=std::min<std::size_t>(8,length+2);while(length>1 && !product[length-1])--length;
  }
  auto& sum=sign?negative:positive;const auto base=static_cast<unsigned>(shift)/32,bits=static_cast<unsigned>(shift)%32;
  std::uint64_t carry=0;
  for(std::size_t offset=0;offset<=length || carry;++offset) {
    std::uint32_t word=offset<length?static_cast<std::uint32_t>(std::uint64_t(product[offset])<<bits):0;
    if(bits && offset && offset<=length)word|=product[offset-1]>>(32-bits);
    assert(base+offset<sum.size());const auto v=std::uint64_t(sum[base+offset])+word+carry;sum[base+offset]=static_cast<std::uint32_t>(v);carry=v>>32;
  }
}
} // namespace
int incircle_sign(const Point2D& a,const Point2D& b,const Point2D& c,const Point2D& d) {
  const std::array<Point2D,4> points{a,b,c,d};std::array<std::array<Dyadic,2>,4> coordinates;
  for(std::size_t r=0;r<4;++r) {
    if(!std::isfinite(points[r].x)||!std::isfinite(points[r].y))throw std::invalid_argument("incircle: nonfinite coordinate");
    coordinates[r]={decode(points[r].x),decode(points[r].y)};
  }
  Magnitude positive{},negative{};std::array<unsigned,4> permutation{0,1,2,3};
  do {
    std::array<unsigned,4> row_for_column{};unsigned inversions=0;
    for(unsigned r=0;r<4;++r) {row_for_column[permutation[r]]=r;for(unsigned s=0;s<r;++s)inversions+=permutation[s]>permutation[r];}
    for(unsigned square=0;square<2;++square) {
      const auto q=coordinates[row_for_column[2]][square];
      term(positive,negative,{coordinates[row_for_column[0]][0],coordinates[row_for_column[1]][1],q,q},inversions%2);
    }
  }while(std::next_permutation(permutation.begin(),permutation.end()));
  for(std::size_t i=positive.size();i-->0;) {if(positive[i]>negative[i])return 1;if(positive[i]<negative[i])return -1;}
  return 0;
}
} // namespace datamunge::geometry
