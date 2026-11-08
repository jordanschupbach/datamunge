/// @file orientation.cpp
/// @brief Exact binary64 determinant signs via bounded aligned integer products.
#include <datamunge/geometry/orientation.hpp>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Binary64-specific fixed integer arithmetic, independent of floating evaluation.
namespace {
static_assert(sizeof(double)==sizeof(std::uint64_t) && std::numeric_limits<double>::is_iec559 &&
    std::numeric_limits<double>::digits==53 && std::numeric_limits<double>::max_exponent==1024,
    "exact orientation requires IEEE binary64 double");
/// @brief Unsigned magnitude scaled by 2^-2148, enough for six full-range products.
/// A product has at most 106+4090=4196 bits; six terms need at most 4199,
/// below this accumulator's 4224 bits. Therefore carry cannot escape its bounds.
using Magnitude = std::array<std::uint32_t,132>;
/// @brief Finite double decoded as sign times mantissa times 2^exponent.
struct Dyadic {
    std::uint64_t mantissa{}; ///< At most 53 bits; zero covers both signed zeros.
    int exponent{};           ///< In [-1074,971], including subnormals.
    bool negative{};          ///< Stored sign bit.
};
/// @brief Decode one validated finite IEEE binary64 value without floating arithmetic.
/// @param value Finite value.
/// @return Exact signed dyadic representation.
Dyadic decode(double value) {
    std::uint64_t bits;std::memcpy(&bits,&value,sizeof(bits));
    const auto exponent=static_cast<unsigned>((bits>>52)&2047);
    return {(bits&0xfffffffffffffULL)|(exponent?0x10000000000000ULL:0),
        exponent?static_cast<int>(exponent)-1075:-1074,bool(bits>>63)};
}
/// @brief Add one nonnegative mantissa product at its exact dyadic bit offset.
/// @param sum Bounded accumulator; the six-term determinant bound prevents overflow.
/// @param a First decoded coordinate.
/// @param b Second decoded coordinate.
void add_product(Magnitude& sum,const Dyadic& a,const Dyadic& b) {
    if (!a.mantissa || !b.mantissa) return;
    const std::array<std::uint32_t,2> x{{static_cast<std::uint32_t>(a.mantissa),static_cast<std::uint32_t>(a.mantissa>>32)}};
    const std::array<std::uint32_t,2> y{{static_cast<std::uint32_t>(b.mantissa),static_cast<std::uint32_t>(b.mantissa>>32)}};
    std::array<std::uint32_t,4> product{};
    for (std::size_t i=0;i<2;++i) {
        std::uint64_t carry=0;
        for (std::size_t j=0;j<2;++j) {
            const auto value=std::uint64_t(x[i])*y[j]+product[i+j]+carry;
            product[i+j]=static_cast<std::uint32_t>(value);carry=value>>32;
        }
        product[i+2]=static_cast<std::uint32_t>(carry);
    }
    const auto shift=static_cast<unsigned>(a.exponent+b.exponent+2148);
    const auto base=shift/32,bits=shift%32;
    std::uint64_t carry=0;
    for (std::size_t offset=0;offset<5 || carry;++offset) {
        std::uint32_t word=offset<4?static_cast<std::uint32_t>(std::uint64_t(product[offset])<<bits):0;
        if (bits && offset>0 && offset<=4) word|=product[offset-1]>>(32-bits);
        assert(base+offset<sum.size());
        const auto value=std::uint64_t(sum[base+offset])+word+carry;
        sum[base+offset]=static_cast<std::uint32_t>(value);carry=value>>32;
    }
}
} // namespace
int orientation_sign(const Point2D& a,const Point2D& b,const Point2D& c) {
    for (double v:{a.x,a.y,b.x,b.y,c.x,c.y})
        if (!std::isfinite(v)) throw std::invalid_argument("orientation: nonfinite coordinate");
    const auto ax=decode(a.x),ay=decode(a.y),bx=decode(b.x),by=decode(b.y),cx=decode(c.x),cy=decode(c.y);
    Magnitude positive{},negative{};
    /// @brief Accumulate one signed term of the expanded determinant exactly.
    /// @param x First decoded factor.
    /// @param y Second decoded factor.
    /// @param subtract Whether the algebraic coefficient is negative.
    const auto term=[&](const Dyadic& x,const Dyadic& y,bool subtract) {
        add_product((x.negative!=y.negative)!=subtract?negative:positive,x,y);
    };
    term(bx,cy,false);term(bx,ay,true);term(ax,cy,true);
    term(by,cx,true);term(by,ax,false);term(ay,cx,false);
    for (std::size_t i=positive.size();i-->0;) {
        if (positive[i]>negative[i]) return 1;
        if (positive[i]<negative[i]) return -1;
    }
    return 0;
}
} // namespace datamunge::geometry
