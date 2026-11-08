/// @file
/// @brief Widened DDA loop, actual sample diagnostics and overflow-free rational quantization.
#include <datamunge/algorithms/dda_line.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private endpoint geometry, unsigned products and common accumulating kernel.
namespace {
static_assert(std::numeric_limits<int>::digits<=31,"DDA supports int coordinates of at most 32 bits");
/// @brief Absolute value of a widened endpoint difference, always within uint32.
/// @param v Signed widened difference, never INT64_MIN.
/// @return Nonnegative magnitude.
std::uint64_t magnitude(std::int64_t v) noexcept {return static_cast<std::uint64_t>(v<0?-v:v);}
/// @brief Integer and signed fractional pieces of one exact endpoint sample.
struct RationalPart {
  std::int64_t base; ///< start plus signed integer quotient, within endpoint bounds.
  std::uint64_t remainder; ///< Nonnegative residual numerator, strictly less than denominator.
  int sign; ///< Direction -1 for negative delta, +1 otherwise; zero remainder ignores it.
};
/// @brief Split start+i*delta/N without any signed multiplication.
/// @param start Initial integer coordinate.
/// @param delta Widened signed endpoint difference.
/// @param i Sample index no greater than N.
/// @param n Major-axis steps, zero allowed only with i=0.
/// @return Exact integer part and signed residual; coincident samples have residual zero.
RationalPart part(int start,std::int64_t delta,std::uint64_t i,std::uint64_t n) noexcept {
  if(!n) return {start,0,1};
  const auto product=magnitude(delta)*i;const int sign=delta<0?-1:1;
  return {static_cast<std::int64_t>(start)+sign*static_cast<std::int64_t>(product/n),product%n,sign};
}
/// @brief Round a split exact sample, with true global signed ties away from zero.
/// @param p Exact quotient/remainder and direction.
/// @param n Denominator, zero only when remainder is zero.
/// @return Exact nearest integer, within the integer endpoints.
int quantize(const RationalPart& p,std::uint64_t n) noexcept {
  auto value=p.base;
  if(p.remainder && (2*p.remainder>n || (2*p.remainder==n && ((p.sign>0 && p.base>=0)||(p.sign<0 && p.base<=0))))) value+=p.sign;
  return static_cast<int>(value);
}
/// @brief Difference of a double or integer coordinate from a split rational sample.
/// @param value Actual accumulator or rounded integer, exactly represented if integer.
/// @param p Exact integer part and signed fractional numerator.
/// @param n Denominator; zero when coincident, with zero fractional term.
/// @return Approximate error, avoiding cancellation between large nearby ideal coordinates.
double difference(double value,const RationalPart& p,std::uint64_t n) noexcept {
  return (value-static_cast<double>(p.base))-(n?p.sign*(static_cast<double>(p.remainder)/static_cast<double>(n)):0);
}
/// @brief Round an actual accumulator safely before any conversion to int.
/// @param v Actual finite double accumulator, possibly drifted beyond endpoint bounds.
/// @return Nearest int with halfway ties away from zero.
/// @throws std::overflow_error Rounded value is nonfinite or outside int.
int rounded(double v) {
  const double r=std::round(v);
  if(!std::isfinite(r) || r<std::numeric_limits<int>::min() || r>std::numeric_limits<int>::max()) throw std::overflow_error("DDA rounded coordinate outside int");
  return static_cast<int>(r);
}
/// @brief Check complete point count against optional limit and container capacity.
/// @param count Inclusive sample count from widened geometry.
/// @param limit Optional zero-disabled point ceiling.
/// @param capacity Actual vector max_size.
/// @throws std::length_error Optional limit or capacity exceeded.
void capacity(std::uint64_t count,std::size_t limit,std::size_t capacity) {
  if((limit && count>limit) || count>capacity) throw std::length_error("DDA complete output exceeds point limit or capacity");
}
/// @brief Run the original accumulating recurrence with optional actual sample capture.
/// @tparam Record Retain exact-reference diagnostics and sample records when true.
/// @param x0 Initial horizontal coordinate.
/// @param y0 Initial vertical coordinate.
/// @param x1 Final horizontal coordinate.
/// @param y1 Final vertical coordinate.
/// @param limit Optional complete-point cap, zero disables.
/// @return Owned complete raster and optional actual samples, with no replay.
/// @throws std::length_error Point cap or vector capacity exceeded before allocation.
/// @throws std::overflow_error Rounded coordinate outside int.
/// @throws std::bad_alloc Allocation fails.
template<bool Record>
DdaTrace raster(int x0,int y0,int x1,int y1,std::size_t limit) {
  DdaTrace t;t.dx=static_cast<std::int64_t>(x1)-x0;t.dy=static_cast<std::int64_t>(y1)-y0;
  t.steps=std::max(magnitude(t.dx),magnitude(t.dy));const auto count=t.steps+1;
  capacity(count,limit,t.points.max_size());if constexpr(Record)capacity(count,limit,t.samples.max_size());
  t.points.reserve(static_cast<std::size_t>(count));if constexpr(Record)t.samples.reserve(static_cast<std::size_t>(count));
  if(t.steps) {t.x_increment=static_cast<double>(t.dx)/t.steps;t.y_increment=static_cast<double>(t.dy)/t.steps;}
  double x=x0,y=y0;
  for(std::uint64_t i=0;;++i) {
    const DDAPoint p{rounded(x),rounded(y)};t.points.push_back(p);
    if constexpr(Record) {
      const auto a=part(x0,t.dx,i,t.steps),b=part(y0,t.dy,i,t.steps);
      DdaSample s{i,x,y,p,{quantize(a,t.steps),quantize(b,t.steps)},difference(x,a,t.steps),difference(y,b,t.steps),difference(p.first,a,t.steps),difference(p.second,b,t.steps)};
      t.differing_samples+=p!=s.exact_point;t.roundings+=2;
      t.max_coordinate_error=std::max({t.max_coordinate_error,std::abs(s.x_error),std::abs(s.y_error)});
      t.max_accumulator_drift=std::max({t.max_accumulator_drift,std::abs(s.x_drift),std::abs(s.y_drift)});
      t.samples.push_back(s);
    }
    if(i==t.steps)break;
    x+=t.x_increment;y+=t.y_increment;if constexpr(Record)t.coordinate_additions+=2;
  }
  return t;
}
} // namespace
std::uint64_t dda_point_count(int x0,int y0,int x1,int y1) noexcept {
  return std::max(magnitude(static_cast<std::int64_t>(x1)-x0),magnitude(static_cast<std::int64_t>(y1)-y0))+1;
}
DDAPoint dda_exact_sample(int x0,int y0,int x1,int y1,std::uint64_t index) {
  const auto n=dda_point_count(x0,y0,x1,y1)-1;
  if(index>n) throw std::out_of_range("DDA sample index exceeds steps");
  return {quantize(part(x0,static_cast<std::int64_t>(x1)-x0,index,n),n),quantize(part(y0,static_cast<std::int64_t>(y1)-y0,index,n),n)};
}
std::vector<DDAPoint> dda_line(int x0,int y0,int x1,int y1) {return raster<false>(x0,y0,x1,y1,0).points;}
DdaTrace dda_line_trace(int x0,int y0,int x1,int y1,std::size_t max_points) {return raster<true>(x0,y0,x1,y1,max_points);}
std::vector<DDAPoint> dda_line_exact(int x0,int y0,int x1,int y1,std::size_t max_points) {
  const auto count=dda_point_count(x0,y0,x1,y1);const auto n=count-1;
  std::vector<DDAPoint> points;capacity(count,max_points,points.max_size());points.reserve(static_cast<std::size_t>(count));
  const auto dx=static_cast<std::int64_t>(x1)-x0,dy=static_cast<std::int64_t>(y1)-y0;
  for(std::uint64_t i=0;i<count;++i) points.emplace_back(quantize(part(x0,dx,i,n),n),quantize(part(y0,dy,i,n),n));
  return points;
}
} // namespace datamunge::algorithms
