/// @file ellipsoid.cpp
/// @brief Positive triangular central cuts and one actual callback-driven ellipsoid optimizer.
#include <datamunge/algorithms/ellipsoid.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Validated geometry, scaled factor arithmetic and optional actual history capture.
namespace ellipsoid_detail {
/// @brief Add logical counts without wrapping.
/// @param a Existing count.
/// @param b Additional count.
/// @return Checked sum.
/// @throws std::length_error On size_t overflow.
std::size_t add(std::size_t a,std::size_t b){if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("ellipsoid count overflow");return a+b;}
/// @brief Multiply geometry counts without wrapping.
/// @param a First factor.
/// @param b Second factor.
/// @return Checked product.
/// @throws std::length_error On size_t overflow.
std::size_t multiply(std::size_t a,std::size_t b){if(a&&b>std::numeric_limits<std::size_t>::max()/a)throw std::length_error("ellipsoid geometry overflow");return a*b;}
/// @brief Complete logical geometry of a nonempty factored state.
struct Geometry {
    std::size_t n; ///< Dimension.
    std::size_t square; ///< All stored factor entries.
    std::size_t triangle; ///< Lower entries, including diagonal.
    std::size_t sample; ///< Sample history cost 1+n+n*n.
    std::size_t iteration; ///< Oracle history cost 1+n.
};
/// @brief Check all matrix/count limits before shape or history allocation.
/// @param n Positive dimension.
/// @return Complete representable counts.
/// @throws std::invalid_argument On zero dimension.
/// @throws std::length_error On matrix/container/count limits.
Geometry geometry(std::size_t n){
    if(!n)throw std::invalid_argument("ellipsoid needs a nonempty center");const auto square=multiply(n,n),next=add(n,1);
    if(n>std::vector<std::vector<double>>{}.max_size()||square>std::vector<double>{}.max_size())throw std::length_error("ellipsoid matrix exceeds container geometry");
    return {n,square,multiply(n,next)/2,add(add(1,n),square),next};
}
/// @brief Validate center coordinates before callback access or copying.
/// @param center Borrowed coordinates.
/// @throws std::invalid_argument On a nonfinite coordinate.
void validate_center(const std::vector<double>& center){for(double x:center)if(!std::isfinite(x))throw std::invalid_argument("ellipsoid center must be finite");}
/// @brief Validate every state entry, including row shape and stored upper zeros.
/// @param state Borrowed mutable public state.
/// @return Checked geometry.
/// @throws std::invalid_argument On invalid center/factor shape, diagonal or values.
/// @throws std::length_error On geometry overflow.
Geometry validate_state(const EllipsoidState& state){
    const auto size=geometry(state.center.size());validate_center(state.center);
    if(state.lower.size()!=size.n)throw std::invalid_argument("ellipsoid factor row count mismatch");
    for(std::size_t i=0;i<size.n;++i){if(state.lower[i].size()!=size.n)throw std::invalid_argument("ellipsoid factor column count mismatch");
        for(std::size_t j=0;j<size.n;++j){const auto x=state.lower[i][j];if(!std::isfinite(x)||(j>i&&x!=0)||(j==i&&x<=0))throw std::invalid_argument("ellipsoid needs finite lower factor with positive diagonal");}}
    return size;
}
/// @brief Validate scalar parameters and both callbacks before any user call.
/// @param f Objective callback.
/// @param subgradient Subgradient callback.
/// @param center Initial coordinates.
/// @param radius Initial positive radius.
/// @param max_iter Nonnegative attempted-iteration cap.
/// @param tol Nonnegative finite absolute objective tolerance.
/// @return Checked state/history geometry.
/// @throws std::invalid_argument On invalid parameters or callbacks.
/// @throws std::length_error On geometry limits.
Geometry validate_arguments(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
    const std::vector<double>& center,double radius,int max_iter,double tol){
    if(!f||!subgradient||!std::isfinite(radius)||radius<=0||max_iter<0||!std::isfinite(tol)||tol<0)throw std::invalid_argument("invalid ellipsoid callbacks/radius/iteration cap/tolerance");
    const auto size=geometry(center.size());validate_center(center);return size;
}
/// @brief Validate oracle vector shape and all entries before indexing.
/// @param g Borrowed returned gradient.
/// @param n Expected dimension.
/// @throws std::invalid_argument On shape or nonfinite entries.
void validate_gradient(const std::vector<double>& g,std::size_t n){
    if(g.size()!=n)throw std::invalid_argument("ellipsoid subgradient dimension mismatch");
    for(auto x:g)if(!std::isfinite(x))throw std::invalid_argument("ellipsoid subgradient must be finite");
}
/// @brief Convert representable finite arithmetic to the public scalar type.
/// @param x Computed extended scalar.
/// @return Finite double conversion.
/// @throws std::overflow_error On nonfinite or out-of-double-range results.
double finite(long double x){
    if(!std::isfinite(x)||std::abs(x)>std::numeric_limits<double>::max())throw std::overflow_error("ellipsoid arithmetic exceeds finite double range");return static_cast<double>(x);
}
/// @brief Preserve strictly positive factor/support values through conversion.
/// @param x Positive extended scalar.
/// @return Strictly positive finite double.
/// @throws std::overflow_error On overflow.
/// @throws std::runtime_error If a positive value is lost or nonpositive.
double positive(long double x){const auto y=finite(x);if(y<=0)throw std::runtime_error("ellipsoid positive factor/support lost representability");return y;}
/// @brief Measure natural log volume from validated actual diagonal values.
/// @param state Valid factor/center.
/// @return Finite natural log volume.
/// @throws std::overflow_error On nonfinite conversion.
double log_volume(const EllipsoidState& state){
    const long double n=state.center.size();long double result=n/2*std::log(std::numbers::pi_v<long double>)-std::lgamma(n/2+1);
    for(std::size_t i=0;i<state.center.size();++i)result+=std::log(static_cast<long double>(state.lower[i][i]));return finite(result);
}
/// @brief Scaled actual projection in factor coordinates, without forming P or squared norms.
struct Direction {
    std::vector<long double> unit; ///< Unit vector proportional to L^T*g for a nonzero gradient.
    double support=0; ///< Finite represented norm ||L^T*g||.
    bool zero=true; ///< Exact all-zero oracle vector.
};
/// @brief Compute a robust normalized projection and its absolute objective support.
/// @param state Valid positive lower factor.
/// @param g Valid finite oracle vector.
/// @return Exact-zero flag or unit direction and positive support.
/// @throws std::overflow_error On nonfinite projection/support.
/// @throws std::runtime_error On lost positive representability.
Direction direction(const EllipsoidState& state,const std::vector<double>& g){
    double scale=0;for(double x:g)scale=std::max(scale,std::abs(x));if(scale==0)return {};
    Direction out;out.zero=false;out.unit.resize(g.size());long double norm=0;
    for(std::size_t j=0;j<g.size();++j){long double value=0;for(std::size_t i=j;i<g.size();++i)value+=static_cast<long double>(state.lower[i][j])*(static_cast<long double>(g[i])/scale);
        if(!std::isfinite(value))throw std::overflow_error("ellipsoid factor projection overflow");out.unit[j]=value;norm=std::hypot(norm,value);}
    if(!std::isfinite(norm))throw std::overflow_error("ellipsoid support norm overflow");
    if(norm<=0)throw std::runtime_error("ellipsoid nonzero support vanished numerically");out.support=positive(static_cast<long double>(scale)*norm);
    for(auto& x:out.unit)x/=norm;return out;
}
/// @brief Form the central cut through a triangular rank-one factor product.
/// @param state Valid input ellipsoid.
/// @param d Nonzero computed unit projection.
/// @return Owned candidate and measured geometric quantities.
/// @throws std::overflow_error On nonfinite geometry.
/// @throws std::runtime_error On factor positivity/representability failure.
EllipsoidCut cut(const EllipsoidState& state,const Direction& d){
    const auto n=state.center.size();EllipsoidCut out;out.support_bound=d.support;out.log_volume_before=log_volume(state);
    out.state.center.resize(n);out.state.lower.assign(n,std::vector<double>(n,0));
    for(std::size_t i=0;i<n;++i){long double step=0;for(std::size_t j=0;j<=i;++j)step+=static_cast<long double>(state.lower[i][j])*d.unit[j];
        out.state.center[i]=finite(static_cast<long double>(state.center[i])-step/(n==1?2:static_cast<long double>(n)+1));out.center_changed|=out.state.center[i]!=state.center[i];}
    if(n==1)out.state.lower[0][0]=positive(static_cast<long double>(state.lower[0][0])/2);
    else{
        const long double nd=n,b=2/(nd+1),scale=std::sqrt(nd*nd/(nd*nd-1));std::vector<long double> diagonal(n),q(n);long double cumulative=0,previous=1;
        for(std::size_t j=0;j<n;++j){cumulative+=d.unit[j]*d.unit[j];const auto s=1-b*cumulative;
            if(!(s>0))throw std::runtime_error("ellipsoid unit-cut factor lost positivity");diagonal[j]=std::sqrt(s/previous);q[j]=b*d.unit[j]/std::sqrt(previous*s);previous=s;}
        // chol(I-b*u*u^T) has a diagonal plus rank-one lower part. Row suffix
        // products compute L*chol(I-b*u*u^T) in O(n^2), with no diagonal subtraction.
        for(std::size_t i=0;i<n;++i){long double tail=0;for(std::size_t j=i+1;j-->0;){const auto value=scale*(static_cast<long double>(state.lower[i][j])*diagonal[j]-q[j]*tail);
            out.state.lower[i][j]=i==j?positive(value):finite(value);tail+=static_cast<long double>(state.lower[i][j])*d.unit[j];}}
    }
    out.log_volume_after=log_volume(out.state);return out;
}
/// @brief Reserve full retained history before its next callback/copy.
/// @param trace Optional owned trace.
/// @param entries Additional logical entries.
/// @param cap Optional quota, zero disables it.
/// @throws std::length_error On quota or arithmetic overflow.
void capture(EllipsoidTrace* trace,std::size_t entries,std::size_t cap){
    if(!trace)return;const auto total=add(trace->captured_entries,entries);if(cap&&total>cap)throw std::length_error("ellipsoid capture limit exceeded");trace->captured_entries=total;
}
/// @brief Call and validate one actual objective evaluation.
/// @param f Caller callback.
/// @param x Borrowed center valid for this call.
/// @return Finite actual callback result.
/// @throws std::invalid_argument On nonfinite return; callback exceptions propagate.
double objective(const std::function<double(const std::vector<double>&)>& f,const std::vector<double>& x){
    const auto value=f(x);if(!std::isfinite(value))throw std::invalid_argument("ellipsoid objective must return finite values");return value;
}
/// @brief Shared callback/cut execution, optionally recording actual values without replay.
/// @param f Valid objective callback.
/// @param subgradient Valid oracle callback.
/// @param center Valid initial center.
/// @param radius Valid positive radius.
/// @param max_iter Valid nonnegative attempted-call cap.
/// @param tol Valid absolute objective-gap tolerance.
/// @param size Validated complete geometry.
/// @param trace Optional capture with initial sample reserved.
/// @param cap Optional retained history quota.
/// @return Owned outcome with honest reason; no history when trace is null.
/// @throws std::invalid_argument Invalid callback results.
/// @throws std::length_error Counters, containers or history quota exceed limits.
/// @throws std::overflow_error Nonfinite computed geometry.
/// @throws std::runtime_error Lost positive numerical representability; callbacks propagate.
EllipsoidOutcome core(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,
    const std::vector<double>& center,double radius,int max_iter,double tol,const Geometry& size,
    EllipsoidTrace* trace,std::size_t cap){
    EllipsoidOutcome out;out.state=ellipsoid_ball(center,radius);out.result.x=center;
    out.work.objective_calls=1;out.result.value=objective(f,out.state.center);
    if(trace)trace->samples.push_back({out.state,out.result.value,out.result.value,log_volume(out.state)});
    for(int attempt=0;attempt<max_iter;++attempt){
        capture(trace,size.iteration,cap);EllipsoidIteration record;record.state_index=out.work.updates;
        out.work.subgradient_calls=add(out.work.subgradient_calls,1);out.result.iterations=attempt+1;
        record.gradient=subgradient(out.state.center);validate_gradient(record.gradient,size.n);const auto d=direction(out.state,record.gradient);
        out.has_support_bound=true;out.last_support_bound=record.support_bound=d.support;
        if(d.zero){out.stop=EllipsoidStop::zero_subgradient;if(trace)trace->iterations.push_back(std::move(record));break;}
        out.work.support_products=add(out.work.support_products,size.triangle);
        if(d.support<=tol){out.stop=EllipsoidStop::support_tolerance;if(trace)trace->iterations.push_back(std::move(record));break;}
        auto next=cut(out.state,d);
        if(!next.center_changed){out.stop=EllipsoidStop::center_stagnation;if(trace)trace->iterations.push_back(std::move(record));break;}
        capture(trace,size.sample,cap);out.work.objective_calls=add(out.work.objective_calls,1);const auto value=objective(f,next.state.center);
        out.state=std::move(next.state);out.work.updates=add(out.work.updates,1);out.work.factor_entries_updated=add(out.work.factor_entries_updated,size.triangle);
        if(value<out.result.value){out.result.value=value;out.result.x=out.state.center;}
        if(trace){record.updated=true;trace->iterations.push_back(std::move(record));trace->samples.push_back({out.state,value,out.result.value,next.log_volume_after});}
    }
    return out;
}
} // namespace ellipsoid_detail

EllipsoidState ellipsoid_ball(const std::vector<double>& center,double radius){
    if(!std::isfinite(radius)||radius<=0)throw std::invalid_argument("ellipsoid radius must be positive and finite");
    const auto size=ellipsoid_detail::geometry(center.size());ellipsoid_detail::validate_center(center);
    EllipsoidState state;state.center=center;state.lower.assign(size.n,std::vector<double>(size.n,0));for(std::size_t i=0;i<size.n;++i)state.lower[i][i]=radius;return state;
}
double ellipsoid_log_volume(const EllipsoidState& state){(void)ellipsoid_detail::validate_state(state);return ellipsoid_detail::log_volume(state);}
double ellipsoid_log_volume_ratio(std::size_t dimension){
    if(!dimension)throw std::invalid_argument("ellipsoid volume ratio needs positive dimension");if(dimension==1)return -std::log(2.);
    const long double t=1/static_cast<long double>(dimension),square=t*t;long double power=t,sum=0;
    // log(rho)=-sum_{k>=1} n^(-(2k-1))/(2k(2k-1)); t<=1/2.
    for(unsigned k=1;k<=256;++k){const long double term=power/(static_cast<long double>(2*k)*(2*k-1));sum+=term;
        if(term<=std::numeric_limits<long double>::epsilon()*sum)break;power*=square;}
    return static_cast<double>(-sum);
}
EllipsoidCut ellipsoid_central_cut(const EllipsoidState& state,const std::vector<double>& gradient){
    const auto size=ellipsoid_detail::validate_state(state);ellipsoid_detail::validate_gradient(gradient,size.n);
    const auto d=ellipsoid_detail::direction(state,gradient);if(d.zero)throw std::invalid_argument("ellipsoid central cut needs nonzero gradient");return ellipsoid_detail::cut(state,d);
}
std::vector<graphics::DiagramPoint> ellipsoid_outline(const EllipsoidState& state,std::size_t segments,std::size_t max_segments){
    const auto size=ellipsoid_detail::validate_state(state);if(size.n!=2||segments<8)throw std::invalid_argument("ellipsoid outline needs dimension two and at least eight segments");
    const auto count=ellipsoid_detail::add(segments,1);if(count>std::vector<graphics::DiagramPoint>{}.max_size()||(max_segments&&segments>max_segments))throw std::length_error("ellipsoid outline sample limit exceeded");
    std::vector<graphics::DiagramPoint> points;points.reserve(count);
    for(std::size_t i=0;i<segments;++i){const long double angle=2*std::numbers::pi_v<long double>*i/segments,c=std::cos(angle),s=std::sin(angle);
        points.push_back({ellipsoid_detail::finite(static_cast<long double>(state.center[0])+state.lower[0][0]*c),ellipsoid_detail::finite(static_cast<long double>(state.center[1])+state.lower[1][0]*c+state.lower[1][1]*s)});}
    points.push_back(points.front());return points;
}
EllipsoidOutcome ellipsoid_minimize_work(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,const std::vector<double>& center,double radius,int max_iter,double tol){
    const auto size=ellipsoid_detail::validate_arguments(f,subgradient,center,radius,max_iter,tol);return ellipsoid_detail::core(f,subgradient,center,radius,max_iter,tol,size,nullptr,0);
}
EllipsoidTrace ellipsoid_minimize_trace(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,const std::vector<double>& center,double radius,int max_iter,double tol,std::size_t max_entries){
    const auto size=ellipsoid_detail::validate_arguments(f,subgradient,center,radius,max_iter,tol);EllipsoidTrace trace;ellipsoid_detail::capture(&trace,size.sample,max_entries);
    trace.outcome=ellipsoid_detail::core(f,subgradient,center,radius,max_iter,tol,size,&trace,max_entries);return trace;
}
EllipsoidResult ellipsoid_minimize(const std::function<double(const std::vector<double>&)>& f,
    const std::function<std::vector<double>(const std::vector<double>&)>& subgradient,std::vector<double> center,double radius,int max_iter,double tol){
    return ellipsoid_minimize_work(f,subgradient,center,radius,max_iter,tol).result;
}
} // namespace datamunge::algorithms
