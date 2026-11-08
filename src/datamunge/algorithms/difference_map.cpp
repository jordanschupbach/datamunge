/// @file difference_map.cpp
/// @brief Shared checked difference-map execution, actual histories and elementary projections.
#include <datamunge/algorithms/difference_map.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
namespace datamunge::algorithms {
/// @brief Local vector-domain and computed-arithmetic checks, without shared state.
namespace {
/// @brief Reject wrong dimensions or nonfinite vector entries before indexing.
/// @param v Borrowed input/output vector.
/// @param n Required dimension.
/// @param who Error-context label.
/// @throws std::invalid_argument For a shape mismatch or any nonfinite entry.
void finite_vector(const std::vector<double>& v,std::size_t n,const char* who) {
  if(v.size()!=n)throw std::invalid_argument(std::string(who)+": projection dimension mismatch");
  for(double value:v)if(!std::isfinite(value))throw std::invalid_argument(std::string(who)+": nonfinite vector");
}
/// @brief Validate a computed binary64 intermediate.
/// @param value Computed coordinate, reciprocal or norm.
/// @return The unchanged finite value.
/// @throws std::overflow_error If value is nonfinite.
double computed(double value) {
  if(!std::isfinite(value))throw std::overflow_error("difference_map: nonfinite computed arithmetic");
  return value;
}
/// @brief Check a wide projection intermediate before narrowing to binary64.
/// @param value Computed long-double coordinate.
/// @return Finite narrowed value.
/// @throws std::overflow_error For nonfinite wide or narrowed arithmetic.
double narrow(long double value) {
  if(!std::isfinite(value))throw std::overflow_error("projection: nonfinite wide arithmetic");
  return computed(static_cast<double>(value));
}
} // namespace
std::size_t difference_map_history_cost(std::size_t dimension) {
  if(dimension>(std::numeric_limits<std::size_t>::max()-1)/8)throw std::length_error("difference_map: history cost overflow");
  return 1+8*dimension;
}
namespace difference_map_detail {
DifferenceMapResult run(const DifferenceProjection& a,const DifferenceProjection& b,std::vector<double> x,double beta,int max_iter,double tol,DifferenceMapTrace* trace,std::size_t max_history_entries) {
  if(!a || !b || !std::isfinite(beta) || beta==0 || max_iter<0 || !std::isfinite(tol) || tol<0)
    throw std::invalid_argument("difference_map: empty projection or invalid controls");
  finite_vector(x,x.size(),"initial state");const double inverse=computed(1.0/beta);
  const std::size_t n=x.size(),cost=trace&&max_iter>0?difference_map_history_cost(n):0;
  DifferenceMapResult result;DifferenceMapWork work;
  if(trace){trace->initial=x;trace->beta=beta;trace->tolerance=tol;trace->max_iterations=max_iter;}
  if(max_iter==0){++work.projection_b_calls;result.solution=b(x);finite_vector(result.solution,n,"B(x0)");}
  for(int it=0;it<max_iter;++it) {
    if(n>std::numeric_limits<std::size_t>::max()-work.coordinate_updates ||
       work.projection_a_calls>std::numeric_limits<std::size_t>::max()-2 || work.projection_b_calls>std::numeric_limits<std::size_t>::max()-2)
      throw std::length_error("difference_map: actual work overflow");
    if(trace && (cost>std::numeric_limits<std::size_t>::max()-work.history_entries ||
       (max_history_entries && (work.history_entries>max_history_entries || cost>max_history_entries-work.history_entries)) || trace->steps.size()==trace->steps.max_size()))
      throw std::length_error("difference_map: actual history cap or capacity");
    DifferenceMapStep step;step.iteration=it+1;if(trace)step.x_before=x;
    ++work.projection_a_calls;step.projection_a=a(x);finite_vector(step.projection_a,n,"A(x)");
    ++work.projection_b_calls;step.projection_b=b(x);finite_vector(step.projection_b,n,"B(x)");
    step.f_a.resize(n);step.f_b.resize(n);
    for(std::size_t i=0;i<n;++i) {
      step.f_a[i]=computed((1-inverse)*step.projection_a[i]+inverse*x[i]);
      step.f_b[i]=computed((1+inverse)*step.projection_b[i]-inverse*x[i]);
    }
    ++work.projection_a_calls;step.estimate_a=a(step.f_b);finite_vector(step.estimate_a,n,"A(f_B)");
    ++work.projection_b_calls;step.estimate_b=b(step.f_a);finite_vector(step.estimate_b,n,"B(f_A)");
    step.x_after=x;
    for(std::size_t i=0;i<n;++i) {
      const double gap=computed(step.estimate_a[i]-step.estimate_b[i]),delta=computed(beta*gap);
      step.x_after[i]=computed(x[i]+delta);
      step.gap_norm=computed(std::hypot(step.gap_norm,gap));step.step_norm=computed(std::hypot(step.step_norm,delta));
    }
    work.coordinate_updates+=n;x=step.x_after;
    result.solution=step.estimate_b;result.iterations=it+1;result.residual=step.step_norm;result.converged=step.step_norm<tol;
    if(trace){work.history_entries+=cost;trace->steps.push_back(std::move(step));}
    if(result.converged)break;
  }
  if(trace){trace->result=result;trace->final_iterate=std::move(x);trace->work=work;trace->stop=result.converged?DifferenceMapStop::step_tolerance:DifferenceMapStop::iteration_limit;}
  return result;
}
} // namespace difference_map_detail
DifferenceMapTrace difference_map_trace(DifferenceProjection a,DifferenceProjection b,std::vector<double> x0,double beta,int max_iter,double tol,std::size_t max_history_entries) {
  DifferenceMapTrace trace;difference_map_detail::run(a,b,std::move(x0),beta,max_iter,tol,&trace,max_history_entries);return trace;
}
std::vector<double> project_affine_hyperplane(const std::vector<double>& point,const std::vector<double>& normal,double offset) {
  if(point.empty() || !std::isfinite(offset))throw std::invalid_argument("project_affine_hyperplane: empty dimension or nonfinite offset");
  finite_vector(point,point.size(),"point");finite_vector(normal,point.size(),"normal");
  double scale=0;for(double v:normal)scale=std::max(scale,std::abs(v));
  if(scale==0)throw std::invalid_argument("project_affine_hyperplane: zero normal");
  long double dot=0,squared=0;
  for(std::size_t i=0;i<point.size();++i){const long double u=static_cast<long double>(normal[i])/scale;dot+=u*point[i];squared+=u*u;}
  const long double factor=(dot-static_cast<long double>(offset)/scale)/squared;
  if(!std::isfinite(factor))throw std::overflow_error("project_affine_hyperplane: nonfinite factor");
  std::vector<double> out(point.size());
  for(std::size_t i=0;i<point.size();++i)out[i]=narrow(static_cast<long double>(point[i])-factor*(static_cast<long double>(normal[i])/scale));
  return out;
}
std::vector<double> project_finite_set(const std::vector<double>& point,const std::vector<std::vector<double>>& points) {
  if(points.empty())throw std::invalid_argument("project_finite_set: empty set");
  finite_vector(point,point.size(),"point");for(const auto& p:points)finite_vector(p,point.size(),"finite set");
  std::size_t best=0;long double best_distance=std::numeric_limits<long double>::infinity();
  for(std::size_t k=0;k<points.size();++k) {
    long double distance=0;for(std::size_t i=0;i<point.size();++i)distance=std::hypot(distance,static_cast<long double>(point[i])-points[k][i]);
    if(!std::isfinite(distance))throw std::overflow_error("project_finite_set: nonfinite distance");
    if(distance<best_distance){best_distance=distance;best=k;}
  }
  return points[best];
}
} // namespace datamunge::algorithms
