/// @file dtw.cpp
/// @brief Shared rolling-row absolute-loss DTW, original tie witnesses and actual cell capture.
#include <datamunge/geometry/dtw.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Internal checked geometry, complete input validation and the shared recurrence.
namespace {
/// @brief Exact checked positive matrix geometry and logical capture count.
struct Shape {
 std::size_t n; ///< Positive first length fitting int.
 std::size_t m; ///< Positive second length fitting int.
 std::size_t cells; ///< Checked n*m positive cells.
 std::size_t entries; ///< Checked n+m+2*n*m capture entries.
};
/// @brief Check signed path indices and all matrix/capture arithmetic before allocation.
/// @param n First length.
/// @param m Second length.
/// @return Checked positive geometry.
/// @throws std::invalid_argument For zero length.
/// @throws std::length_error For index/product/capture overflow.
Shape geometry(std::size_t n,std::size_t m) {
 if(!n||!m)throw std::invalid_argument("DTW inputs must be nonempty");
 const auto max=std::numeric_limits<std::size_t>::max();
 if(n>static_cast<std::size_t>(std::numeric_limits<int>::max())||m>static_cast<std::size_t>(std::numeric_limits<int>::max())||n>max/m||n>max-m)
  throw std::length_error("DTW index or product bounds exceeded");
 const auto cells=n*m;if(cells>(max-n-m)/2)throw std::length_error("DTW capture geometry overflow");
 return {n,m,cells,n+m+2*cells};
}
/// @brief Validate every finite scalar after complete shape arithmetic.
/// @param a First borrowed input.
/// @param b Second borrowed input.
/// @return Checked geometry.
/// @throws std::invalid_argument For empty/nonfinite inputs.
/// @throws std::length_error For geometry bounds.
Shape validate(const std::vector<double>& a,const std::vector<double>& b) {
 const auto s=geometry(a.size(),b.size());
 for(const auto* x:{&a,&b})for(double v:*x)if(!std::isfinite(v))throw std::invalid_argument("DTW inputs must be finite");
 return s;
}
/// @brief Execute one validated recurrence, writing optional actual positive-cell evidence.
/// @param a Valid finite input.
/// @param b Valid finite input.
/// @param s Checked dimensions/capture geometry.
/// @param trace Optional local owned capture; input/cap checks already succeeded.
/// @return Actual finite result/work.
/// @throws std::length_error For container size limits.
/// @throws std::overflow_error For nonfinite final optimal cost.
/// @throws std::logic_error If an internal finite-path predecessor invariant fails.
/// @throws std::bad_alloc For working/capture/path allocations.
DTWComputation run(const std::vector<double>& a,const std::vector<double>& b,const Shape& s,DTWTrace* trace) {
 std::vector<DTWPredecessor> ordinary;
 auto& directions=trace?trace->predecessors:ordinary;
 directions.resize(s.cells,DTWPredecessor::unrepresentable);
 if(trace)trace->cumulative.resize(s.cells);
 const double infinity=std::numeric_limits<double>::infinity();
 std::vector<double> previous(s.m+1,infinity),current(s.m+1,infinity);previous[0]=0;
 DTWComputation out;
 for(std::size_t i=1;i<=s.n;++i){current[0]=infinity;
  for(std::size_t j=1;j<=s.m;++j){
   const auto k=(i-1)*s.m+j-1;const double diagonal=previous[j-1],up=previous[j],left=current[j-1];
   const double value=std::abs(a[i-1]-b[j-1])+std::min({up,left,diagonal});current[j]=value;
   ++out.work.cells;out.work.predecessor_reads+=3;
   if(std::isfinite(value))directions[k]=(i==1&&j==1)?DTWPredecessor::origin
      :diagonal<=up&&diagonal<=left?DTWPredecessor::diagonal
      :up<=left?DTWPredecessor::advance_a:DTWPredecessor::advance_b;
   if(trace)trace->cumulative[k]=value;
  }
  previous.swap(current);
 }
 if(!std::isfinite(previous[s.m]))throw std::overflow_error("DTW optimal cost exceeds finite double range");
 out.result.distance=previous[s.m];std::size_t i=s.n-1,j=s.m-1;
 for(;;){out.result.path_a.push_back(static_cast<int>(i));out.result.path_b.push_back(static_cast<int>(j));++out.work.traceback_steps;
  if(i==0&&j==0)break;const auto p=directions[i*s.m+j];
  if(p==DTWPredecessor::diagonal&&i&&j){--i;--j;}
  else if(p==DTWPredecessor::advance_a&&i)--i;
  else if(p==DTWPredecessor::advance_b&&j)--j;
  else throw std::logic_error("DTW finite traceback invariant failed");
 }
 std::reverse(out.result.path_a.begin(),out.result.path_a.end());std::reverse(out.result.path_b.begin(),out.result.path_b.end());return out;
}
} // namespace
std::size_t dtw_trace_entries(std::size_t n,std::size_t m){return geometry(n,m).entries;}
DTWComputation dynamic_time_warping_work(const std::vector<double>& a,const std::vector<double>& b){const auto s=validate(a,b);return run(a,b,s,nullptr);}
DTWTrace dynamic_time_warping_trace(const std::vector<double>& a,const std::vector<double>& b,std::size_t max_entries){
 const auto s=validate(a,b);if(max_entries&&s.entries>max_entries)throw std::length_error("DTW complete capture cap exceeded");
 DTWTrace trace;trace.a=a;trace.b=b;trace.captured_entries=s.entries;
 auto c=run(a,b,s,&trace);trace.result=std::move(c.result);trace.work=c.work;return trace;
}
double dtw_path_cost(const std::vector<double>& a,const std::vector<double>& b,const std::vector<int>& pa,const std::vector<int>& pb){
 const auto s=validate(a,b);
 if(pa.empty()||pa.size()!=pb.size())throw std::invalid_argument("DTW witness paths must be nonempty and parallel");
 for(std::size_t k=0;k<pa.size();++k){if(pa[k]<0||pb[k]<0||static_cast<std::size_t>(pa[k])>=s.n||static_cast<std::size_t>(pb[k])>=s.m)throw std::invalid_argument("DTW witness index outside input");
  if(k){const auto di=static_cast<std::int64_t>(pa[k])-pa[k-1],dj=static_cast<std::int64_t>(pb[k])-pb[k-1];
   if(di<0||dj<0||di>1||dj>1||di+dj==0)throw std::invalid_argument("DTW witness step must advance one or both axes by one");}}
 if(pa.front()!=0||pb.front()!=0||pa.back()!=static_cast<int>(s.n-1)||pb.back()!=static_cast<int>(s.m-1))throw std::invalid_argument("DTW witness endpoints must cover both complete inputs");
 double cost=0;for(std::size_t k=0;k<pa.size();++k){cost+=std::abs(a[pa[k]]-b[pb[k]]);if(!std::isfinite(cost))throw std::overflow_error("DTW witness cost exceeds finite double range");}return cost;
}
graphics::RPlot plot_dtw_alignment(const std::vector<double>& a,const std::vector<double>& b,const DTWResult& r,double offset,std::size_t max_pairs){
 const auto cost=dtw_path_cost(a,b,r.path_a,r.path_b);
 if(!std::isfinite(r.distance)||r.distance!=cost||!std::isfinite(offset)||offset<0)throw std::invalid_argument("DTW drawing requires exact witness cost and finite nonnegative offset");
 for(double v:a)if(!std::isfinite(v+offset))throw std::overflow_error("DTW first display offset overflows");
 for(double v:b)if(!std::isfinite(v-offset))throw std::overflow_error("DTW second display offset overflows");
 if(max_pairs&&r.path_a.size()>max_pairs)throw std::length_error("DTW complete drawing pair cap exceeded");
 std::vector<double> xa(a.size()),ya(a.size()),xb(b.size()),yb(b.size()),x0,y0,x1,y1;
 for(std::size_t i=0;i<a.size();++i){xa[i]=i;ya[i]=a[i]+offset;}
 for(std::size_t j=0;j<b.size();++j){xb[j]=j;yb[j]=b[j]-offset;}
 for(std::size_t k=0;k<r.path_a.size();++k){x0.push_back(r.path_a[k]);y0.push_back(ya[r.path_a[k]]);x1.push_back(r.path_b[k]);y1.push_back(yb[r.path_b[k]]);}
 auto plot=graphics::RPlot::create();plot.segments(x0,y0,x1,y1,{203,213,225},1).line(xa,ya,"series a",{37,99,235},2).line(xb,yb,"series b",{220,38,38},2)
   .legend({"series a","series b","alignment"},{{37,99,235},{220,38,38},{203,213,225}})
   .x_label("sample index").y_label("value (display offset only)").title("DTW scalar alignment");return plot;
}
} // namespace datamunge::geometry
