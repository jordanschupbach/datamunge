/// @file d_star.cpp
/// @brief Exact bounded-label D* Lite with a preallocated indexed heap and actual repair capture.
#include <datamunge/algorithms/d_star.hpp>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Internal grid bounds, exact arithmetic and drawing validation.
namespace {
/// @brief Validated positive dimensions and safe flattened cell count.
struct Shape {
  int rows; ///< Positive rows, representable as int.
  int columns; ///< Positive columns, representable as int.
  std::size_t count; ///< Product at most INT_MAX.
};
/// @brief Validate dimensions and complete cell budget before allocation.
/// @param rows Positive rows.
/// @param columns Positive columns.
/// @param limit Zero disables only optional cell cap.
/// @return Validated dimensions/product.
/// @throws std::invalid_argument For nonpositive dimensions.
/// @throws std::length_error For count/implementation/cap bounds.
Shape dimensions(int rows,int columns,std::size_t limit) {
  if(rows<=0 || columns<=0)throw std::invalid_argument("D* Lite grid dimensions must be positive");
  if(rows>std::numeric_limits<int>::max()/columns)throw std::length_error("D* Lite cell count exceeds INT_MAX");
  const auto count=static_cast<std::size_t>(rows)*static_cast<std::size_t>(columns);
  if(limit && count>limit)throw std::length_error("D* Lite grid cell cap exceeded");
  return {rows,columns,count};
}
/// @brief Validate all grid row widths and representable dimensions/count.
/// @param grid Borrowed occupancy.
/// @param limit Optional complete-cell cap.
/// @return Validated dimensions.
/// @throws std::invalid_argument For empty/ragged shape.
/// @throws std::length_error For dimensions/count/cap bounds.
Shape shape(const std::vector<std::vector<bool>>& grid,std::size_t limit) {
  if(grid.empty() || grid[0].empty())throw std::invalid_argument("D* Lite grid is empty");
  if(grid.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()) || grid[0].size()>static_cast<std::size_t>(std::numeric_limits<int>::max()))throw std::length_error("D* Lite dimensions exceed int");
  for(const auto& row:grid)if(row.size()!=grid[0].size())throw std::invalid_argument("D* Lite grid is ragged");
  return dimensions(static_cast<int>(grid.size()),static_cast<int>(grid[0].size()),limit);
}
/// @brief Test coordinates before flattening or signed arithmetic.
/// @param p Row/column.
/// @param rows Valid rows.
/// @param columns Valid columns.
/// @return Whether p lies inside.
bool inside(std::pair<int,int> p,int rows,int columns) {return p.first>=0&&p.second>=0&&p.first<rows&&p.second<columns;}
/// @brief Exact widened Manhattan distance between int coordinates.
/// @param a First cell.
/// @param b Second cell.
/// @return Nonnegative sum, with no signed-int subtraction overflow.
std::uint64_t manhattan(std::pair<int,int> a,std::pair<int,int> b) {
  const auto dr=static_cast<std::int64_t>(a.first)-b.first,dc=static_cast<std::int64_t>(a.second)-b.second;
  return static_cast<std::uint64_t>(dr<0?-dr:dr)+static_cast<std::uint64_t>(dc<0?-dc:dc);
}
/// @brief Validate one coordinate with a consistent exception before mutation.
/// @param p Supplied cell.
/// @param rows Valid rows.
/// @param columns Valid columns.
/// @throws std::invalid_argument If outside.
void coordinate(std::pair<int,int> p,int rows,int columns) {
  if(!inside(p,rows,columns))throw std::invalid_argument("D* Lite coordinate outside grid");
}
} // namespace

DStarLiteGrid::DStarLiteGrid(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,std::pair<int,int> goal,std::size_t max_cells) {
  const auto s=shape(blocked,max_cells);coordinate(start,s.rows,s.columns);coordinate(goal,s.rows,s.columns);
  rows_=s.rows;columns_=s.columns;count_=s.count;start_=start;goal_=goal;blocked_=blocked;
  g_.assign(count_,d_star_infinity);rhs_=g_;
  heap_.reserve(count_);position_.assign(count_,std::numeric_limits<std::size_t>::max());keys_.assign(count_,{d_star_infinity,d_star_infinity});
  if(!blocked_[goal.first][goal.second]){const auto target=index(goal);rhs_[target]=0;insert(target,key(target));}
}
std::size_t DStarLiteGrid::index(std::pair<int,int> cell) const {return static_cast<std::size_t>(cell.first)*columns_+cell.second;}
std::array<std::size_t,4> DStarLiteGrid::neighbors(std::size_t u,std::size_t& count) const {
  const auto row=u/static_cast<std::size_t>(columns_),column=u%static_cast<std::size_t>(columns_);count=0;std::array<std::size_t,4> out{};
  if(row+1<static_cast<std::size_t>(rows_))out[count++]=u+columns_;
  if(row>0)out[count++]=u-columns_;
  if(column+1<static_cast<std::size_t>(columns_))out[count++]=u+1;
  if(column>0)out[count++]=u-1;
  return out;
}
DStarKey DStarLiteGrid::key(std::size_t u) const {
  const auto m=std::min(g_[u],rhs_[u]);if(m==d_star_infinity)return {d_star_infinity,d_star_infinity};
  const std::pair<int,int> cell{static_cast<int>(u/columns_),static_cast<int>(u%columns_)};
  return {m+manhattan(start_,cell)+modifier_,m};
}
bool DStarLiteGrid::before(std::size_t a,std::size_t b) const {return keys_[a]<keys_[b] || (keys_[a]==keys_[b]&&a<b);}
void DStarLiteGrid::swap_heap(std::size_t a,std::size_t b) {std::swap(heap_[a],heap_[b]);position_[heap_[a]]=a;position_[heap_[b]]=b;}
void DStarLiteGrid::sift_up(std::size_t p) {while(p>0){const auto parent=(p-1)/2;if(!before(heap_[p],heap_[parent]))break;swap_heap(p,parent);p=parent;}}
void DStarLiteGrid::sift_down(std::size_t p) {
  for(;;){auto child=2*p+1;if(child>=heap_.size())return;if(child+1<heap_.size()&&before(heap_[child+1],heap_[child]))++child;if(!before(heap_[child],heap_[p]))return;swap_heap(p,child);p=child;}
}
void DStarLiteGrid::erase(std::size_t u) {
  const auto p=position_[u];if(p==std::numeric_limits<std::size_t>::max())return;
  swap_heap(p,heap_.size()-1);heap_.pop_back();position_[u]=std::numeric_limits<std::size_t>::max();++pending_.queue_removals;
  if(p<heap_.size()){if(p>0&&before(heap_[p],heap_[(p-1)/2]))sift_up(p);else sift_down(p);}
}
void DStarLiteGrid::insert(std::size_t u,DStarKey priority) {
  if(position_[u]!=std::numeric_limits<std::size_t>::max())throw std::logic_error("D* Lite duplicate indexed entry");
  keys_[u]=priority;position_[u]=heap_.size();heap_.push_back(u);sift_up(heap_.size()-1);++pending_.queue_inserts;pending_.queue_peak=std::max(pending_.queue_peak,heap_.size());
}
void DStarLiteGrid::update_vertex(std::size_t u) {
  ++pending_.vertex_updates;const int row=static_cast<int>(u/columns_),column=static_cast<int>(u%columns_);
  rhs_[u]=d_star_infinity;
  if(u==index(goal_)){if(!blocked_[row][column])rhs_[u]=0;}
  else if(!blocked_[row][column]) {
    std::size_t count=0;const auto next=neighbors(u,count);
    for(std::size_t i=0;i<count;++i){++pending_.neighbor_checks;const auto v=next[i];if(blocked_[v/columns_][v%columns_])continue;
      if(g_[v]!=d_star_infinity && g_[v]+1<count_)rhs_[u]=std::min(rhs_[u],g_[v]+1);}
  }
  erase(u);if(g_[u]!=rhs_[u])insert(u,key(u));
}
void DStarLiteGrid::work_room(std::size_t units) const {
  const auto max=std::numeric_limits<std::size_t>::max();if(units>=max/128)throw std::length_error("D* Lite work bound overflow");
  const auto room=128*(units+1);
  for(auto value:{pending_.changed_cells,pending_.start_moves,pending_.vertex_updates,pending_.neighbor_checks,pending_.queue_inserts,pending_.queue_removals,pending_.pops,pending_.lowerings,pending_.raisings,pending_.key_refreshes,pending_.queue_peak})
    if(value>max-room)throw std::length_error("D* Lite work counter headroom exhausted");
}
void DStarLiteGrid::update_cells(const std::vector<GridCellUpdate>& changes) {
  for(const auto& c:changes)coordinate(c.cell,rows_,columns_);work_room(changes.size());
  for(const auto& c:changes){auto cell=c.cell;if(blocked_[cell.first][cell.second]==c.blocked)continue;
    blocked_[cell.first][cell.second]=c.blocked;++pending_.changed_cells;const auto u=index(cell);update_vertex(u);
    std::size_t count=0;const auto next=neighbors(u,count);for(std::size_t i=0;i<count;++i)update_vertex(next[i]);}
}
void DStarLiteGrid::move_start(std::pair<int,int> start) {
  coordinate(start,rows_,columns_);if(start==start_)return;work_room(1);
  const auto delta=manhattan(start_,start),limit=d_star_infinity-4*static_cast<std::uint64_t>(count_);
  if(modifier_>limit-delta)throw std::overflow_error("D* Lite movement key headroom exhausted");
  modifier_+=delta;start_=start;++pending_.start_moves;
}
GridPathResult DStarLiteGrid::route() const {
  GridPathResult result;const auto source=index(start_),target=index(goal_);
  if(blocked_[start_.first][start_.second]||blocked_[goal_.first][goal_.second]||g_[source]==d_star_infinity)return result;
  auto u=source;result.path.push_back(start_);
  while(u!=target){std::size_t count=0;const auto next=neighbors(u,count);auto best=d_star_infinity;std::size_t winner=count_;
    for(std::size_t i=0;i<count;++i){const auto v=next[i];if(blocked_[v/columns_][v%columns_]||g_[v]==d_star_infinity)continue;
      const auto cost=g_[v]+1;if(cost<best||(cost==best&&v<winner)){best=cost;winner=v;}}
    if(winner==count_||best!=g_[u]||g_[winner]>=g_[u]||result.path.size()>=count_)throw std::logic_error("D* Lite route descent invariant failed");
    u=winner;result.path.push_back({static_cast<int>(u/columns_),static_cast<int>(u%columns_)});
  }
  result.found=true;result.cost=static_cast<double>(result.path.size()-1);return result;
}
DStarRepair DStarLiteGrid::run(DStarTrace* trace,std::size_t limit) {
  work_room(count_);
  if(trace){if(count_>(std::numeric_limits<std::size_t>::max()-heap_.size())/5)throw std::length_error("D* Lite trace count overflow");
    const auto base=5*count_+heap_.size();if(limit&&base>limit)throw std::length_error("D* Lite trace base cap exceeded");
    trace->captured_entries=base;trace->blocked=blocked_;trace->start=start_;trace->goal=goal_;trace->key_modifier=modifier_;trace->work_before=pending_;
    trace->initial_g=g_;trace->initial_rhs=rhs_;trace->initial_open.reserve(heap_.size());for(auto u:heap_)trace->initial_open.push_back({u,keys_[u]});}
  const auto source=index(start_);
  for(;;){const DStarKey top=heap_.empty()?DStarKey{d_star_infinity,d_star_infinity}:keys_[heap_.front()];
    if(!(top<key(source)) && g_[source]==rhs_[source])break;
    if(heap_.empty())throw std::logic_error("D* Lite inconsistent start without queue entry");
    const auto u=heap_.front();DStarStep step;step.vertex=u;step.old_key=keys_[u];step.current_key=key(u);step.g_before=g_[u];step.rhs_before=rhs_[u];
    if(trace){if(trace->captured_entries==std::numeric_limits<std::size_t>::max()||(limit&&trace->captured_entries>=limit))throw std::length_error("D* Lite trace pop cap exceeded");trace->steps.push_back(step);++trace->captured_entries;}
    erase(u);++pending_.pops;
    if(step.old_key<step.current_key){step.action=DStarAction::rekey;insert(u,step.current_key);++pending_.key_refreshes;}
    else if(g_[u]>rhs_[u]){step.action=DStarAction::lower;g_[u]=rhs_[u];++pending_.lowerings;std::size_t count=0;const auto next=neighbors(u,count);for(std::size_t i=0;i<count;++i)update_vertex(next[i]);}
    else {step.action=DStarAction::raise;g_[u]=d_star_infinity;++pending_.raisings;update_vertex(u);std::size_t count=0;const auto next=neighbors(u,count);for(std::size_t i=0;i<count;++i)update_vertex(next[i]);}
    step.g_after=g_[u];step.rhs_after=rhs_[u];if(trace)trace->steps.back()=step;
  }
  DStarRepair result;result.route=route();result.work=pending_;
  if(trace){trace->final_g=g_;trace->final_rhs=rhs_;trace->final_top=heap_.empty()?DStarKey{d_star_infinity,d_star_infinity}:keys_[heap_.front()];}
  pending_={};pending_.queue_peak=heap_.size();return result;
}
DStarRepair DStarLiteGrid::repair() {return run(nullptr,0);}
DStarTrace DStarLiteGrid::repair_trace(std::size_t limit) {DStarTrace trace;trace.repair=run(&trace,limit);return trace;}
GridPathResult d_star_replan(const std::vector<std::vector<bool>>& blocked,std::pair<int,int> start,std::pair<int,int> goal) {
  const auto s=shape(blocked,0);if(!inside(start,s.rows,s.columns)||!inside(goal,s.rows,s.columns))return {};
  return DStarLiteGrid(blocked,start,goal,0).repair().route;
}
std::vector<std::vector<bool>> grid_obstacle_mask(int rows,int columns,const std::vector<std::pair<int,int>>& obstacles,std::size_t max_cells) {
  dimensions(rows,columns,max_cells);for(auto p:obstacles)coordinate(p,rows,columns);
  std::vector<std::vector<bool>> grid(rows,std::vector<bool>(columns,false));for(auto p:obstacles)grid[p.first][p.second]=true;return grid;
}
graphics::RPlot plot_grid_route(const std::vector<std::vector<bool>>& blocked,const GridPathResult& path,std::pair<int,int> start,std::pair<int,int> goal,std::size_t max_cells) {
  const auto s=shape(blocked,max_cells);coordinate(start,s.rows,s.columns);coordinate(goal,s.rows,s.columns);
  if(!path.found){if(!path.path.empty()||path.cost!=0)throw std::invalid_argument("grid route failure record must be empty/zero");}
  else {if(path.path.empty()||path.path.front()!=start||path.path.back()!=goal||!std::isfinite(path.cost)||path.cost!=static_cast<double>(path.path.size()-1))throw std::invalid_argument("grid route endpoints/cost malformed");
    for(std::size_t i=0;i<path.path.size();++i){const auto p=path.path[i];coordinate(p,s.rows,s.columns);if(blocked[p.first][p.second]||(i&&manhattan(p,path.path[i-1])!=1))throw std::invalid_argument("grid route crosses blocked/nonadjacent cell");}}
  auto plot=graphics::RPlot::create();
  for(int r=0;r<s.rows;++r)for(int c=0;c<s.columns;++c)plot.polygon({c-.5,c+.5,c+.5,c-.5},{-r-.5,-r-.5,-r+.5,-r+.5},blocked[r][c]?graphics::RGBA{30,41,59}:graphics::RGBA{241,245,249});
  if(path.found){std::vector<double>x,y;for(auto [r,c]:path.path){x.push_back(c);y.push_back(-r);}plot.line(x,y,"route",{37,99,235},3).points(x,y,"",{37,99,235},4);}
  plot.points({static_cast<double>(start.second)},{-static_cast<double>(start.first)},"",{134,239,172},10)
      .text(start.second,-start.first-.12,start==goal?"S=G":"S",{15,23,42},16);
  if(start!=goal)plot.points({static_cast<double>(goal.second)},{-static_cast<double>(goal.first)},"",{253,186,116},10).text(goal.second,-goal.first-.12,"G",{15,23,42},16);
  std::vector<double>x,y;std::vector<std::string>xl,yl;for(int c=0;c<s.columns;++c){x.push_back(c);xl.push_back(std::to_string(c));}for(int r=s.rows-1;r>=0;--r){y.push_back(-r);yl.push_back(std::to_string(r));}
  plot.x_ticks(x,xl).y_ticks(y,yl).x_label("column").y_label("row").x_limits(-.5,s.columns-.5).y_limits(-s.rows+.5,2.5).title("Four-neighbor unit-grid route");
  return plot;
}
} // namespace datamunge::algorithms
