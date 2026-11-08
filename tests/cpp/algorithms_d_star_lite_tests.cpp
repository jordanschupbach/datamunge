#include <gtest/gtest.h>
#include <datamunge/algorithms/d_star.hpp>
#include <algorithm>
#include <map>
#include <queue>
#include <random>
#include <limits>
using namespace datamunge::algorithms;
namespace {
using Grid=std::vector<std::vector<bool>>;
std::vector<std::size_t> adjacent(std::size_t u,std::size_t R,std::size_t C){std::vector<std::size_t> n;const auto r=u/C,c=u%C;if(r+1<R)n.push_back(u+C);if(r)n.push_back(u-C);if(c+1<C)n.push_back(u+1);if(c)n.push_back(u-1);return n;}
int bfs(const Grid&g,std::pair<int,int>s,std::pair<int,int>t){if(g[s.first][s.second]||g[t.first][t.second])return -1;std::queue<std::pair<int,int>>q;q.push(s);std::vector<std::vector<int>>d(g.size(),std::vector<int>(g[0].size(),-1));d[s.first][s.second]=0;while(!q.empty()){auto p=q.front();q.pop();if(p==t)return d[p.first][p.second];for(auto v:adjacent(p.first*g[0].size()+p.second,g.size(),g[0].size())){int r=v/g[0].size(),c=v%g[0].size();if(!g[r][c]&&d[r][c]<0){d[r][c]=d[p.first][p.second]+1;q.push({r,c});}}}return -1;}
void path_check(const Grid&g,std::pair<int,int>s,std::pair<int,int>t,const GridPathResult&p){int expected=bfs(g,s,t);ASSERT_EQ(p.found,expected>=0);if(expected<0){EXPECT_TRUE(p.path.empty());EXPECT_EQ(p.cost,0);return;}ASSERT_EQ(p.cost,expected);ASSERT_EQ(p.path.size(),expected+1);ASSERT_EQ(p.path.front(),s);ASSERT_EQ(p.path.back(),t);for(std::size_t i=0;i<p.path.size();++i){auto[r,c]=p.path[i];ASSERT_GE(r,0);ASSERT_GE(c,0);ASSERT_LT(r,g.size());ASSERT_LT(c,g[0].size());EXPECT_FALSE(g[r][c]);if(i)EXPECT_EQ(std::abs(r-p.path[i-1].first)+std::abs(c-p.path[i-1].second),1);}}
void certificate(const DStarTrace&t){const auto R=t.blocked.size(),C=t.blocked[0].size(),N=R*C,goal=t.goal.first*C+t.goal.second,source=t.start.first*C+t.start.second;auto g=t.initial_g,rhs=t.initial_rhs;ASSERT_EQ(g.size(),N);ASSERT_EQ(rhs.size(),N);std::map<std::size_t,DStarKey>open;for(auto[u,k]:t.initial_open){EXPECT_FALSE(open.count(u));open[u]=k;}for(std::size_t u=0;u<N;++u)EXPECT_EQ(open.count(u)!=0,g[u]!=rhs[u]);
auto key=[&](std::size_t u){auto m=std::min(g[u],rhs[u]);if(m==d_star_infinity)return DStarKey{d_star_infinity,d_star_infinity};auto h=std::abs(t.start.first-static_cast<int>(u/C))+std::abs(t.start.second-static_cast<int>(u%C));return DStarKey{m+h+t.key_modifier,m};};auto w=t.work_before;
auto erase=[&](std::size_t u){if(open.erase(u))++w.queue_removals;};auto insert=[&](std::size_t u,DStarKey k){EXPECT_FALSE(open.count(u));open[u]=k;++w.queue_inserts;w.queue_peak=std::max(w.queue_peak,open.size());};
auto update=[&](std::size_t u){++w.vertex_updates;rhs[u]=d_star_infinity;if(u==goal){if(!t.blocked[u/C][u%C])rhs[u]=0;}else if(!t.blocked[u/C][u%C])for(auto v:adjacent(u,R,C)){++w.neighbor_checks;if(!t.blocked[v/C][v%C]&&g[v]!=d_star_infinity&&g[v]+1<N)rhs[u]=std::min(rhs[u],g[v]+1);}erase(u);if(g[u]!=rhs[u])insert(u,key(u));};
for(auto&s:t.steps){ASSERT_FALSE(open.empty());auto best=open.begin();for(auto it=open.begin();it!=open.end();++it)if(std::pair{it->second,it->first}<std::pair{best->second,best->first})best=it;ASSERT_EQ(s.vertex,best->first);EXPECT_EQ(s.old_key,best->second);EXPECT_EQ(s.current_key,key(s.vertex));EXPECT_EQ(s.g_before,g[s.vertex]);EXPECT_EQ(s.rhs_before,rhs[s.vertex]);auto u=s.vertex;erase(u);++w.pops;if(s.old_key<s.current_key){EXPECT_EQ(s.action,DStarAction::rekey);insert(u,s.current_key);++w.key_refreshes;}else if(g[u]>rhs[u]){EXPECT_EQ(s.action,DStarAction::lower);g[u]=rhs[u];++w.lowerings;for(auto v:adjacent(u,R,C))update(v);}else{EXPECT_EQ(s.action,DStarAction::raise);g[u]=d_star_infinity;++w.raisings;update(u);for(auto v:adjacent(u,R,C))update(v);}EXPECT_EQ(s.g_after,g[u]);EXPECT_EQ(s.rhs_after,rhs[u]);for(std::size_t v=0;v<N;++v)EXPECT_EQ(open.count(v)!=0,g[v]!=rhs[v]);}
EXPECT_EQ(g,t.final_g);EXPECT_EQ(rhs,t.final_rhs);EXPECT_EQ(g[source],rhs[source]);DStarKey top{d_star_infinity,d_star_infinity};for(auto[u,k]:open)top=std::min(top,k);EXPECT_EQ(top,t.final_top);EXPECT_FALSE(top<key(source));EXPECT_EQ(t.captured_entries,5*N+t.initial_open.size()+t.steps.size());const auto&a=t.repair.work;EXPECT_EQ(a.pops,w.pops);EXPECT_EQ(a.lowerings,w.lowerings);EXPECT_EQ(a.raisings,w.raisings);EXPECT_EQ(a.key_refreshes,w.key_refreshes);EXPECT_EQ(a.vertex_updates,w.vertex_updates);EXPECT_EQ(a.neighbor_checks,w.neighbor_checks);EXPECT_EQ(a.queue_inserts,w.queue_inserts);EXPECT_EQ(a.queue_removals,w.queue_removals);EXPECT_EQ(a.queue_peak,w.queue_peak);EXPECT_EQ(a.pops,a.lowerings+a.raisings+a.key_refreshes);path_check(t.blocked,t.start,t.goal,t.repair.route);}
}
TEST(DStarLite, OriginalTinyAndBlockedBoundaries){auto p=d_star_replan({{0,0},{0,0}},{0,0},{1,1});EXPECT_TRUE(p.found);EXPECT_EQ(p.cost,2);EXPECT_EQ(p.path,(std::vector<std::pair<int,int>>{{0,0},{0,1},{1,1}}));EXPECT_EQ(d_star_replan({{0,0,0},{0,1,0},{0,0,0}},{0,0},{2,2}).cost,4);for(bool blocked:{false,true}){DStarLiteGrid one({{blocked}},{0,0},{0,0});certificate(one.repair_trace());}EXPECT_FALSE(d_star_replan({{false}},{-1,0},{0,0}).found);}
TEST(DStarLite, AllThreeByThreeMasksAndTerminalPairsMatchIndependentBfs){for(unsigned mask=0;mask<512;++mask){Grid grid(3,std::vector<bool>(3));for(int i=0;i<9;++i)grid[i/3][i%3]=(mask>>i)&1;for(int s=0;s<9;++s)for(int t=0;t<9;++t)path_check(grid,{s/3,s%3},{t/3,t%3},d_star_replan(grid,{s/3,s%3},{t/3,t%3}));}}
TEST(DStarLite, ExhaustiveTwoByThreeFlipAndRestoreTracesMatchBfsAndActualQueueCertificate){for(unsigned mask=0;mask<64;++mask){Grid grid(2,std::vector<bool>(3));for(int i=0;i<6;++i)grid[i/3][i%3]=(mask>>i)&1;for(int s=0;s<6;++s)for(int t=0;t<6;++t){DStarLiteGrid p(grid,{s/3,s%3},{t/3,t%3});certificate(p.repair_trace());for(int u=0;u<6;++u){p.update_cells({{{u/3,u%3},!grid[u/3][u%3]}});certificate(p.repair_trace());p.update_cells({{{u/3,u%3},grid[u/3][u%3]}});certificate(p.repair_trace());}EXPECT_EQ(p.repair().work.pops,0);}}}
TEST(DStarLite, SeededBatchEditsStartMovesAndLazyKeyRefreshesStayOptimal){std::mt19937 rng(91);bool lower=false,raise=false,rekey=false;for(int trial=0;trial<60;++trial){Grid g(5,std::vector<bool>(7));for(auto&row:g)for(std::size_t c=0;c<row.size();++c)row[c]=(rng()%5)==0;DStarLiteGrid p(g,{2,0},{2,6});certificate(p.repair_trace());for(int j=0;j<40;++j){std::vector<GridCellUpdate>edits;for(int k=0;k<3;++k)edits.push_back({{static_cast<int>(rng()%5),static_cast<int>(rng()%7)},bool(rng()&1)});p.update_cells(edits);if(j%2==0)p.move_start({static_cast<int>(rng()%5),static_cast<int>(rng()%7)});auto t=p.repair_trace();certificate(t);for(auto&s:t.steps){lower|=s.action==DStarAction::lower;raise|=s.action==DStarAction::raise;rekey|=s.action==DStarAction::rekey;}}}EXPECT_TRUE(lower);EXPECT_TRUE(raise);EXPECT_TRUE(rekey);}
TEST(DStarLite, CaptureLimitsResumeAndCopiesHaveIndependentState){Grid g(2,std::vector<bool>(2));DStarLiteGrid p(g,{0,0},{1,1});auto original=p;auto full=original.repair_trace();const auto exact=full.captured_entries;
EXPECT_NO_THROW(p.repair_trace(exact));
DStarLiteGrid short_one(g,{0,0},{1,1});EXPECT_THROW(short_one.repair_trace(exact-1),std::length_error);
certificate(short_one.repair_trace(0));
DStarLiteGrid none(g,{0,0},{1,1});EXPECT_THROW(none.repair_trace(20),std::length_error);
EXPECT_EQ(none.repair().work.pops,full.repair.work.pops);
DStarLiteGrid partial(g,{0,0},{1,1});EXPECT_THROW(partial.repair_trace(22),std::length_error);
auto resumed=partial.repair_trace();EXPECT_GT(resumed.work_before.pops,0);EXPECT_LT(resumed.steps.size(),resumed.repair.work.pops);certificate(resumed);
auto copy=p;copy.update_cells({{{0,0},true}});EXPECT_FALSE(copy.repair().route.found);EXPECT_TRUE(p.repair().route.found);EXPECT_FALSE(p.grid()[0][0]);DStarLiteGrid assigned({{false}},{0,0},{0,0});assigned=p;assigned.update_cells({{{0,1},true}});EXPECT_TRUE(assigned.repair().route.found);EXPECT_FALSE(p.grid()[0][1]);assigned=assigned;EXPECT_TRUE(assigned.repair().route.found);}
TEST(DStarLite, CompleteShapeCoordinateAndBudgetValidationBeforeMutation){EXPECT_THROW(DStarLiteGrid({}, {0,0},{0,0}),std::invalid_argument);
EXPECT_THROW(DStarLiteGrid({{},{}},{0,0},{0,0}),std::invalid_argument);
EXPECT_THROW(DStarLiteGrid({{false},{false,false}},{0,0},{0,0}),std::invalid_argument);
EXPECT_THROW(DStarLiteGrid({{false}},{1,0},{0,0}),std::invalid_argument);
EXPECT_THROW(grid_obstacle_mask(0,3),std::invalid_argument);
EXPECT_THROW(grid_obstacle_mask(50000,50000,{},0),std::length_error);
EXPECT_THROW(grid_obstacle_mask(2,3,{},5),std::length_error);
EXPECT_NO_THROW(grid_obstacle_mask(2,3,{},6));
EXPECT_NO_THROW(grid_obstacle_mask(2,3,{},0));
EXPECT_THROW(grid_obstacle_mask(2,3,{{-1,0}}),std::invalid_argument);
DStarLiteGrid p({{false,false}},{0,0},{0,1});(void)p.repair();auto copy=p;
EXPECT_THROW(p.update_cells({{{0,0},true},{{9,9},false}}),std::invalid_argument);
EXPECT_EQ(p.grid(),copy.grid());
EXPECT_THROW(p.move_start({std::numeric_limits<int>::min(),0}),std::invalid_argument);
EXPECT_EQ(p.start(),copy.start());p.update_cells({{{0,0},true},{{0,0},false}});auto r=p.repair();EXPECT_EQ(r.work.changed_cells,2);EXPECT_TRUE(r.route.found);}
TEST(DStarLite, NativePlotValidatesRoutesAndSupportsSingleAndFailedCells){auto g=grid_obstacle_mask(2,2);auto r=d_star_replan(g,{0,0},{1,1});EXPECT_FALSE(plot_grid_route(g,r,{0,0},{1,1}).series().empty());auto bad=r;bad.cost=99;
EXPECT_THROW(plot_grid_route(g,bad,{0,0},{1,1}),std::invalid_argument);
bad=r;bad.path={{0,0},{1,1}};bad.cost=1;
EXPECT_THROW(plot_grid_route(g,bad,{0,0},{1,1}),std::invalid_argument);
EXPECT_THROW(plot_grid_route(g,r,{0,0},{1,1},3),std::length_error);
EXPECT_FALSE(plot_grid_route({{false}},d_star_replan({{false}},{0,0},{0,0}),{0,0},{0,0}).series().empty());
EXPECT_FALSE(plot_grid_route({{true}},GridPathResult{},{0,0},{0,0}).series().empty());}
TEST(DStarLite, LongCorridorBlockReopenAndWarmRepairReuseRetainedLabels){constexpr int n=50001;auto g=grid_obstacle_mask(1,n);DStarLiteGrid p(g,{0,0},{0,n-1});auto cold=p.repair();EXPECT_EQ(cold.route.cost,n-1);EXPECT_EQ(cold.work.lowerings,n);EXPECT_EQ(p.repair().work.pops,0);p.move_start({0,1});EXPECT_EQ(p.repair().route.cost,n-2);p.update_cells({{{0,n/2},true}});EXPECT_FALSE(p.repair().route.found);p.update_cells({{{0,n/2},false}});EXPECT_EQ(p.repair().route.cost,n-2);p.update_cells({{{0,n-1},true}});EXPECT_FALSE(p.repair().route.found);p.update_cells({{{0,n-1},false}});EXPECT_EQ(p.repair().route.cost,n-2);}
