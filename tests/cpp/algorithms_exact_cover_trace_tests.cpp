#include <datamunge/algorithms/exact_cover.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <numeric>
#include <random>
#include <set>
using namespace datamunge::algorithms;
namespace {
bool brute(int columns,const std::vector<std::vector<int>>& rows){for(std::size_t mask=0;mask<(std::size_t{1}<<rows.size());++mask){std::vector<int> count(columns);bool valid=true;for(std::size_t r=0;r<rows.size();++r)if(mask&(std::size_t{1}<<r))for(int c:rows[r])if(++count[c]>1)valid=false;if(valid&&std::all_of(count.begin(),count.end(),[](int n){return n==1;}))return true;}return false;}
void witness(int columns,const std::vector<std::vector<int>>& rows,const ExactCoverResult& result){if(!result.solved){EXPECT_TRUE(result.rows.empty());return;}std::vector<int> counts(columns);std::set<int> chosen;for(int id:result.rows){ASSERT_GE(id,0);ASSERT_LT(static_cast<std::size_t>(id),rows.size());EXPECT_TRUE(chosen.insert(id).second);EXPECT_FALSE(rows[id].empty());for(int c:rows[id])++counts[c];}for(int n:counts)EXPECT_EQ(n,1);}
void mesh(const std::vector<ExactCoverNode>& nodes,int columns,const std::vector<std::vector<int>>& rows){std::size_t ones=0;for(const auto& row:rows)ones+=row.size();ASSERT_EQ(nodes.size(),1+columns+ones);ASSERT_FALSE(nodes.empty());EXPECT_EQ(nodes[0].column,-1);EXPECT_EQ(nodes[0].row,-1);EXPECT_EQ(nodes[0].right,columns?1:0);EXPECT_EQ(nodes[0].left,columns);EXPECT_EQ(nodes[0].up,0);EXPECT_EQ(nodes[0].down,0);
 for(std::size_t id=0;id<nodes.size();++id){const auto& n=nodes[id];for(int link:{n.left,n.right,n.up,n.down}){ASSERT_GE(link,0);ASSERT_LT(static_cast<std::size_t>(link),nodes.size());}EXPECT_EQ(nodes[n.left].right,int(id));EXPECT_EQ(nodes[n.right].left,int(id));EXPECT_EQ(nodes[n.up].down,int(id));EXPECT_EQ(nodes[n.down].up,int(id));}
 for(int c=1;c<=columns;++c){EXPECT_EQ(nodes[c].column,c);EXPECT_EQ(nodes[c].row,-1);std::size_t count=0;for(int id=nodes[c].down;id!=c;id=nodes[id].down){ASSERT_LT(++count,nodes.size());EXPECT_EQ(nodes[id].column,c);}EXPECT_EQ(nodes[c].size,int(count));const auto expected=std::count_if(rows.begin(),rows.end(),[c](const auto& row){return std::find(row.begin(),row.end(),c-1)!=row.end();});EXPECT_EQ(count,static_cast<std::size_t>(expected));}
 std::size_t id=1+columns;for(std::size_t r=0;r<rows.size();++r){const auto first=id;for(std::size_t k=0;k<rows[r].size();++k,++id){EXPECT_EQ(nodes[id].column,rows[r][k]+1);EXPECT_EQ(nodes[id].row,int(r));EXPECT_EQ(nodes[id].size,0);EXPECT_EQ(nodes[id].left,int(k? id-1:first+rows[r].size()-1));EXPECT_EQ(nodes[id].right,int(k+1<rows[r].size()?id+1:first));}}}
void certificate(const ExactCoverTrace& t){const auto c=t.num_columns;const auto& rows=t.input_rows;witness(c,rows,t.outcome.result);EXPECT_EQ(t.initial_nodes,t.final_nodes);mesh(t.initial_nodes,c,rows);mesh(t.final_nodes,c,rows);std::size_t ones=0;for(const auto& row:rows)ones+=row.size();std::size_t capture=rows.size()+ones+2*t.initial_nodes.size(),chosen=0,attempts=0,restored=0,dead=0,solutions=0,scans=0,covers=0,peak=0;
 ASSERT_FALSE(t.steps.empty());for(std::size_t index=0;index<t.steps.size();++index){const auto& s=t.steps[index];EXPECT_EQ(s.depth,s.partial_rows.size());peak=std::max(peak,s.depth);capture+=1+c+s.depth;std::vector<bool> covered(c);std::set<int> selected;
  for(int r:s.partial_rows){ASSERT_GE(r,0);ASSERT_LT(static_cast<std::size_t>(r),rows.size());EXPECT_TRUE(selected.insert(r).second);for(int col:rows[r]){EXPECT_FALSE(covered[col]);covered[col]=true;}}
  if(s.kind==ExactCoverEvent::unselect_row){ASSERT_GE(s.column,0);ASSERT_LT(s.column,c);EXPECT_FALSE(covered[s.column]);covered[s.column]=true;EXPECT_FALSE(selected.contains(s.row));++restored;}
  std::vector<int> expected(c,-1);for(int col=0;col<c;++col)if(!covered[col]){expected[col]=0;for(const auto& row:rows)if(std::find(row.begin(),row.end(),col)!=row.end()&&std::none_of(row.begin(),row.end(),[&](int item){return covered[item];}))++expected[col];}EXPECT_EQ(s.column_sizes,expected);
  if(s.kind==ExactCoverEvent::choose_column){++chosen;int best=-1;for(int col=0;col<c;++col)if(expected[col]>=0){++scans;if(best<0||expected[col]<expected[best])best=col;}EXPECT_EQ(s.column,best);if(best>=0&&expected[best]>0)++covers;}
  if(s.kind==ExactCoverEvent::select_row){++attempts;ASSERT_FALSE(s.partial_rows.empty());EXPECT_EQ(s.row,s.partial_rows.back());covers+=rows[s.row].size()-1;EXPECT_NE(std::find(rows[s.row].begin(),rows[s.row].end(),s.column),rows[s.row].end());}
  if(s.kind==ExactCoverEvent::dead_end){++dead;ASSERT_GE(s.column,0);EXPECT_EQ(expected[s.column],0);ASSERT_GT(index,0u);EXPECT_EQ(t.steps[index-1].kind,ExactCoverEvent::choose_column);}
  if(s.kind==ExactCoverEvent::solution){++solutions;EXPECT_TRUE(std::all_of(covered.begin(),covered.end(),[](bool b){return b;}));EXPECT_EQ(s.partial_rows,t.outcome.result.rows);}
  if(s.kind==ExactCoverEvent::finished){EXPECT_EQ(index+1,t.steps.size());EXPECT_TRUE(s.partial_rows.empty());}
 }
 const auto& w=t.outcome.work;EXPECT_EQ(t.steps.back().kind,ExactCoverEvent::finished);EXPECT_EQ(capture,t.captured_entries);EXPECT_EQ(w.search_entries,chosen+solutions);EXPECT_EQ(w.column_scans,scans);EXPECT_EQ(w.row_attempts,attempts);EXPECT_EQ(w.row_restorations,restored);EXPECT_EQ(attempts,restored);EXPECT_EQ(w.dead_ends,dead);EXPECT_EQ(w.maximum_depth,peak);EXPECT_EQ(w.covers,covers);EXPECT_EQ(w.uncovers,covers);EXPECT_EQ(w.header_link_writes,4*covers);EXPECT_EQ(w.data_link_writes,2*w.column_size_updates);EXPECT_EQ(w.column_size_updates%2,0u);EXPECT_EQ(solutions,t.outcome.result.solved?1u:0u);
}
}
TEST(ExactCoverTrace, TinyBacktrackingCapturesActualResidualsAndRestoresEveryNode){
 const std::vector<std::vector<int>> rows{{0,1},{0,2},{1,2},{0,1,2}};const auto t=exact_cover_trace(3,rows);certificate(t);EXPECT_EQ(t.outcome.result.rows,(std::vector<int>{3}));EXPECT_EQ(t.steps.size(),13u);EXPECT_EQ(t.captured_entries,99u);EXPECT_EQ(t.outcome.work.covers,5u);EXPECT_EQ(t.outcome.work.dead_ends,2u);EXPECT_EQ(t.outcome.work.column_scans,5u);EXPECT_EQ(t.outcome.work.search_entries,4u);EXPECT_EQ(t.steps[4].kind,ExactCoverEvent::unselect_row);EXPECT_EQ(t.steps[4].column_sizes,(std::vector<int>{-1,1,1}));
 const std::vector<std::vector<int>> original{{0,3,6},{0,3},{3,4,6},{2,4,5},{1,6},{2,5}};const auto legacy=exact_cover_trace(7,original);certificate(legacy);EXPECT_EQ(legacy.outcome.result.rows,(std::vector<int>{4,1,3}));
 const auto paper=exact_cover_trace(7,{{2,4,5},{0,3,6},{1,2,5},{0,3},{1,6},{3,4,6}});certificate(paper);EXPECT_TRUE(paper.outcome.result.solved);
}
TEST(ExactCoverTrace, ExhaustiveBinaryMatricesAndRandomRowsMatchIndependentSubsetOracle){
 std::size_t fixtures=0;for(int columns=0;columns<=3;++columns)for(std::size_t count=0;count<=4;++count){const auto possibilities=std::size_t{1}<<(columns*count);for(std::size_t mask=0;mask<possibilities;++mask){std::vector<std::vector<int>> rows(count);for(std::size_t r=0;r<count;++r)for(int c=0;c<columns;++c)if(mask&(std::size_t{1}<<(r*columns+c)))rows[r].push_back(c);const auto t=exact_cover_trace(columns,rows);certificate(t);EXPECT_EQ(t.outcome.result.solved,brute(columns,rows));EXPECT_EQ(exact_cover(columns,rows).rows,t.outcome.result.rows);EXPECT_EQ(exact_cover_work(columns,rows).result.rows,t.outcome.result.rows);++fixtures;}}EXPECT_EQ(fixtures,5058u);
 std::mt19937 rng(2000);for(int trial=0;trial<500;++trial){int cols=rng()%9;std::vector<std::vector<int>> rows(rng()%13);for(auto& row:rows){for(int c=0;c<cols;++c)if(rng()%3==0)row.push_back(c);if(rng()%2)std::reverse(row.begin(),row.end());}const auto t=exact_cover_trace(cols,rows);certificate(t);EXPECT_EQ(t.outcome.result.solved,brute(cols,rows));}
}
TEST(ExactCoverTrace, ExactWholeBaseOnlineQuotasAndCopiesHaveExplicitOwnership){
 const std::vector<std::vector<int>> rows{{0,1},{0,2},{1,2},{0,1,2}};EXPECT_EQ(exact_cover_trace(3,rows,99).captured_entries,99u);
 for(std::size_t cap=1;cap<99;++cap){EXPECT_THROW((void)exact_cover_trace(3,rows,cap),std::length_error);}
 EXPECT_EQ(exact_cover_trace(3,rows,0).captured_entries,99u);EXPECT_EQ(exact_cover_trace(0,{},4).captured_entries,4u);
 EXPECT_THROW((void)exact_cover_trace(0,{},3),std::length_error);
 auto input=rows;const auto owned=exact_cover_trace(3,input);input[0].clear();auto copy=owned;copy.input_rows[0].clear();copy.initial_nodes[0].right=99;copy.final_nodes[0].right=99;copy.steps[0].column_sizes[0]=99;copy.outcome.result.rows.clear();EXPECT_EQ(owned.input_rows,rows);EXPECT_EQ(owned.initial_nodes[0].right,1);EXPECT_EQ(owned.final_nodes[0].right,1);EXPECT_EQ(owned.steps[0].column_sizes[0],3);EXPECT_EQ(owned.outcome.result.rows,(std::vector<int>{3}));
}
TEST(ExactCoverTrace, InvalidIncidenceAndGeometryFailBeforeUnsafeAccessWhileEmptyRowsStayValid){
 EXPECT_THROW((void)exact_cover(-1,{}),std::invalid_argument);
 for(const auto& bad:std::vector<std::vector<std::vector<int>>>{{{-1}},{{1}},{{0,0}},{{0,0,0}}}){EXPECT_THROW((void)exact_cover_trace(1,bad,0),std::invalid_argument);}
 EXPECT_THROW((void)exact_cover(0,{{0}}),std::invalid_argument);
 EXPECT_EQ(exact_cover_node_count(3,4,9),13u);
 EXPECT_THROW((void)exact_cover_node_count(std::numeric_limits<int>::max(),0,0),std::length_error);
 EXPECT_THROW((void)exact_cover_node_count(0,std::numeric_limits<std::size_t>::max(),0),std::length_error);
 EXPECT_THROW((void)exact_cover_node_count(0,0,std::numeric_limits<std::size_t>::max()),std::length_error);
 const auto empty=exact_cover_trace(0,{{},{}});certificate(empty);EXPECT_TRUE(empty.outcome.result.solved);EXPECT_TRUE(empty.outcome.result.rows.empty());
 const auto duplicate=exact_cover_trace(2,{{},{1,0},{1,0},{}});certificate(duplicate);EXPECT_EQ(duplicate.outcome.result.rows,(std::vector<int>{1}));const auto impossible=exact_cover_trace(3,{{0},{1}});certificate(impossible);EXPECT_FALSE(impossible.outcome.result.solved);
}
TEST(ExactCoverTrace, DominoBuilderPreservesAllPlacementsAndNativePicturesValidateTheirCovers){
 for(int h=0;h<=6;++h)for(int w=0;w<=8;++w){const auto p=domino_exact_cover(h,w);std::vector<std::vector<int>> expected;for(int r=0;r<h;++r)for(int c=0;c<w;++c){if(c+1<w)expected.push_back({r*w+c,r*w+c+1});if(r+1<h)expected.push_back({r*w+c,(r+1)*w+c});}EXPECT_EQ(p.num_columns,h*w);EXPECT_EQ(p.placements,expected);const auto outcome=exact_cover_work(p.num_columns,p.placements);witness(p.num_columns,p.placements,outcome.result);EXPECT_EQ(outcome.result.solved,(h*w)%2==0);EXPECT_EQ(outcome.work.covers,outcome.work.uncovers);if((h*w)%2==0||h*w<=15){const auto t=exact_cover_trace(p.num_columns,p.placements);certificate(t);EXPECT_EQ(t.outcome.result.rows,outcome.result.rows);}}
 const auto p=domino_exact_cover(4,6);const auto result=exact_cover(p.num_columns,p.placements);EXPECT_EQ(result.rows.size(),12u);EXPECT_NE(domino_tiling_diagram(p,result).to_svg().find("Selected domino placement row IDs"),std::string::npos);
 const auto small=domino_exact_cover(2,2);const auto r=exact_cover(small.num_columns,small.placements);
 EXPECT_THROW((void)domino_tiling_diagram(small,r,64,1100,3),std::length_error);
 EXPECT_NO_THROW((void)domino_tiling_diagram(small,r,64,1100,0));
 EXPECT_THROW((void)domino_tiling_diagram(small,r,0),std::invalid_argument);
 EXPECT_THROW((void)domino_tiling_diagram(small,{true,{0}}),std::invalid_argument);
 EXPECT_THROW((void)domino_tiling_diagram(small,{true,{0,0}}),std::invalid_argument);
 EXPECT_THROW((void)domino_tiling_diagram(small,{true,{-1}}),std::invalid_argument);
 EXPECT_THROW((void)domino_tiling_diagram(small,{false,{0}}),std::invalid_argument);
 for(int mode=0;mode<4;++mode){auto bad=small;if(mode==0)bad.num_columns=3;if(mode==1)bad.placements[0]={0};if(mode==2)bad.placements[0]={0,4};if(mode==3)bad.placements[0]={0,3};EXPECT_THROW((void)domino_tiling_diagram(bad,r),std::invalid_argument);}
 EXPECT_THROW((void)domino_exact_cover(-1,2),std::invalid_argument);
 EXPECT_THROW((void)domino_exact_cover(std::numeric_limits<int>::max(),std::numeric_limits<int>::max()),std::length_error);
 const auto zero=domino_exact_cover(std::numeric_limits<int>::max(),0);EXPECT_EQ(zero.num_columns,0);EXPECT_TRUE(zero.placements.empty());EXPECT_NE(domino_tiling_diagram(zero,{true,{}}).to_svg().find("Empty board"),std::string::npos);
 EXPECT_NE(domino_tiling_diagram(domino_exact_cover(1,1),{false,{}}).to_svg().find("Reported unsolved"),std::string::npos);
}
TEST(ExactCoverTrace, EqualNodeFamiliesExposeHeuristicCostAndDeepSearchUsesExplicitFrames){
 for(std::size_t n:{1,2,4,8,16,32,64}){std::vector<std::vector<int>> singleton(n);std::vector<int> whole(n);for(std::size_t i=0;i<n;++i){singleton[i]={int(i)};whole[i]=i;}const auto a=exact_cover_trace(n,singleton),b=exact_cover_trace(n,{whole});certificate(a);certificate(b);EXPECT_EQ(a.initial_nodes.size(),b.initial_nodes.size());EXPECT_EQ(a.outcome.work.column_scans,n*(n+1)/2);EXPECT_EQ(b.outcome.work.column_scans,n);EXPECT_EQ(a.outcome.work.maximum_depth,n);EXPECT_EQ(b.outcome.work.maximum_depth,1u);EXPECT_EQ(a.outcome.work.data_link_writes,0u);EXPECT_EQ(b.outcome.work.data_link_writes,4*(n-1));EXPECT_EQ(a.captured_entries,(9*n*n+23*n+8)/2);EXPECT_EQ(b.captured_entries,10*n+10);}
 const std::size_t n=4096;std::vector<std::vector<int>> rows(n);for(std::size_t i=0;i<n;++i)rows[i]={int(i)};const auto r=exact_cover_work(n,rows);ASSERT_TRUE(r.result.solved);ASSERT_EQ(r.result.rows.size(),n);for(std::size_t i=0;i<n;++i)EXPECT_EQ(r.result.rows[i],int(i));EXPECT_EQ(r.work.maximum_depth,n);EXPECT_EQ(r.work.search_entries,n+1);EXPECT_EQ(r.work.column_scans,n*(n+1)/2);EXPECT_EQ(r.work.covers,n);EXPECT_EQ(r.work.uncovers,n);EXPECT_EQ(r.work.data_link_writes,0u);
}
