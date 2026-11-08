#include <gtest/gtest.h>
#include <datamunge/algorithms/dpll.hpp>
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <set>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Formula=std::vector<std::vector<int>>;
int id(int l){return l<0?-l:l;}
bool literal(int l,const std::vector<int>& a){return a[id(l)]==(l>0?1:-1);}
bool model(const Formula& f,const std::vector<int>&a){for(auto&c:f){bool sat=false;for(int l:c)sat|=literal(l,a);if(!sat)return false;}return true;}
bool brute(int n,const Formula&f){for(unsigned m=0;m<(1u<<n);++m){std::vector<int>a(n+1);for(int i=1;i<=n;++i)a[i]=(m>>(i-1))&1?1:-1;if(model(f,a))return true;}return false;}
void certificate(int n,const Formula&f,const DpllTrace&t){ASSERT_EQ(t.num_vars,n);ASSERT_EQ(t.clauses,f);std::map<std::size_t,std::vector<int>>states;std::map<std::size_t,std::size_t>parents,depth;std::map<std::size_t,int>incoming;std::map<std::size_t,std::vector<std::size_t>>children;std::set<std::size_t>failed;std::size_t last_failed=0,nodes=0,units=0,pures=0,conflicts=0,backs=0,maxdepth=0,entries=1+f.size();for(auto&c:f)entries+=c.size();
for(auto&e:t.events){ASSERT_EQ(e.assignment.size(),n+1);EXPECT_EQ(e.assignment[0],0);for(int a:e.assignment)EXPECT_TRUE(a==0||a==1||a==-1);entries+=1+e.assignment.size();maxdepth=std::max(maxdepth,e.depth);
if(e.kind==DpllEventKind::enter){EXPECT_EQ(e.node,nodes++);ASSERT_FALSE(states.count(e.node));if(e.node==0){EXPECT_EQ(e.parent,std::numeric_limits<std::size_t>::max());EXPECT_EQ(e.depth,0);EXPECT_EQ(e.literal,0);EXPECT_EQ(e.assignment,std::vector<int>(n+1,0));}else{ASSERT_TRUE(states.count(e.parent));auto a=states[e.parent];ASSERT_NE(e.literal,0);EXPECT_EQ(a[id(e.literal)],0);a[id(e.literal)]=e.literal>0?1:-1;EXPECT_EQ(e.assignment,a);EXPECT_EQ(e.depth,depth[e.parent]+1);children[e.parent].push_back(e.node);EXPECT_LE(children[e.parent].size(),2);}states[e.node]=e.assignment;parents[e.node]=e.parent;incoming[e.node]=e.literal;depth[e.node]=e.depth;}
else{ASSERT_TRUE(states.count(e.node));auto before=states[e.node];EXPECT_EQ(e.parent,parents[e.node]);EXPECT_EQ(e.depth,depth[e.node]);if(e.kind==DpllEventKind::unit){++units;ASSERT_LT(e.clause,f.size());int pending=0,only=0;for(int l:f[e.clause]){EXPECT_FALSE(literal(l,before));if(!before[id(l)]){++pending;only=l;}}EXPECT_EQ(pending,1);EXPECT_EQ(e.literal,only);ASSERT_EQ(before[id(e.literal)],0);before[id(e.literal)]=e.literal>0?1:-1;EXPECT_EQ(e.assignment,before);states[e.node]=e.assignment;}
else if(e.kind==DpllEventKind::pure){++pures;ASSERT_NE(e.literal,0);EXPECT_EQ(before[id(e.literal)],0);bool occurs=false;for(auto&c:f){bool sat=false;for(int l:c)sat|=literal(l,before);if(sat)continue;for(int l:c)if(id(l)==id(e.literal)){EXPECT_EQ(l,e.literal);occurs=true;}}EXPECT_TRUE(occurs);before[id(e.literal)]=e.literal>0?1:-1;EXPECT_EQ(e.assignment,before);states[e.node]=e.assignment;}
else{EXPECT_EQ(e.assignment,before);if(e.kind==DpllEventKind::conflict){++conflicts;ASSERT_LT(e.clause,f.size());for(int l:f[e.clause]){EXPECT_NE(before[id(l)],0);EXPECT_FALSE(literal(l,before));}failed.insert(e.node);last_failed=e.node;}
else if(e.kind==DpllEventKind::backtrack){++backs;EXPECT_EQ(parents[last_failed],e.node);EXPECT_EQ(e.literal,incoming[last_failed]);EXPECT_TRUE(failed.count(last_failed));}
else if(e.kind==DpllEventKind::exhausted){ASSERT_EQ(children[e.node].size(),2);EXPECT_TRUE(failed.count(children[e.node][0]));EXPECT_TRUE(failed.count(children[e.node][1]));EXPECT_EQ(incoming[children[e.node][0]],-incoming[children[e.node][1]]);EXPECT_GT(incoming[children[e.node][0]],0);failed.insert(e.node);last_failed=e.node;}
else if(e.kind==DpllEventKind::satisfied){EXPECT_TRUE(model(f,before));auto a=before;for(int v=1;v<=n;++v)if(!a[v])a[v]=1;EXPECT_EQ(a,t.solve.result.assignment);}
else FAIL()<<"unknown event";}}if(e.kind!=DpllEventKind::unit&&e.kind!=DpllEventKind::conflict)EXPECT_EQ(e.clause,f.size());}
const auto&w=t.solve.work;EXPECT_EQ(nodes,w.nodes);EXPECT_EQ(w.branches,nodes-1);EXPECT_EQ(w.decisions,children.size());EXPECT_EQ(units,w.unit_steps);EXPECT_EQ(pures,w.pure_steps);EXPECT_EQ(conflicts,w.conflicts);EXPECT_EQ(backs,w.backtracks);EXPECT_EQ(maxdepth,w.max_depth);EXPECT_EQ(entries,t.captured_entries);ASSERT_FALSE(t.events.empty());if(t.solve.result.satisfiable){EXPECT_EQ(t.events.back().kind,DpllEventKind::satisfied);EXPECT_TRUE(model(f,t.solve.result.assignment));EXPECT_TRUE(cnf_assignment_satisfies(n,f,t.solve.result.assignment));}else{EXPECT_TRUE(failed.count(0));EXPECT_TRUE(t.solve.result.assignment.empty());EXPECT_EQ(w.nodes,2*w.decisions+1);EXPECT_EQ(w.conflicts,w.decisions+1);EXPECT_EQ(w.backtracks,w.branches);} }
std::vector<std::vector<int>> clauses(int n){std::vector<std::vector<int>>out;int limit=1;for(int i=0;i<n;++i)limit*=3;for(int code=0;code<limit;++code){int x=code;std::vector<int>c;for(int v=1;v<=n;++v){int digit=x%3;x/=3;if(digit)c.push_back(digit==1?v:-v);}out.push_back(c);}return out;}
Formula legacy_generate(int n,std::size_t m,std::mt19937&rng){Formula f;for(std::size_t c=0;c<m;++c){std::vector<int>cl;while(cl.size()<3){const int v=1+static_cast<int>(rng()%n);bool dup=false;for(int l:cl)if(id(l)==v)dup=true;if(!dup)cl.push_back((rng()&1)?v:-v);}f.push_back(cl);}return f;}
}
TEST(DpllTrace, TinyPropagationOriginalAssignmentAndBacktracking){for(auto [n,f]:std::vector<std::pair<int,Formula>>{{2,{{1},{-1,2}}},{3,{{1,2},{-1,3},{-2,-3},{1,-3}}},{1,{{1},{-1}}},{2,{{1,2},{1,-2},{-1,2},{-1,-2}}},{3,{{-1,2},{-1,-2},{1,3}}},{0,{}},{0,{{}}}}){auto t=dpll_sat_trace(n,f);certificate(n,f,t);EXPECT_EQ(t.solve.result.satisfiable,brute(n,f));EXPECT_EQ(t.solve.result.assignment,dpll_sat(n,f).assignment);}EXPECT_EQ(dpll_sat(3,{{1,2},{-1,3},{-2,-3},{1,-3}}).assignment,(std::vector<int>{0,1,-1,1}));}
TEST(DpllTrace, ExhaustiveTwoVariableSetsAndOrderedThreeClauseThreeVariableFormulas){const auto two=clauses(2);for(unsigned mask=0;mask<(1u<<two.size());++mask){Formula f;for(std::size_t i=0;i<two.size();++i)if((mask>>i)&1)f.push_back(two[i]);auto t=dpll_sat_trace(2,f);certificate(2,f,t);EXPECT_EQ(t.solve.result.satisfiable,brute(2,f));}const auto three=clauses(3);for(auto&a:three)for(auto&b:three)for(auto&c:three){Formula f{a,b,c};auto t=dpll_sat_trace(3,f);certificate(3,f,t);EXPECT_EQ(t.solve.result.satisfiable,brute(3,f));} }
TEST(DpllTrace, DuplicateTautologicalUnusedAndSeededRawFormulasRemainSound){std::mt19937 rng(73);for(int trial=0;trial<800;++trial){int n=1+rng()%7,m=rng()%15;Formula f;for(int i=0;i<m;++i){std::vector<int>c;int width=rng()%6;for(int j=0;j<width;++j){int v=1+rng()%n;c.push_back(rng()&1?v:-v);}f.push_back(c);}auto t=dpll_sat_trace(n,f);certificate(n,f,t);EXPECT_EQ(t.solve.result.satisfiable,brute(n,f));auto plain=dpll_sat(n,f);EXPECT_EQ(plain.assignment,t.solve.result.assignment);auto work=dpll_sat_work(n,f);EXPECT_EQ(work.result.assignment,plain.assignment);EXPECT_EQ(work.work.literal_visits,t.solve.work.literal_visits);EXPECT_EQ(work.work.clause_visits,t.solve.work.clause_visits);}auto t=dpll_sat_trace(4,{{1,1},{2,-2}});certificate(4,t.clauses,t);EXPECT_EQ(t.solve.result.assignment[3],1);EXPECT_EQ(t.solve.result.assignment[4],1);}
TEST(DpllTrace, CompleteValidationCapsAndOwnership){auto t=dpll_sat_trace(2,{{1},{-1,2}});EXPECT_EQ(t.captured_entries,22);EXPECT_NO_THROW(dpll_sat_trace(2,{{1},{-1,2}},22));
EXPECT_THROW(dpll_sat_trace(2,{{1},{-1,2}},21),std::length_error);
EXPECT_NO_THROW(dpll_sat_trace(2,{{1},{-1,2}},0));
EXPECT_THROW(dpll_sat_trace(2,{{1},{-1,2}},9),std::length_error);
EXPECT_THROW(dpll_sat_trace(std::numeric_limits<int>::max(),{},1),std::length_error);
for(auto f:std::vector<Formula>{{{0}},{{3}},{{-3}},{{std::numeric_limits<int>::min()}},{{},{0}},{{1,-1,3}}}){EXPECT_THROW(dpll_sat(2,f),std::invalid_argument);
EXPECT_THROW(dpll_sat_trace(2,f,1),std::invalid_argument);}
EXPECT_THROW(dpll_sat(-1,{}),std::invalid_argument);
EXPECT_THROW(dpll_sat(std::numeric_limits<int>::max(),{{0}}),std::invalid_argument);
auto copy=t;t.clauses[0][0]=-1;t.events[1].assignment[1]=-1;EXPECT_EQ(copy.clauses[0][0],1);EXPECT_EQ(copy.events[1].assignment[1],1);}
TEST(DpllTrace, SuppliedModelValidationAndZeroVariables){EXPECT_TRUE(cnf_assignment_satisfies(0,{},{99}));EXPECT_FALSE(cnf_assignment_satisfies(0,{{}},{0}));EXPECT_FALSE(cnf_assignment_satisfies(2,{{1,2}},{0,-1,-1}));EXPECT_TRUE(cnf_assignment_satisfies(2,{{1,-2}},{99,-1,-1}));
EXPECT_THROW(cnf_assignment_satisfies(2,{},{}),std::invalid_argument);
EXPECT_THROW(cnf_assignment_satisfies(2,{}, {0,0,1}),std::invalid_argument);
EXPECT_THROW(cnf_assignment_satisfies(2,{{0}}, {0,1,1}),std::invalid_argument);}
TEST(DpllTrace, GeneratorMatchesLegacyStreamAndChecksBeforeEngineMutation){for(int n:{3,4,25})for(unsigned seed:{0u,4242u})for(std::size_t m:{0u,1u,20u}){std::mt19937 a(seed),b(seed);EXPECT_EQ(random_3sat_formula(n,m,a),legacy_generate(n,m,b));EXPECT_EQ(a,b);}std::mt19937 rng(4242),copy=rng;
EXPECT_THROW(random_3sat_formula(2,1,rng),std::invalid_argument);
EXPECT_EQ(rng,copy);
EXPECT_THROW(random_3sat_formula(3,4,rng,11),std::length_error);
EXPECT_EQ(rng,copy);
EXPECT_THROW(random_3sat_formula(3,std::numeric_limits<std::size_t>::max(),rng),std::length_error);
EXPECT_EQ(rng,copy);
EXPECT_NO_THROW(random_3sat_formula(3,4,rng,12));
EXPECT_NO_THROW(random_3sat_formula(3,4,rng,0));}
TEST(DpllTrace, ExplicitStackHandlesLongDecisionDepthWithoutCapturingQuadraticHistory){constexpr int n=1024;Formula f;for(int v=1;v<=n;++v)f.push_back({v,-v});auto r=dpll_sat_work(n,f);EXPECT_TRUE(r.result.satisfiable);EXPECT_EQ(r.work.max_depth,n);EXPECT_EQ(r.work.nodes,n+1);EXPECT_EQ(r.work.branches,n);EXPECT_EQ(r.work.backtracks,0);EXPECT_EQ(r.work.unit_steps,0);EXPECT_EQ(r.work.pure_steps,0);EXPECT_TRUE(cnf_assignment_satisfies(n,f,r.result.assignment));}
