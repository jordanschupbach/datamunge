#include <gtest/gtest.h>
#include <datamunge/algorithms/davis_putnam.hpp>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
using CNF=std::vector<std::vector<int>>;
bool evaluate(const CNF& f,unsigned assignment) {
  for(const auto& c:f) {
    bool sat=false;for(int l:c) {const auto v=static_cast<unsigned>(l<0?-l:l);sat |= (((assignment>>(v-1))&1)!=0)==(l>0);}
    if(!sat) return false;
  }
  return true;
}
bool brute(int n,const CNF& f) {
  for(unsigned a=0;a<(1u<<n);++a) if(evaluate(f,a)) return true;
  return false;
}
void canonical(const CNF& f) {
  EXPECT_TRUE(std::is_sorted(f.begin(),f.end()));EXPECT_EQ(std::adjacent_find(f.begin(),f.end()),f.end());
  for(const auto& c:f) {
    EXPECT_TRUE(std::is_sorted(c.begin(),c.end()));EXPECT_EQ(std::adjacent_find(c.begin(),c.end()),c.end());
    for(int l:c) EXPECT_FALSE(std::binary_search(c.begin(),c.end(),-l));
  }
}
void check(int n,const CNF& f) {
  const auto t=davis_putnam_trace(n,f);EXPECT_EQ(t.satisfiable,brute(n,f));EXPECT_EQ(t.satisfiable,davis_putnam(n,f));
  canonical(t.initial);for(unsigned a=0;a<(1u<<n);++a) EXPECT_EQ(evaluate(f,a),evaluate(t.initial,a));
  CNF before=t.initial;DpWork work;work.peak_clauses=before.size();std::set<int> removed;
  for(const auto& s:t.steps) {
    canonical(s.after);ASSERT_NE(s.literal,0);const int x=s.literal<0?-s.literal:s.literal;
    EXPECT_TRUE(removed.insert(x).second);EXPECT_EQ(s.before_clauses,before.size());
    for(const auto& c:s.after) for(int l:c) EXPECT_NE(l<0?-l:l,x);
    for(unsigned a=0;a<(1u<<n);++a) {
      const unsigned bit=1u<<(x-1);
      EXPECT_EQ(evaluate(s.after,a),evaluate(before,a|bit)||evaluate(before,a&~bit));
    }
    if(s.rule==DpRule::resolution) {
      ASSERT_GT(s.literal,0);std::vector<std::vector<int>> pos,neg;std::set<std::vector<int>> next;
      for(const auto& c:before) {
        if(std::binary_search(c.begin(),c.end(),x)) pos.push_back(c);
        else if(std::binary_search(c.begin(),c.end(),-x)) neg.push_back(c);
        else next.insert(c);
      }
      EXPECT_EQ(s.positive_clauses,pos.size());EXPECT_EQ(s.negative_clauses,neg.size());EXPECT_EQ(s.pairs,pos.size()*neg.size());
      std::size_t taut=0,dup=0;
      for(const auto& p:pos) for(const auto& q:neg) {
        std::vector<int> c;for(int l:p) if(l!=x) c.push_back(l);for(int l:q) if(l!=-x) c.push_back(l);
        std::sort(c.begin(),c.end());c.erase(std::unique(c.begin(),c.end()),c.end());
        bool bad=false;for(int l:c) if(std::binary_search(c.begin(),c.end(),-l)) bad=true;
        if(bad) ++taut;else if(!next.insert(c).second) ++dup;
      }
      EXPECT_EQ(CNF(next.begin(),next.end()),s.after);EXPECT_EQ(s.tautologies,taut);EXPECT_EQ(s.duplicates,dup);
      ++work.resolution_steps;work.resolution_pairs+=s.pairs;work.tautological_resolvents+=taut;work.duplicate_resolvents+=dup;
    } else {
      EXPECT_EQ(s.pairs,0);EXPECT_EQ(s.positive_clauses,0);EXPECT_EQ(s.negative_clauses,0);EXPECT_EQ(s.tautologies,0);EXPECT_EQ(s.duplicates,0);
      if(s.rule==DpRule::unit) ++work.unit_steps;else {EXPECT_EQ(s.rule,DpRule::pure);++work.pure_steps;}
    }
    work.peak_clauses=std::max(work.peak_clauses,s.after.size());before=s.after;
  }
  EXPECT_EQ(t.satisfiable,before.empty());if(!t.satisfiable) {ASSERT_FALSE(before.empty());EXPECT_TRUE(before[0].empty());}
  EXPECT_EQ(work.unit_steps,t.work.unit_steps);EXPECT_EQ(work.pure_steps,t.work.pure_steps);EXPECT_EQ(work.resolution_steps,t.work.resolution_steps);
  EXPECT_EQ(work.resolution_pairs,t.work.resolution_pairs);EXPECT_EQ(work.tautological_resolvents,t.work.tautological_resolvents);
  EXPECT_EQ(work.duplicate_resolvents,t.work.duplicate_resolvents);EXPECT_EQ(work.peak_clauses,t.work.peak_clauses);
}
}
TEST(DavisTrace, OriginalFormulasAndMechanism) {
  check(3,{{1,2},{-1,3}});check(1,{{1},{-1}});check(2,{{1,2},{1,-2},{-1,2},{-1,-2}});
  check(4,{{1,-2,3},{-1,2,4},{2,-3,-4},{1,3,-4},{-1,-2,4}});
  const auto t=davis_putnam_trace(2,{{1,2},{1,-2},{-1,2},{-1,-2}});
  ASSERT_EQ(t.steps.size(),2);EXPECT_EQ(t.steps[0].rule,DpRule::resolution);EXPECT_EQ(t.steps[0].literal,1);
  EXPECT_EQ(t.steps[0].after,(CNF{{-2},{2}}));EXPECT_EQ(t.steps[0].pairs,4);EXPECT_EQ(t.steps[0].tautologies,2);
  EXPECT_EQ(t.steps[1].rule,DpRule::unit);EXPECT_EQ(t.steps[1].after,(CNF{{}}));
}
TEST(DavisTrace, AllTwoVariableFormulasAndThreeClauseThreeVariableFormulas) {
  std::vector<CNF> universes;
  for(int n:{2,3}) {
    CNF clauses;unsigned total=1;for(int v=0;v<n;++v) total*=3;
    for(unsigned code=0;code<total;++code) {
      auto k=code;std::vector<int> c;for(int v=1;v<=n;++v) {if(k%3==1)c.push_back(v);if(k%3==2)c.push_back(-v);k/=3;}clauses.push_back(c);
    }
    check(n,{});
    if(n==2) {for(unsigned mask=1;mask<(1u<<clauses.size());++mask) {CNF f;for(std::size_t j=0;j<clauses.size();++j) if(mask&(1u<<j)) f.push_back(clauses[j]);check(n,f);}}
    else for(std::size_t a=0;a<clauses.size();++a) {
      check(n,{clauses[a]});for(std::size_t b=a+1;b<clauses.size();++b) {
        check(n,{clauses[a],clauses[b]});for(std::size_t c=b+1;c<clauses.size();++c) check(n,{clauses[a],clauses[b],clauses[c]});
      }
    }
  }
}
TEST(DavisTrace, SeededRandomTruthTablesAndStateProjection) {
  std::mt19937 rng(1960);
  for(int trial=0;trial<1500;++trial) {
    const int n=1+rng()%7;CNF f;
    for(unsigned c=0,nc=rng()%25;c<nc;++c) {std::vector<int> row;for(unsigned j=0,nl=rng()%6;j<nl;++j) {int v=1+rng()%n;row.push_back(rng()%2?v:-v);}f.push_back(row);}
    check(n,f);
  }
}
TEST(DavisTrace, NormalizationEmptyOwnershipAndSparseExtremes) {
  check(0,{});check(0,{{}});check(2,{{2,1,1},{1,2},{1,-1},{-2,2},{1,2}});
  CNF f{{1,1},{1},{-2,2}};const auto saved=f;auto t=davis_putnam_trace(2,f);EXPECT_EQ(f,saved);f.clear();EXPECT_EQ(t.initial,(CNF{{1}}));
  EXPECT_TRUE(davis_putnam(INT_MAX,{{INT_MAX},{1,-1}}));EXPECT_FALSE(davis_putnam(INT_MAX,{{INT_MAX},{-INT_MAX}}));
  EXPECT_TRUE(davis_putnam(INT_MAX,{}));
  const auto p=davis_putnam_trace(3,{{1,2},{-1,3}});ASSERT_FALSE(p.steps.empty());EXPECT_EQ(p.steps[0].rule,DpRule::pure);EXPECT_EQ(p.steps[0].literal,2);
}
TEST(DavisTrace, ValidationPrecedesTrivialVerdictAndCaps) {
  EXPECT_THROW(davis_putnam(-1,{}),std::invalid_argument);
  EXPECT_THROW(davis_putnam(0,{{1,-1}}),std::invalid_argument);
  EXPECT_THROW(davis_putnam(1,{{},{0}}),std::invalid_argument);
  EXPECT_THROW(davis_putnam(1,{{},{2,-2}}),std::invalid_argument);
  EXPECT_THROW(davis_putnam(INT_MAX,{{INT_MIN}}),std::invalid_argument);
  EXPECT_THROW(davis_putnam_trace(1,{{1},{2}},1,1,1),std::invalid_argument);
}
TEST(DavisTrace, ExactCompleteCapsAndZeroDisable) {
  const CNF four{{1,2},{1,-2},{-1,2},{-1,-2}};
  EXPECT_NO_THROW(davis_putnam_trace(2,four,4,4,20));
  EXPECT_THROW(davis_putnam_trace(2,four,3,4,20),std::length_error);
  EXPECT_THROW(davis_putnam_trace(2,four,4,3,20),std::length_error);
  // initial 1+4+8=13, after resolution 1+2+2=5, after unit 1+1+0=2: total20.
  EXPECT_THROW(davis_putnam_trace(2,four,4,4,19),std::length_error);
  EXPECT_NO_THROW(davis_putnam_trace(2,four,0,0,0));
  EXPECT_NO_THROW(davis_putnam_trace(1,{{1},{1},{1}},1,1,4));
  EXPECT_NO_THROW(davis_putnam_trace(0,{},1,1,1));
  EXPECT_THROW(davis_putnam_trace(1,{{1}},1,1,3),std::length_error);
  // For k=3 input8 fits but first next state11 exceeds10.
  EXPECT_THROW(davis_putnam_trace(7,davis_putnam_grid_formula(3),10),std::length_error);
}
TEST(DavisTrace, ParentGridGrowthAndCumulativePairs) {
  for(std::size_t k=2;k<=8;++k) {
    const auto f=davis_putnam_grid_formula(k);ASSERT_EQ(f.size(),2*k+2);std::size_t literals=0;for(const auto& c:f)literals+=c.size();EXPECT_EQ(literals,6*k);
    const auto t=davis_putnam_trace(2*k+1,f);ASSERT_FALSE(t.steps.empty());const auto& s=t.steps[0];
    EXPECT_FALSE(t.satisfiable);EXPECT_EQ(s.rule,DpRule::resolution);EXPECT_EQ(s.literal,1);EXPECT_EQ(s.pairs,k*k);EXPECT_EQ(s.after.size(),k*k+2);
    EXPECT_EQ(s.positive_clauses,k);EXPECT_EQ(s.negative_clauses,k);EXPECT_EQ(s.tautologies,0);EXPECT_EQ(s.duplicates,0);
    if(k<=3)check(2*k+1,f);
    EXPECT_NO_THROW(davis_putnam_trace(2*k+1,f,0,t.work.resolution_pairs,0));
    EXPECT_THROW(davis_putnam_trace(2*k+1,f,0,t.work.resolution_pairs-1,0),std::length_error);
  }
  check(3,davis_putnam_grid_formula(1));
  EXPECT_THROW(davis_putnam_grid_formula(0),std::invalid_argument);
  EXPECT_THROW(davis_putnam_grid_formula(std::numeric_limits<std::size_t>::max(),0),std::length_error);
  EXPECT_THROW(davis_putnam_grid_formula(3,7),std::length_error);
  EXPECT_NO_THROW(davis_putnam_grid_formula(3,8));
}
TEST(DavisTrace, DuplicateResolventsAreActualWork) {
  const CNF f{{1,2},{1,2,3},{-1,2},{-1,2,3},{-2,-3}};
  check(3,f);const auto t=davis_putnam_trace(3,f);EXPECT_GT(t.work.duplicate_resolvents,0);
  auto reordered=f;std::reverse(reordered.begin(),reordered.end());for(auto& c:reordered)std::reverse(c.begin(),c.end());
  const auto r=davis_putnam_trace(3,reordered);EXPECT_EQ(r.initial,t.initial);EXPECT_EQ(r.work.resolution_pairs,t.work.resolution_pairs);ASSERT_EQ(r.steps.size(),t.steps.size());
  for(std::size_t i=0;i<t.steps.size();++i) {EXPECT_EQ(r.steps[i].literal,t.steps[i].literal);EXPECT_EQ(r.steps[i].after,t.steps[i].after);}
}
