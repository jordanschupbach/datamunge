#include <gtest/gtest.h>
#include <datamunge/algorithms/cyk_parser.hpp>
#include <algorithm>
#include <climits>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
CNFGrammar balanced() {CNFGrammar g;g.unary['a']={1};g.unary['b']={2};g.binary={{0,1,2},{0,1,3},{3,0,2}};return g;}
std::string word(std::size_t n,std::size_t mask) {std::string s(n,'a');for(auto& c:s) {c=mask%2?'b':'a';mask/=2;}return s;}
void verify_tree(const CNFGrammar& g,const CykAnalysis& a) {
  const auto nodes=cyk_first_parse(a);if(!a.accepted) {EXPECT_TRUE(nodes.empty());return;}
  ASSERT_EQ(nodes.size(),2*a.input.size()-1);
  ASSERT_EQ(nodes[0].nonterminal,g.start);std::string yield;
  for(std::size_t i=0;i<nodes.size();++i) {
    const auto& node=nodes[i];
  ASSERT_LE(node.begin+node.length,a.input.size());
  ASSERT_GT(node.length,0);
    if(node.length==1) {
      EXPECT_EQ(node.left_child,std::numeric_limits<std::size_t>::max());
  EXPECT_EQ(node.right_child,std::numeric_limits<std::size_t>::max());
      EXPECT_EQ(node.terminal,a.input[node.begin]);const auto it=g.unary.find(node.terminal);
  ASSERT_NE(it,g.unary.end());
      EXPECT_NE(std::find(it->second.begin(),it->second.end(),node.nonterminal),it->second.end());yield.push_back(node.terminal);
    }else {
      ASSERT_GT(node.left_child,i);
  ASSERT_GT(node.right_child,node.left_child);
  ASSERT_LT(node.right_child,nodes.size());
      const auto& left=nodes[node.left_child];const auto& right=nodes[node.right_child];
      EXPECT_EQ(left.begin,node.begin);
  EXPECT_EQ(right.begin,node.begin+left.length);
  EXPECT_EQ(left.length+right.length,node.length);
      const std::array<int,3> rule{node.nonterminal,left.nonterminal,right.nonterminal};
  EXPECT_NE(std::find(g.binary.begin(),g.binary.end(),rule),g.binary.end());
    }
  }
  EXPECT_EQ(yield,a.input);
}
// Independent bounded language generation: fixed point of terminal sets and
// grammar-language concatenation. It never reads input substrings or CYK cells.
std::vector<std::set<std::string>> generated(const CNFGrammar& g,const std::vector<int>& ids,std::size_t limit) {
  std::vector<std::set<std::string>> language(ids.size());
  const auto index=[&](int id){return static_cast<std::size_t>(std::find(ids.begin(),ids.end(),id)-ids.begin());};
  for(const auto& row:g.unary) for(int id:row.second) language[index(id)].insert(std::string(1,row.first));
  bool changed=true;while(changed) {
    changed=false;const auto previous=language;
    for(const auto& r:g.binary) for(const auto& left:previous[index(r[1])]) for(const auto& right:previous[index(r[2])])
      if(left.size()+right.size()<=limit) changed=language[index(r[0])].insert(left+right).second || changed;
  }
  return language;
}
}
TEST(CykChart, HandTwoAndFourByteChartsAndWitnesses) {
  const auto g=balanced();const auto a=cyk_analyze(g,"ab"),b=cyk_analyze(g,"aabb");
  EXPECT_TRUE(a.accepted);
  ASSERT_EQ(a.cells.size(),3);
  EXPECT_EQ(cyk_cell(a,0,1).entries[0].nonterminal,1);
  EXPECT_EQ(cyk_cell(a,1,1).entries[0].nonterminal,2);
  EXPECT_EQ(cyk_cell(a,0,2).entries[0].left_length,1);
  EXPECT_EQ(cyk_cell(a,0,2).entries[0].binary_rule,0);
  EXPECT_EQ(b.cells.size(),10);
  EXPECT_EQ(b.work.split_points,10);
  EXPECT_EQ(b.work.binary_rule_tests,18);
  EXPECT_EQ(b.work.skipped_empty_splits,4);
  EXPECT_EQ(b.work.new_states,7);
  EXPECT_TRUE(cyk_cell(b,0,2).entries.empty());
  EXPECT_EQ(cyk_cell(b,1,3).entries[0].nonterminal,3);
  EXPECT_EQ(cyk_cell(b,0,4).entries[0].binary_rule,1);
  verify_tree(g,a);verify_tree(g,b);
}
TEST(CykChart, OriginalEightExamplesAndAllBinaryWordsThroughTwelve) {
  const auto g=balanced();const std::vector<std::string> examples{"ab","aabb","aaabbb","aab","abb","ba","aabbb","aaaabbbb"};
  const std::vector<bool> expected{true,true,true,false,false,false,false,true};
  for(std::size_t i=0;i<examples.size();++i) EXPECT_EQ(cyk_recognize(g,examples[i]),expected[i]);
  std::size_t cases=0,accepted=0;
  for(std::size_t n=0;n<=12;++n) for(std::size_t mask=0;mask<(std::size_t{1}<<n);++mask) {
    const auto s=word(n,mask);const bool member=n>=2 && n%2==0 && s==std::string(n/2,'a')+std::string(n/2,'b');
    const auto a=cyk_analyze(g,s);
  EXPECT_EQ(a.accepted,member);
  EXPECT_EQ(cyk_recognize(g,s),member);verify_tree(g,a);++cases;accepted+=a.accepted;
  }
  EXPECT_EQ(cases,8191);
  EXPECT_EQ(accepted,6);
}
TEST(CykChart, SeededGrammarsMatchIndependentBoundedLanguageGeneration) {
  std::mt19937 rng(875);const std::vector<int> ids{INT_MIN,-7,INT_MAX};std::size_t cases=0;
  for(int trial=0;trial<100;++trial) {
    CNFGrammar g;g.start=ids[rng()%ids.size()];
    for(char c:{'a','b'}) for(int id:ids) if(rng()%2) {g.unary[c].push_back(id);if(rng()%3==0) g.unary[c].push_back(id);}
    const auto count=rng()%10;for(std::size_t r=0;r<count;++r) g.binary.push_back({ids[rng()%3],ids[rng()%3],ids[rng()%3]});
    const auto language=generated(g,ids,6);const auto start=static_cast<std::size_t>(std::find(ids.begin(),ids.end(),g.start)-ids.begin());
    for(std::size_t n=0;n<=6;++n) for(std::size_t mask=0;mask<(std::size_t{1}<<n);++mask) {
      const auto s=word(n,mask);const auto a=cyk_analyze(g,s);
  EXPECT_EQ(a.accepted,language[start].count(s)!=0);
  EXPECT_EQ(cyk_recognize(g,s),a.accepted);verify_tree(g,a);
      std::size_t states=0;for(const auto& cell:a.cells) {states+=cell.entries.size();for(std::size_t i=1;i<cell.entries.size();++i) EXPECT_LT(cell.entries[i-1].nonterminal,cell.entries[i].nonterminal);}
      EXPECT_EQ(states,a.work.new_states);
  EXPECT_LE(a.work.binary_rule_tests,a.shape.max_rule_tests);
  EXPECT_EQ(a.work.split_points,a.shape.splits);++cases;
    }
  }
  EXPECT_EQ(cases,12700);
}
TEST(CykChart, AmbiguityDuplicateRulesAndDeterministicFirstRuleChoice) {
  CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};const auto a=cyk_analyze(g,"aaaa");
  EXPECT_EQ(a.work.matching_rules,10);
  EXPECT_EQ(a.work.new_states,10);
  EXPECT_EQ(cyk_cell(a,0,4).entries[0].left_length,1);verify_tree(g,a);
  g.unary['a']={0,0,0};g.binary.push_back({0,0,0});const auto b=cyk_analyze(g,"aaaa");
  EXPECT_EQ(b.work.terminal_rule_tests,12);
  EXPECT_EQ(b.work.binary_rule_tests,20);
  EXPECT_EQ(b.work.matching_rules,20);
  EXPECT_EQ(b.work.new_states,a.work.new_states);
  EXPECT_EQ(cyk_cell(b,0,4).entries[0].binary_rule,0);verify_tree(g,b);
  CNFGrammar h;h.unary['a']={1,3};h.unary['b']={2,4};h.binary={{0,1,2},{0,3,4}};
  const auto first=cyk_analyze(h,"ab");
  EXPECT_EQ(cyk_first_parse(first)[1].nonterminal,1);
  std::reverse(h.binary.begin(),h.binary.end());const auto second=cyk_analyze(h,"ab");
  EXPECT_EQ(cyk_first_parse(second)[1].nonterminal,3);
  EXPECT_TRUE(second.accepted);
  verify_tree(h,second);
}
TEST(CykChart, AllByteTerminalsNulUtf8AndSignedSparseIds) {
  CNFGrammar g;g.start=INT_MIN;
  for(unsigned byte=0;byte<256;++byte) g.unary[static_cast<char>(byte)]={INT_MIN,INT_MAX};
  for(unsigned byte=0;byte<256;++byte) {const std::string s(1,static_cast<char>(byte));const auto a=cyk_analyze(g,s);
  EXPECT_TRUE(a.accepted);
  EXPECT_EQ(a.nonterminals,(std::vector<int>{INT_MIN,INT_MAX}));verify_tree(g,a);}
  EXPECT_FALSE(cyk_recognize(g,""));
  EXPECT_FALSE(cyk_recognize(g,std::string(2,'\0')));
  CNFGrammar utf;utf.unary[static_cast<char>(0xc3)]={-7};utf.unary[static_cast<char>(0xa9)]={INT_MAX};utf.binary={{0,-7,INT_MAX}};
  const auto a=cyk_analyze(utf,std::string("\xc3\xa9",2));
  EXPECT_TRUE(a.accepted);
  EXPECT_EQ(a.input.size(),2);verify_tree(utf,a);
  EXPECT_FALSE(cyk_recognize(utf,"x"));
}
TEST(CykChart, ExactShapeArithmeticOverflowAndCompletePotentialCaps) {
  for(std::size_t n=0;n<=100;++n) {
    std::size_t cells=0,splits=0;for(std::size_t len=1;len<=n;++len) for(std::size_t begin=0;begin+len<=n;++begin) {++cells;for(std::size_t k=1;k<len;++k) ++splits;}
    const auto p=cyk_shape(n,7,11);
  EXPECT_EQ(p.cells,cells);
  EXPECT_EQ(p.splits,splits);
  EXPECT_EQ(p.state_slots,cells*7);
  EXPECT_EQ(p.max_rule_tests,splits*11);
  }
  EXPECT_THROW(cyk_shape(1,0,1),std::invalid_argument);
  EXPECT_THROW(cyk_shape(std::numeric_limits<std::size_t>::max(),1,0),std::length_error);
  EXPECT_THROW(cyk_shape(2,std::numeric_limits<std::size_t>::max(),0),std::length_error);
  EXPECT_THROW(cyk_shape(3,1,std::numeric_limits<std::size_t>::max()),std::length_error);
  if constexpr(std::numeric_limits<std::size_t>::digits>=64) {const auto p=cyk_shape(3000000,1,0);
  EXPECT_EQ(p.cells,4500001500000ULL);
  EXPECT_EQ(p.splits,4499999999999500000ULL);}
  CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
  EXPECT_THROW(cyk_analyze(g,"aaaa",9,0),std::length_error);
  EXPECT_THROW(cyk_analyze(g,"bbbb",0,9),std::length_error);
  EXPECT_NO_THROW(cyk_analyze(g,"aaaa",10,10));
  const auto a=cyk_analyze(g,"aaaa",0,0);
  EXPECT_THROW(cyk_first_parse(a,6),std::length_error);
  EXPECT_EQ(cyk_first_parse(a,7).size(),7);
  EXPECT_EQ(cyk_first_parse(a,0).size(),7);
  EXPECT_TRUE(cyk_first_parse(cyk_analyze(g,"b"),1).empty());
}
TEST(CykChart, OwnedResultsBorrowBoundsAndMalformedVisitedWitnesses) {
  auto g=balanced();std::string s="aabb";const auto a=cyk_analyze(g,s);g.binary.clear();s="xxxx";
  EXPECT_EQ(a.input,"aabb");
  EXPECT_EQ(cyk_first_parse(a).size(),7);
  EXPECT_THROW(cyk_cell(a,0,0),std::out_of_range);
  EXPECT_THROW(cyk_cell(a,4,1),std::out_of_range);
  EXPECT_THROW(cyk_cell(a,0,5),std::out_of_range);
  auto bad=a;bad.cells.pop_back();
  EXPECT_THROW(cyk_cell(bad,0,1),std::invalid_argument);
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  bad=a;bad.cells[0].begin=9;
  EXPECT_THROW(cyk_cell(bad,0,1),std::invalid_argument);
  bad=a;bad.accepted=false;
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  bad=a;bad.cells[0].entries.clear();
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  bad=a;bad.cells.back().entries[0].left_length=4;
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  bad=a;bad.cells[0].entries[0].terminal='b';
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  bad=a;bad.cells.back().entries.push_back(bad.cells.back().entries[0]);
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  bad=a;bad.cells.back().entries[0].binary_rule=3;
  EXPECT_THROW(cyk_first_parse(bad),std::invalid_argument);
  CykAnalysis empty;empty.accepted=true;
  EXPECT_THROW(cyk_first_parse(empty),std::invalid_argument);
}
TEST(CykChart, ActualCubicRuleTestsAndEmptyChildSkipsAreSeparate) {
  CNFGrammar g;g.unary['a']={0};g.binary={{0,0,0}};
  for(std::size_t n=1;n<=64;++n) {const auto a=cyk_analyze(g,std::string(n,'a'));
  EXPECT_EQ(a.work.split_points,a.shape.splits);
  EXPECT_EQ(a.work.binary_rule_tests,a.shape.splits);
  EXPECT_EQ(a.work.matching_rules,a.shape.splits);
  EXPECT_EQ(a.work.skipped_empty_splits,0);
  EXPECT_EQ(a.work.new_states,a.shape.cells);verify_tree(g,a);}
  g.unary.clear();const auto a=cyk_analyze(g,"aaaa");
  EXPECT_FALSE(a.accepted);
  EXPECT_EQ(a.work.split_points,10);
  EXPECT_EQ(a.work.skipped_empty_splits,10);
  EXPECT_EQ(a.work.binary_rule_tests,0);
  EXPECT_EQ(a.work.new_states,0);
  const auto empty=cyk_analyze(g,"");
  EXPECT_FALSE(empty.accepted);
  EXPECT_TRUE(empty.cells.empty());
  EXPECT_EQ(empty.work.terminal_lookups,0);
  EXPECT_EQ(empty.nonterminals,(std::vector<int>{0}));
}
TEST(CykChart, IterativeRightBranchWitnessPreservesPreorderAndAllLeaves) {
  CNFGrammar g;g.start=-7;g.unary['a']={-7};g.binary={{-7,-7,-7}};
  const auto a=cyk_analyze(g,std::string(200,'a'));verify_tree(g,a);const auto tree=cyk_first_parse(a);
  EXPECT_EQ(tree.size(),399);
  std::size_t index=0;for(std::size_t remaining=200;remaining>1;--remaining) {ASSERT_LT(index,tree.size());
  EXPECT_EQ(tree[index].length,remaining);
  EXPECT_EQ(tree[tree[index].left_child].length,1);index=tree[index].right_child;}
}
