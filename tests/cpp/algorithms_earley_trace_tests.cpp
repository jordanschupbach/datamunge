#include <gtest/gtest.h>
#include <datamunge/algorithms/earley_parser.hpp>
#include <algorithm>
#include <map>
#include <random>
#include <set>
#include <tuple>
using namespace datamunge::algorithms;
namespace {
ContextFreeGrammar anbn(){return {{{"S",{"a","S","b"}},{"S",{"a","b"}}},"S"};}
// Independent least fixed point of CFG interval languages; no Earley items or agenda.
bool language(const ContextFreeGrammar& g,const std::vector<std::string>& tokens){
 const auto names=g.nonterminals();const auto n=tokens.size();
 std::map<std::string,std::vector<std::vector<bool>>> spans;
 for(const auto& name:names)spans[name]=std::vector<std::vector<bool>>(n+1,std::vector<bool>(n+1));
 bool changed;
 do{changed=false;for(const auto& rule:g.productions)for(std::size_t i=0;i<=n;++i)for(std::size_t j=i;j<=n;++j){
  std::set<std::size_t> positions{i};
  for(const auto& symbol:rule.rhs){std::set<std::size_t> next;
   for(auto p:positions){if(names.count(symbol)){for(auto q=p;q<=j;++q)if(spans[symbol][p][q])next.insert(q);}
    else if(p<j&&tokens[p]==symbol)next.insert(p+1);}
   positions=std::move(next);
  }
  if(positions.count(j)&&!spans[rule.lhs][i][j]){spans[rule.lhs][i][j]=true;changed=true;}
 }}while(changed);
 const auto found=spans.find(g.start);return found!=spans.end()&&found->second[0][n];
}
void certificate(const EarleyTrace& t){
 const auto n=t.tokens.size();
 ASSERT_EQ(t.chart.size(),n+1);
 ASSERT_EQ(t.reasons.size(),n+1);
 const auto nt=t.grammar.nonterminals();std::uint64_t items=0,seeds=0,predictions=0,batches=0,checks=0,matches=0,joins=0,late=0;
 std::map<std::string,std::size_t> groups;for(const auto& p:t.grammar.productions){++groups[p.lhs];seeds+=p.lhs==t.grammar.start;}
 // (LHS,origin,end) -> first inserted completed item ordinal.
 std::map<std::tuple<std::string,std::size_t,std::size_t>,std::size_t> completed;
 bool accepted=false;
 for(std::size_t k=0;k<=n;++k){ASSERT_EQ(t.chart[k].size(),t.reasons[k].size());std::set<std::tuple<int,int,int>> unique;std::set<std::string> requests;
  for(std::size_t i=0;i<t.chart[k].size();++i){const auto e=t.chart[k][i];++items;
   ASSERT_GE(e.prod,0);
 ASSERT_LT(e.prod,t.grammar.productions.size());
 ASSERT_GE(e.dot,0);
 ASSERT_GE(e.origin,0);
 ASSERT_LE(e.origin,k);
   const auto& p=t.grammar.productions[e.prod];
 ASSERT_LE(e.dot,p.rhs.size());
 EXPECT_TRUE(unique.emplace(e.prod,e.dot,e.origin).second);
   const auto& reason=t.reasons[k][i];const auto here=std::pair{k,i};
   if(reason.action==EarleyAction::seed){EXPECT_EQ(k,0);
 EXPECT_EQ(e.dot,0);
 EXPECT_EQ(e.origin,0);
 EXPECT_EQ(p.lhs,t.grammar.start);
 EXPECT_EQ(reason.first.column,earley_no_item);
 EXPECT_EQ(reason.first.index,earley_no_item);}
   else{const auto a=reason.first;
 ASSERT_LT(a.column,t.chart.size());
 ASSERT_LT(a.index,t.chart[a.column].size());
 EXPECT_TRUE((std::pair{a.column,a.index}<here));
    const auto parent=t.chart[a.column][a.index];const auto& pp=t.grammar.productions[parent.prod];
 ASSERT_LT(parent.dot,pp.rhs.size());
    if(reason.action==EarleyAction::predict){EXPECT_EQ(a.column,k);
 EXPECT_EQ(e.origin,k);
 EXPECT_EQ(e.dot,0);
 EXPECT_EQ(pp.rhs[parent.dot],p.lhs);
 EXPECT_TRUE(nt.count(p.lhs));}
    else if(reason.action==EarleyAction::scan){ASSERT_LT(a.column,n);
 EXPECT_EQ(k,a.column+1);
 EXPECT_EQ(e.prod,parent.prod);
 EXPECT_EQ(e.origin,parent.origin);
 EXPECT_EQ(e.dot,parent.dot+1);
 EXPECT_FALSE(nt.count(pp.rhs[parent.dot]));
 EXPECT_EQ(pp.rhs[parent.dot],t.tokens[a.column]);}
    else{ASSERT_EQ(reason.action,EarleyAction::complete);const auto b=reason.second;
 ASSERT_LT(b.column,t.chart.size());
 ASSERT_LT(b.index,t.chart[b.column].size());
 EXPECT_TRUE((std::pair{b.column,b.index}<here));
     const auto c=t.chart[b.column][b.index];const auto& cp=t.grammar.productions[c.prod];
 EXPECT_EQ(c.dot,cp.rhs.size());
 EXPECT_EQ(c.origin,a.column);
 EXPECT_EQ(b.column,k);
 EXPECT_EQ(pp.rhs[parent.dot],cp.lhs);
 EXPECT_EQ(e.prod,parent.prod);
 EXPECT_EQ(e.dot,parent.dot+1);
 EXPECT_EQ(e.origin,parent.origin);
    }
   }
   if(reason.action!=EarleyAction::complete){EXPECT_EQ(reason.second.column,earley_no_item);
 EXPECT_EQ(reason.second.index,earley_no_item);}
   if(static_cast<std::size_t>(e.dot)==p.rhs.size()){
    completed.try_emplace({p.lhs,e.origin,k},i);
    accepted|=k==n&&e.origin==0&&p.lhs==t.grammar.start;
   }else if(nt.count(p.rhs[e.dot]))requests.insert(p.rhs[e.dot]);
   else{++checks;if(k<n&&p.rhs[e.dot]==t.tokens[k])++matches;}
  }
  batches+=requests.size();for(const auto& symbol:requests)predictions+=groups[symbol];
 }
 // Count final waiter/span pairs independently of the processing algorithm.
 for(const auto& [key,first]:completed){const auto& [symbol,origin,end]=key;
  for(std::size_t wi=0;wi<t.chart[origin].size();++wi){const auto e=t.chart[origin][wi];const auto& p=t.grammar.productions[e.prod];
   if(static_cast<std::size_t>(e.dot)<p.rhs.size()&&p.rhs[e.dot]==symbol){++joins;if(origin==end&&wi>first)++late;}}
 }
 const auto& w=t.work;
 EXPECT_EQ(t.accepted,accepted);
 EXPECT_EQ(w.unique_items,items);
 EXPECT_EQ(w.items_processed,items);
 EXPECT_EQ(w.prediction_batches,batches);
 EXPECT_EQ(w.predicted_productions,predictions);
 EXPECT_EQ(w.scan_checks,checks);
 EXPECT_EQ(w.scan_matches,matches);
 EXPECT_EQ(w.completed_spans,completed.size());
 EXPECT_EQ(w.completion_joins,joins);
 EXPECT_EQ(w.nullable_joins,late);
 EXPECT_EQ(w.insert_attempts,seeds+predictions+matches+joins);
 EXPECT_EQ(w.duplicate_items,w.insert_attempts-items);
 EXPECT_EQ(t.captured_entries,earley_trace_base_entries(t.grammar,n)+2*items);
}
}
TEST(EarleyTrace, OriginalLanguageAcrossAllShortBinaryWordsAndTinyDeductions){
 const auto g=anbn();
 for(int n=0;n<=12;++n)for(int bits=0;bits<(1<<n);++bits){std::vector<std::string> tokens(n);for(int i=0;i<n;++i)tokens[i]=(bits>>i)&1?"b":"a";
  bool expected=n>0&&n%2==0;for(int i=0;i<n;++i)expected&=tokens[i]==(i<n/2?"a":"b");
  const auto t=earley_trace(g,tokens);
 EXPECT_EQ(t.accepted,expected);certificate(t);
 }
 const ContextFreeGrammar tiny{{{"S",{"A"}},{"A",{"a"}}},"S"};const auto t=earley_trace(tiny,{"a"});certificate(t);
 EXPECT_TRUE(t.accepted);
 EXPECT_EQ(t.work.unique_items,4);
 EXPECT_EQ(t.work.completion_joins,1);
 EXPECT_EQ(t.captured_entries,16);
 EXPECT_EQ(t.reasons[0][0].action,EarleyAction::seed);
 EXPECT_EQ(t.reasons[0][1].action,EarleyAction::predict);
 EXPECT_EQ(t.reasons[1][0].action,EarleyAction::scan);
 EXPECT_EQ(t.reasons[1][1].action,EarleyAction::complete);
}
TEST(EarleyTrace, RandomNullableCyclicGrammarsMatchIndependentIntervalLanguages){
 std::mt19937 rng(691);const std::vector<std::string> names{"S","A","B"},symbols{"S","A","B","a","b"};
 for(int trial=0;trial<120;++trial){ContextFreeGrammar g;g.start="S";
  for(int i=0;i<9;++i){Production p;p.lhs=names[i%3];const auto length=rng()%4;for(std::size_t j=0;j<length;++j)p.rhs.push_back(symbols[rng()%symbols.size()]);g.productions.push_back(p);}
  for(int n=0;n<=5;++n)for(int bits=0;bits<(1<<n);++bits){std::vector<std::string> tokens(n);for(int i=0;i<n;++i)tokens[i]=(bits>>i)&1?"b":"a";
   const auto t=earley_trace(g,tokens);
 EXPECT_EQ(t.accepted,language(g,tokens));certificate(t);}
 }
}
TEST(EarleyTrace, NullableWaitersUnitCyclesAndRuleOrderAllReachClosure){
 ContextFreeGrammar g{{{"S",{"A","B"}},{"A",{}},{"B",{"A"}}},"S"};
 for(int permutation=0;permutation<6;++permutation){const auto t=earley_trace(g,{});
 EXPECT_TRUE(t.accepted);
 EXPECT_GT(t.work.nullable_joins,0);certificate(t);std::rotate(g.productions.begin(),g.productions.begin()+1,g.productions.end());if(permutation==2)std::reverse(g.productions.begin(),g.productions.end());}
 ContextFreeGrammar cycles{{{"S",{"S","S"}},{"S",{}},{"S",{"a"}}},"S"};
 for(int n=0;n<=12;++n){auto t=earley_trace(cycles,std::vector<std::string>(n,"a"));
 EXPECT_TRUE(t.accepted);certificate(t);}
 ContextFreeGrammar unproductive{{{"S",{"A"}},{"A",{"B"}},{"B",{"A"}}},"S"};
 EXPECT_FALSE(earley_recognize(unproductive,{}));
 const auto original=earley_trace(cycles,{"a","a"});cycles.productions.push_back(cycles.productions[2]);const auto duplicated=earley_trace(cycles,{"a","a"});
 EXPECT_EQ(original.accepted,duplicated.accepted);
 EXPECT_GT(duplicated.work.unique_items,original.work.unique_items);certificate(duplicated);
}
TEST(EarleyTrace, OpaqueSymbolsEmptyTokensAndMissingStartHavePreciseSemantics){
 EXPECT_FALSE(earley_recognize({},{}));
 EXPECT_EQ(earley_chart_sizes({}, {"a","b"}),(std::vector<int>{0,0,0}));
 ContextFreeGrammar g{{{"S",{"a"}}},"missing"};auto t=earley_trace(g,{"a"});
 EXPECT_FALSE(t.accepted);certificate(t);
 const std::string nul(1,'\0');g={{{"",{nul,""}},{"",{}}},""};t=earley_trace(g,{nul,nul});
 EXPECT_TRUE(t.accepted);certificate(t);
 g={{{"S",{""}}},"S"};
 EXPECT_FALSE(earley_recognize(g,{}));
 EXPECT_TRUE(earley_recognize(g,{""}));certificate(earley_trace(g,{""}));
 g={{{"起点",{"id",nul}}},"起点"};
 EXPECT_TRUE(earley_recognize(g,{"id",nul}));
 EXPECT_FALSE(earley_recognize(g,{"i","d",nul}));certificate(earley_trace(g,{"id",nul}));
}
TEST(EarleyTrace, CompleteAndOnlineCaptureCapsOwnershipAndSelectedItemValidation){
 auto g=anbn();std::vector<std::string> word{"a","a","a","a","b","b","b","b"};const auto full=earley_trace(g,word);
 EXPECT_EQ(full.captured_entries,75);
 EXPECT_EQ(earley_trace_base_entries(g,8),25);
 EXPECT_EQ(earley_trace(g,word,75).chart,full.chart);
 EXPECT_EQ(earley_trace(g,word,0).chart,full.chart);
 EXPECT_THROW((void)earley_trace(g,word,74),std::length_error);
 EXPECT_THROW((void)earley_trace(g,word,24),std::length_error);
 EXPECT_THROW((void)earley_trace_base_entries(g,static_cast<std::size_t>(std::numeric_limits<int>::max())+1),std::length_error);
 auto copy=full;g.start="changed";word[0]="changed";copy.grammar.start="other";copy.tokens[0]="other";copy.chart[0][0].dot=99;
 EXPECT_EQ(full.grammar.start,"S");
 EXPECT_EQ(full.tokens[0],"a");
 EXPECT_EQ(full.chart[0][0].dot,0);
 EXPECT_NE(earley_item_text(full,{0,0}).find("p0"),std::string::npos);
 EXPECT_THROW((void)earley_item_text(full,{}),std::invalid_argument);
 EXPECT_THROW((void)earley_item_text(copy,{0,0}),std::invalid_argument);
 auto bad=full;bad.chart[0][0].prod=-1;
 EXPECT_THROW((void)earley_item_text(bad,{0,0}),std::invalid_argument);
 ContextFreeGrammar bytes{{{"S",{std::string(1,'\0')}}},"S"};const auto escaped=earley_item_text(earley_trace(bytes,{std::string(1,'\0')}),{0,0});
 EXPECT_EQ(escaped.find('\0'),std::string::npos);
}
TEST(EarleyTrace, LongUnitChainUsesIterativeIndexedClosure){
 constexpr int depth=20000;ContextFreeGrammar g;g.start="n0";
 for(int i=0;i<depth;++i)g.productions.push_back({"n"+std::to_string(i),{"n"+std::to_string(i+1)}});
 g.productions.push_back({"n"+std::to_string(depth),{"a"}});
 const auto r=earley_recognize_work(g,{"a"});
 EXPECT_TRUE(r.accepted);
 EXPECT_EQ(r.chart_sizes,(std::vector<int>{depth+1,depth+1}));
 EXPECT_EQ(r.work.unique_items,2*(depth+1));
 EXPECT_EQ(r.work.completion_joins,depth);
}
TEST(EarleyTrace, ActualGrowthFamilyIdentitiesAndAmbiguousJoinCounts){
 const auto g=anbn();
 for(int k=1;k<=64;++k){std::vector<std::string> word(k,"a");word.insert(word.end(),k,"b");const auto r=earley_recognize_work(g,word);
 EXPECT_TRUE(r.accepted);
 EXPECT_EQ(r.work.unique_items,6*k+1);}
 ContextFreeGrammar ambiguous{{{"S",{"S","S"}},{"S",{"a"}}},"S"};
 for(int n=1;n<=32;++n){const auto t=earley_trace(ambiguous,std::vector<std::string>(n,"a"));
 EXPECT_TRUE(t.accepted);certificate(t);}
}
