#include <gtest/gtest.h>
#include <datamunge/algorithms/string_metrics.hpp>
#include <algorithm>
#include <deque>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
using namespace datamunge::algorithms;
namespace {
std::vector<std::string> words(std::size_t limit) {
  std::vector<std::string> out;std::size_t count=1;
  for(std::size_t n=0;n<=limit;++n,count*=3) for(std::size_t code=0;code<count;++code) {auto x=code;std::string s(n,'a');for(char& c:s) {c=static_cast<char>('a'+x%3);x/=3;}out.push_back(s);}return out;
}
// Exact shortest-path oracle in the edit graph, with no prefix DP recurrence.
// Every target length<=L has distance<=L. No <=L-edit path can grow beyond
// source.size()+L. Mapping outside-alphabet symbols to 'a' cannot increase cost,
// so the ternary graph also gives the unrestricted full-byte distance here.
std::unordered_map<std::string,std::size_t> graph_distances(const std::string& source,const std::vector<std::string>& targets,std::size_t limit) {
  std::unordered_set<std::string> missing(targets.begin(),targets.end());missing.erase(source);
  std::unordered_map<std::string,std::size_t> distance{{source,0}};std::deque<std::string> queue{source};
  while(!missing.empty() && !queue.empty()) {
    const auto s=queue.front();queue.pop_front();const auto depth=distance.at(s);if(depth==limit) continue;
    const auto add=[&](std::string next) {if(distance.emplace(next,depth+1).second) {missing.erase(next);queue.push_back(std::move(next));}};
    for(std::size_t i=0;i<s.size();++i) {
      auto next=s;next.erase(i,1);add(next);
      for(char c:{'a','b','c'}) if(c!=s[i]) {next=s;next[i]=c;add(next);}
      if(i+1<s.size() && s[i]!=s[i+1]) {next=s;std::swap(next[i],next[i+1]);add(next);}
    }
    if(s.size()<source.size()+limit) for(std::size_t i=0;i<=s.size();++i) for(char c:{'a','b','c'}) {auto next=s;next.insert(i,1,c);add(next);}
  }
  EXPECT_TRUE(missing.empty());return distance;
}
// Independent full-table OSA control for the production three-row implementation.
std::size_t osa_full(const std::string& a,const std::string& b) {
  std::vector<std::vector<std::size_t>> d(a.size()+1,std::vector<std::size_t>(b.size()+1));
  for(std::size_t i=0;i<=a.size();++i) d[i][0]=i;for(std::size_t j=0;j<=b.size();++j) d[0][j]=j;
  for(std::size_t i=1;i<=a.size();++i) for(std::size_t j=1;j<=b.size();++j) {
    d[i][j]=std::min({d[i-1][j]+1,d[i][j-1]+1,d[i-1][j-1]+(a[i-1]!=b[j-1])});
    if(i>1 && j>1 && a[i-1]==b[j-2] && a[i-2]==b[j-1]) d[i][j]=std::min(d[i][j],d[i-2][j-2]+1);
  }return d.back().back();
}
void verify(const std::string& a,const std::string& b) {
  const auto t=damerau_levenshtein_trace(a,b);
  EXPECT_EQ(t.source,a);
  EXPECT_EQ(t.target,b);
  EXPECT_EQ(t.distance,damerau_levenshtein_distance(a,b));
  const auto script=damerau_edit_script(t);
  EXPECT_EQ(script.size(),t.distance);
  EXPECT_EQ(apply_byte_edits(a,script),b);
  // Replay independently by operation kind, rather than the generic replace helper.
  auto current=a;
  for(const auto& e:script) {
    EXPECT_NE(e.kind,DamerauMove::match);
  ASSERT_LE(e.position,current.size());
  ASSERT_LE(e.before.size(),current.size()-e.position);
  EXPECT_EQ(current.substr(e.position,e.before.size()),e.before);
    switch(e.kind) {
      case DamerauMove::substitute:ASSERT_EQ(e.after.size(),1);current[e.position]=e.after[0];break;
      case DamerauMove::insert:current.insert(e.position,e.after);break;
      case DamerauMove::erase:current.erase(e.position,1);break;
      case DamerauMove::transpose:ASSERT_LT(e.position+1,current.size());std::swap(current[e.position],current[e.position+1]);break;
      default:FAIL()<<"unexpected native edit";
    }
  }EXPECT_EQ(current,b);
  std::size_t matches=0,eligible=0,selected=0;
  for(const auto& c:t.cells) {
    const auto i=c.source_prefix,j=c.target_prefix;std::size_t last_i=0,last_j=0;
    for(std::size_t k=1;k<i;++k) if(a[k-1]==b[j-1]) last_i=k;
    for(std::size_t k=1;k<j;++k) if(b[k-1]==a[i-1]) last_j=k;
    EXPECT_EQ(c.last_source,last_i);
  EXPECT_EQ(c.last_target,last_j);
  EXPECT_EQ(c.value,damerau_prefix_distance(t,i,j));
    EXPECT_EQ(c.diagonal,damerau_prefix_distance(t,i-1,j-1)+(a[i-1]!=b[j-1]));
    EXPECT_EQ(c.insertion,std::min(t.shape.bound,damerau_prefix_distance(t,i,j-1)+1));
    EXPECT_EQ(c.deletion,std::min(t.shape.bound,damerau_prefix_distance(t,i-1,j)+1));
    const auto bridge=last_i && last_j?std::min(t.shape.bound,damerau_prefix_distance(t,last_i-1,last_j-1)+(i-last_i-1)+1+(j-last_j-1)):t.shape.bound;
    EXPECT_EQ(c.transposition,bridge);
  EXPECT_EQ(c.value,std::min({c.diagonal,c.insertion,c.deletion,c.transposition}));
    matches+=a[i-1]==b[j-1];eligible+=last_i && last_j;selected+=c.choice==DamerauMove::transpose;
  }
  EXPECT_EQ(t.work.cells_evaluated,a.size()*b.size());
  EXPECT_EQ(t.work.byte_comparisons,t.work.cells_evaluated);
  EXPECT_EQ(t.work.byte_matches,matches);
  EXPECT_EQ(t.work.eligible_transpositions,eligible);
  EXPECT_EQ(t.work.selected_transpositions,selected);
  EXPECT_EQ(t.work.last_row_updates,a.size());
}
}
TEST(DamerauTrace, OriginalPairsAndExecutableUnrestrictedWitness) {
  const std::vector<std::pair<std::string,std::string>> pairs{{"ab","ba"},{"abcd","abdc"},{"receive","recieve"},{"data","adta"},{"CA","ABC"}};
  for(std::size_t i=0;i<pairs.size();++i) {const auto& [a,b]=pairs[i];
  EXPECT_EQ(levenshtein_distance(a,b),i<4?2:3);
  EXPECT_EQ(damerau_levenshtein_distance(a,b),i<4?1:2);
  EXPECT_EQ(optimal_string_alignment_distance(a,b),i<4?1:3);verify(a,b);}
  const auto t=damerau_levenshtein_trace("CA","ABC");const auto e=damerau_edit_script(t);
  ASSERT_EQ(e.size(),2);
  EXPECT_EQ(e[0].kind,DamerauMove::transpose);
  EXPECT_EQ(e[0].position,0);
  EXPECT_EQ(apply_byte_edits("CA",{e[0]}),"AC");
  EXPECT_EQ(e[1].kind,DamerauMove::insert);
  EXPECT_EQ(e[1].position,1);
  EXPECT_EQ(e[1].after,"B");
  const std::vector<std::vector<std::size_t>> expected{{0,1,2,3},{1,1,2,2},{2,1,2,2}};
  for(std::size_t i=0;i<expected.size();++i) for(std::size_t j=0;j<expected[i].size();++j) EXPECT_EQ(damerau_prefix_distance(t,i,j),expected[i][j]);
  EXPECT_EQ(t.cells.back().last_source,1);
  EXPECT_EQ(t.cells.back().last_target,1);
  EXPECT_EQ(t.cells.back().transposition,2);verify("ABC","CA");
}
TEST(DamerauTrace, AllTernaryWordsThroughFourMatchIndependentShortestEditGraph) {
  const auto all=words(4);
  ASSERT_EQ(all.size(),121);std::size_t pairs=0;
  for(const auto& a:all) {
    const auto oracle=graph_distances(a,all,4);
    for(const auto& b:all) {const auto distance=damerau_levenshtein_distance(a,b);
  EXPECT_EQ(distance,oracle.at(b))<<a<<" -> "<<b;
  EXPECT_EQ(optimal_string_alignment_distance(a,b),osa_full(a,b));
  EXPECT_LE(distance,optimal_string_alignment_distance(a,b));
  EXPECT_LE(optimal_string_alignment_distance(a,b),levenshtein_distance(a,b));verify(a,b);++pairs;}
  }EXPECT_EQ(pairs,14641);
}
TEST(DamerauTrace, ExhaustiveFiniteMetricTrianglesAndOsaCounterexample) {
  const auto all=words(4);std::vector<std::vector<std::size_t>> d(all.size(),std::vector<std::size_t>(all.size()));
  for(std::size_t i=0;i<all.size();++i) for(std::size_t j=0;j<all.size();++j) {d[i][j]=damerau_levenshtein_distance(all[i],all[j]);
  EXPECT_EQ(d[i][j]==0,i==j);
  EXPECT_GE(d[i][j],all[i].size()>all[j].size()?all[i].size()-all[j].size():all[j].size()-all[i].size());
  EXPECT_LE(d[i][j],std::max(all[i].size(),all[j].size()));}
  for(std::size_t i=0;i<all.size();++i) for(std::size_t j=0;j<all.size();++j) {EXPECT_EQ(d[i][j],d[j][i]);for(std::size_t k=0;k<all.size();++k) EXPECT_LE(d[i][k],d[i][j]+d[j][k]);}
  EXPECT_EQ(optimal_string_alignment_distance("CA","ABC"),3);
  EXPECT_EQ(optimal_string_alignment_distance("CA","AC"),1);
  EXPECT_EQ(optimal_string_alignment_distance("AC","ABC"),1);
}
TEST(DamerauTrace, SeededRepeatedAndFullBytePairsReplayActualRecords) {
  std::mt19937 rng(896);
  for(int trial=0;trial<1000;++trial) {std::string a(rng()%13,'\0'),b(rng()%13,'\0');for(char& c:a) c=static_cast<char>(rng()%(trial%2?256:4));for(char& c:b) c=static_cast<char>(rng()%(trial%2?256:4));verify(a,b);
  EXPECT_EQ(damerau_levenshtein_distance(a,b),damerau_levenshtein_distance(b,a));}
}
TEST(DamerauTrace, AllBytesNulAndUtf8RemainLiteralByteSequences) {
  for(unsigned c=0;c<256;++c) {
    std::string a;a.push_back(static_cast<char>(c));a.push_back(static_cast<char>((c+1)%256));auto b=a;std::swap(b[0],b[1]);
  EXPECT_EQ(damerau_levenshtein_distance(a,b),1);verify(a,b);
  }
  verify(std::string("\0A",2),std::string("A\0",2));verify(std::string(1,'\0'),"");verify("",std::string(1,'\0'));
  EXPECT_EQ(damerau_levenshtein_distance(std::string("\xc3\xa9",2),"e"),2);verify(std::string("\xc3\xa9",2),"e");
  EXPECT_EQ(damerau_levenshtein_distance("A","a"),1);
}
TEST(DamerauTrace, GeometryAndCompleteCellScriptCaps) {
  EXPECT_EQ(damerau_shape(0,0).cells,4);
  EXPECT_EQ(damerau_shape(2,3).cells,20);
  EXPECT_THROW(damerau_shape(std::numeric_limits<std::size_t>::max(),0),std::length_error);
  EXPECT_THROW(damerau_shape(std::numeric_limits<std::size_t>::max()-2,0),std::length_error);
  EXPECT_THROW(damerau_shape(0,std::numeric_limits<std::size_t>::max()),std::length_error);
  EXPECT_THROW(damerau_levenshtein_trace("CA","ABC",19),std::length_error);
  EXPECT_NO_THROW(damerau_levenshtein_trace("CA","ABC",20));
  EXPECT_NO_THROW(damerau_levenshtein_trace("CA","ABC",0));
  EXPECT_THROW(damerau_levenshtein_trace("","",3),std::length_error);
  EXPECT_THROW(damerau_levenshtein_trace("",std::string(100,'a'),203),std::length_error);
  const auto t=damerau_levenshtein_trace("CA","ABC");
  EXPECT_THROW(damerau_edit_script(t,1),std::length_error);
  EXPECT_EQ(damerau_edit_script(t,2).size(),2);
  EXPECT_EQ(damerau_edit_script(t,0).size(),2);
  EXPECT_TRUE(damerau_edit_script(damerau_levenshtein_trace("same","same"),1).empty());
}
TEST(DamerauTrace, OwnedInputsAndMalformedVisitedRecordsRejectSafely) {
  std::string a="CA",b="ABC";const auto t=damerau_levenshtein_trace(a,b);a="x";b="y";
  EXPECT_EQ(apply_byte_edits(t.source,damerau_edit_script(t)),"ABC");
  EXPECT_THROW(damerau_prefix_distance(t,3,0),std::out_of_range);
  EXPECT_THROW(damerau_prefix_distance(t,0,4),std::out_of_range);
  auto bad=t;bad.table.pop_back();
  EXPECT_THROW(damerau_prefix_distance(bad,0,0),std::invalid_argument);
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.cells.pop_back();
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.cells.back().last_source=0;
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.cells.back().last_target=3;
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.cells.back().choice=static_cast<DamerauMove>(999);
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.table.back()=1;bad.distance=1;bad.cells.back().value=1;
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.cells.back().source_prefix=1;
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
  bad=t;bad.table[bad.shape.columns+1]=1;
  EXPECT_THROW(damerau_edit_script(bad),std::invalid_argument);
}
TEST(DamerauTrace, LiteralByteInstructionValidationAndCurrentStatePositions) {
  const std::vector<ByteEdit> edits{{DamerauMove::transpose,0,"AB","BA"},{DamerauMove::insert,3,"","X"},{DamerauMove::erase,1,"A",""},{DamerauMove::substitute,0,"B","Y"},{DamerauMove::match,1,"C","C"}};
  EXPECT_EQ(apply_byte_edits("ABC",edits),"YCX");
  EXPECT_EQ(apply_byte_edits("ABC",{}),"ABC");
  const std::string input="ab";
  EXPECT_THROW(apply_byte_edits(input,{{DamerauMove::erase,0,"x",""}}),std::invalid_argument);
  EXPECT_THROW(apply_byte_edits(input,{{DamerauMove::transpose,0,"ab","ab"}}),std::invalid_argument);
  EXPECT_THROW(apply_byte_edits(input,{{DamerauMove::substitute,0,"a","a"}}),std::invalid_argument);
  EXPECT_THROW(apply_byte_edits(input,{{static_cast<DamerauMove>(999),0,"",""}}),std::invalid_argument);
  EXPECT_THROW(apply_byte_edits(input,{{DamerauMove::insert,3,"","x"}}),std::out_of_range);
  EXPECT_THROW(apply_byte_edits(input,{{DamerauMove::erase,2,"b",""}}),std::out_of_range);
  EXPECT_EQ(input,"ab");
}
TEST(DamerauTrace, RectangularActualWorkAndEmptyBorders) {
  for(std::size_t n:{8,16,32,64,128}) for(std::size_t m:{n/2,n*2}) {
    const auto t=damerau_levenshtein_trace(std::string(n,'a'),std::string(m,'b'));
  EXPECT_EQ(t.distance,std::max(n,m));
  EXPECT_EQ(t.work.cells_evaluated,n*m);
  EXPECT_EQ(t.work.byte_matches,0);
  EXPECT_EQ(t.work.eligible_transpositions,0);
  EXPECT_EQ(t.work.selected_transpositions,0);
  EXPECT_EQ(t.table.size(),(n+2)*(m+2));
  }
  verify("","");verify("abc","");verify("","abc");verify("abc","abc");
}
