#include <datamunge/algorithms/string_metrics.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
namespace {
using Pair=std::pair<unsigned char,unsigned char>;
std::vector<Pair> pairs(const std::string& s) {
  std::vector<Pair> out;
  for(std::size_t i=1;i<s.size();++i)out.emplace_back(static_cast<unsigned char>(s[i-1]),static_cast<unsigned char>(s[i]));
  return out;
}
void check(const std::string& a,const std::string& b) {
  const auto aa=pairs(a),bb=pairs(b);auto sorted_a=aa,sorted_b=bb;std::sort(sorted_a.begin(),sorted_a.end());std::sort(sorted_b.begin(),sorted_b.end());
  std::vector<Pair> intersection,united;
  std::set_intersection(sorted_a.begin(),sorted_a.end(),sorted_b.begin(),sorted_b.end(),std::back_inserter(intersection));
  std::set_union(sorted_a.begin(),sorted_a.end(),sorted_b.begin(),sorted_b.end(),std::back_inserter(united));
  const auto t=dice_coefficient_trace(a,b,0);const std::size_t total=aa.size()+bb.size();
  const double want=total?2.0*intersection.size()/total:double(a==b);
  const double j=united.empty()?double(a==b):double(intersection.size())/united.size();
  EXPECT_DOUBLE_EQ(dice_coefficient(a,b),want);EXPECT_DOUBLE_EQ(t.overlap.coefficient,want);EXPECT_DOUBLE_EQ(t.overlap.jaccard,j);
  EXPECT_EQ(t.left,a);EXPECT_EQ(t.right,b);EXPECT_EQ(t.overlap.left_bigrams,aa.size());EXPECT_EQ(t.overlap.right_bigrams,bb.size());
  EXPECT_EQ(t.overlap.shared,intersection.size());EXPECT_EQ(t.overlap.total,total);EXPECT_EQ(t.overlap.union_count,united.size());
  EXPECT_NEAR(want,2*j/(1+j),1e-15);EXPECT_GE(want,0);EXPECT_LE(want,1);
  EXPECT_EQ(t.work.left_updates,aa.size());EXPECT_EQ(t.work.right_lookups,bb.size());
  EXPECT_EQ(t.work.matches,intersection.size());EXPECT_EQ(t.work.misses,bb.size()-intersection.size());
  std::map<Pair,std::pair<std::size_t,std::size_t>> hist;
  for(const auto& p:aa)++hist[p].first;
  for(const auto& p:bb)++hist[p].second;
  ASSERT_EQ(hist.size(),t.counts.size());std::size_t k=0;
  for(const auto& [p,count]:hist) {
    const auto& row=t.counts[k++];EXPECT_EQ(row.key,256*unsigned(p.first)+unsigned(p.second));
    EXPECT_EQ(row.left_count,count.first);EXPECT_EQ(row.right_count,count.second);EXPECT_EQ(row.shared_count,std::min(count.first,count.second));
  }
  // Independent occurrence matching: scan a list of unconsumed source tokens.
  std::vector<bool> used(aa.size(),false);ASSERT_EQ(t.events.size(),bb.size());
  for(std::size_t j0=0;j0<bb.size();++j0) {
    std::size_t remaining=0,first=aa.size();
    for(std::size_t i=0;i<aa.size();++i)if(!used[i] && aa[i]==bb[j0]){++remaining;if(first==aa.size())first=i;}
    const auto& e=t.events[j0];EXPECT_EQ(e.right_offset,j0);EXPECT_EQ(e.key,256*unsigned(bb[j0].first)+unsigned(bb[j0].second));
    EXPECT_EQ(e.remaining_before,remaining);EXPECT_EQ(e.remaining_after,remaining?remaining-1:0);EXPECT_EQ(e.matched,remaining!=0);
    if(first<aa.size())used[first]=true;
  }
}
std::vector<std::string> words(const std::array<unsigned char,3>& alphabet) {
  std::vector<std::string> out{std::string{}};std::size_t begin=0,end=1;
  for(int length=1;length<=4;++length) {
    for(std::size_t i=begin;i<end;++i)for(auto c:alphabet)out.push_back(out[i]+char(c));
    begin=end;end=out.size();
  }
  return out;
}
}
TEST(DiceTrace, OriginalFiveAndShortConventions) {
  const std::vector<std::pair<std::string,std::string>> inputs{{"night","nacht"},{"france","french"},{"healed","sealed"},{"night","night"},{"abc","xyz"}};
  const std::vector<double> values{.25,.4,.8,1,0};
  for(std::size_t i=0;i<inputs.size();++i){check(inputs[i].first,inputs[i].second);EXPECT_DOUBLE_EQ(dice_coefficient(inputs[i].first,inputs[i].second),values[i]);}
  for(const std::string a:{"","a","b","aa","ab"})for(const std::string b:{"","a","b","aa","ab"})check(a,b);
  EXPECT_DOUBLE_EQ(dice_coefficient("a","b"),0);EXPECT_DOUBLE_EQ(dice_coefficient("","a"),0);
}
TEST(DiceTrace, ExhaustiveSmallAsciiAndBinaryStrings) {
  std::size_t comparisons=0;
  for(auto alphabet:{std::array<unsigned char,3>{'a','b','c'},std::array<unsigned char,3>{0,128,255}}) {
    const auto all=words(alphabet);ASSERT_EQ(all.size(),121);
    for(const auto& a:all)for(const auto& b:all){check(a,b);++comparisons;}
  }
  EXPECT_EQ(comparisons,29282);
}
TEST(DiceTrace, SeededBinaryOracleAndByteRenamingInvariance) {
  std::mt19937_64 rng(20261107);
  for(int trial=0;trial<2000;++trial) {
    std::string a(rng()%151,'\0'),b(rng()%151,'\0');
    for(char& c:a)c=char(rng()&255);
    for(char& c:b)c=char(rng()&255);
    if(trial%4==0)b=a.substr(0,a.size()/2)+b;
    check(a,b);EXPECT_DOUBLE_EQ(dice_coefficient(a,b),dice_coefficient(b,a));
    std::string renamed_a=a,renamed_b=b;
    for(char& c:renamed_a)c=char(static_cast<unsigned char>(c)^255);
    for(char& c:renamed_b)c=char(static_cast<unsigned char>(c)^255);
    EXPECT_DOUBLE_EQ(dice_coefficient(a,b),dice_coefficient(renamed_a,renamed_b));
  }
}
TEST(DiceTrace, MultiplicityCollisionsAndNonmetricCounterexample) {
  const auto t=dice_coefficient_trace("aa","aaaa");ASSERT_EQ(t.counts.size(),1);ASSERT_EQ(t.events.size(),3);
  EXPECT_EQ(t.counts[0].left_count,1);EXPECT_EQ(t.counts[0].right_count,3);EXPECT_EQ(t.overlap.shared,1);
  EXPECT_DOUBLE_EQ(t.overlap.coefficient,.5);EXPECT_DOUBLE_EQ(t.overlap.jaccard,1.0/3);
  EXPECT_TRUE(t.events[0].matched);EXPECT_FALSE(t.events[1].matched);EXPECT_FALSE(t.events[2].matched);
  check("abaca","acaba");EXPECT_DOUBLE_EQ(dice_coefficient("abaca","acaba"),1);
  const double direct=1-dice_coefficient("ab","bc"),via=(1-dice_coefficient("ab","abc"))+(1-dice_coefficient("abc","bc"));
  EXPECT_GT(direct,via);EXPECT_NEAR(via,2.0/3,1e-15);EXPECT_DOUBLE_EQ(direct,1);
  EXPECT_DOUBLE_EQ(dice_coefficient("night","Night"),.75);EXPECT_DOUBLE_EQ(dice_coefficient("night","night "),8.0/9);
}
TEST(DiceTrace, All65536BytePairsAndUnsignedSortedRows) {
  std::string text;text.reserve(131072);
  std::array<std::size_t,65536> expected{};
  for(unsigned a=0;a<256;++a)for(unsigned b=0;b<256;++b){text.push_back(char(a));text.push_back(char(b));}
  for(std::size_t i=1;i<text.size();++i)++expected[256*static_cast<unsigned char>(text[i-1])+static_cast<unsigned char>(text[i])];
  const auto t=dice_coefficient_trace(text,text);ASSERT_EQ(t.counts.size(),65536);EXPECT_EQ(t.events.size(),131071);
  EXPECT_EQ(t.overlap.shared,131071);EXPECT_DOUBLE_EQ(t.overlap.coefficient,1);EXPECT_DOUBLE_EQ(t.overlap.jaccard,1);
  for(std::size_t key=0;key<65536;++key) {
    EXPECT_GT(expected[key],0);const auto& row=t.counts[key];EXPECT_EQ(row.key,key);
    EXPECT_EQ(row.left_count,expected[key]);EXPECT_EQ(row.right_count,expected[key]);EXPECT_EQ(row.shared_count,expected[key]);
  }
  EXPECT_EQ(t.work.misses,0);EXPECT_EQ(t.work.matches,131071);
}
TEST(DiceTrace, LiteralNulsUtf8AndNoImplicitNormalization) {
  check(std::string("a\0a",3),std::string("a\0b",3));
  EXPECT_DOUBLE_EQ(dice_coefficient(std::string("a\0a",3),std::string("a\0b",3)),.5);
  const std::string composed="\xc3\xa9",decomposed="e\xcc\x81";
  check(composed,decomposed);EXPECT_DOUBLE_EQ(dice_coefficient(composed,decomposed),0);
  check("a b","ab");check("a-b","ab");check(std::string("\xff\x80\0",3),std::string("\xff\x80",2));
  EXPECT_DOUBLE_EQ(dice_coefficient("a b","ab"),0);
}
TEST(DiceTrace, CountOnlyArithmeticAndOverflowPrecedence) {
  const auto big=dice_overlap_counts(3000000000ULL,1000000000ULL,1000000000ULL);
  EXPECT_EQ(big.total,4000000000ULL);EXPECT_EQ(big.union_count,3000000000ULL);
  EXPECT_DOUBLE_EQ(big.coefficient,.5);EXPECT_DOUBLE_EQ(big.jaccard,1.0/3);
  const auto max=std::numeric_limits<std::size_t>::max();const auto zero=dice_overlap_counts(max,0,0);
  EXPECT_EQ(zero.total,max);EXPECT_DOUBLE_EQ(zero.coefficient,0);EXPECT_DOUBLE_EQ(zero.jaccard,0);
  const auto equal=dice_overlap_counts(max/2,max/2,max/2);EXPECT_EQ(equal.total,max-1);
  EXPECT_DOUBLE_EQ(equal.coefficient,1);EXPECT_DOUBLE_EQ(equal.jaccard,1);
  EXPECT_THROW(dice_overlap_counts(max,1,0),std::length_error);
  EXPECT_THROW(dice_overlap_counts(max,max,max),std::length_error);
  EXPECT_THROW(dice_overlap_counts(max,1,2),std::invalid_argument);
  EXPECT_THROW(dice_overlap_counts(0,0,1),std::invalid_argument);
  EXPECT_DOUBLE_EQ(dice_overlap_counts(0,0,0).coefficient,1);
  EXPECT_DOUBLE_EQ(dice_overlap_counts(0,0,0,false).coefficient,0);
  EXPECT_DOUBLE_EQ(dice_overlap_counts(0,0,0,false).jaccard,0);
}
TEST(DiceTrace, CompleteCapsEarlyCasesAndOwnership) {
  EXPECT_NO_THROW(dice_coefficient_trace("night","nacht",8));
  EXPECT_THROW(dice_coefficient_trace("night","nacht",7),std::length_error);
  EXPECT_NO_THROW(dice_coefficient_trace("night","nacht",0));
  EXPECT_NO_THROW(dice_coefficient_trace("aa","aaaa",4));
  EXPECT_THROW(dice_coefficient_trace("aa","aaaa",3),std::length_error);
  EXPECT_THROW(dice_coefficient_trace("night","night",7),std::length_error);
  EXPECT_NO_THROW(dice_coefficient_trace("a","b",1));
  const std::string large(1000002,'a');
  EXPECT_THROW(dice_coefficient_trace(large,"x"),std::length_error);
  EXPECT_DOUBLE_EQ(dice_coefficient(large,"x"),0);
  std::string a="night",b="nacht";auto t=dice_coefficient_trace(a,b);a[0]='x';b.clear();
  EXPECT_EQ(t.left,"night");EXPECT_EQ(t.right,"nacht");auto copy=t;t.events[3].matched=false;t.counts.clear();t.left.clear();
  EXPECT_TRUE(copy.events[3].matched);EXPECT_EQ(copy.counts.size(),7);EXPECT_EQ(copy.left,"night");
  const auto equal=dice_coefficient_trace("night","night");EXPECT_EQ(equal.work.left_updates,4);EXPECT_EQ(equal.work.matches,4);
}
TEST(DiceTrace, RepeatedLargeCountsAndActualLinearOccurrenceWork) {
  for(std::size_t k:{2,4,8,64,256,1024,4096}) {
    const auto t=dice_coefficient_trace(std::string(k,'a'),std::string(3*k,'a'));
    EXPECT_EQ(t.counts.size(),1);EXPECT_EQ(t.overlap.shared,k-1);EXPECT_EQ(t.work.left_updates+t.work.right_lookups,4*k-2);
    EXPECT_EQ(t.events.size(),3*k-1);EXPECT_EQ(t.work.matches,k-1);EXPECT_EQ(t.work.misses,2*k);
    EXPECT_DOUBLE_EQ(t.overlap.coefficient,2.0*(k-1)/(4*k-2));
  }
  const auto t=dice_coefficient_trace(std::string(100001,'a'),std::string(300001,'a'));
  ASSERT_EQ(t.counts.size(),1);EXPECT_EQ(t.counts[0].left_count,100000);EXPECT_EQ(t.counts[0].right_count,300000);
  EXPECT_EQ(t.work.matches,100000);EXPECT_EQ(t.work.misses,200000);EXPECT_EQ(t.events.size(),300000);
  EXPECT_DOUBLE_EQ(t.overlap.coefficient,.5);EXPECT_EQ(t.overlap.union_count,300000);
}
