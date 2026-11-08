#include <datamunge/algorithms/blast.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <limits>
#include <map>
#include <random>
#include <tuple>
using namespace datamunge::algorithms;
namespace {
using Key=std::tuple<int,int,int,int>;
Key key(const HSP& h){return {h.query_start,h.db_start,h.length,h.score};}
HSP brute_anchored(const std::string& q,const std::string& d,BlastSeed seed,int w){
    HSP best{seed.query_start,seed.db_start,w,2*w};
    const int left=std::min(seed.query_start,seed.db_start);
    const int right=std::min(int(q.size())-seed.query_start-w,int(d.size())-seed.db_start-w);
    for(int l=0;l<=left;++l)for(int r=0;r<=right;++r){
        const int startq=seed.query_start-l,startd=seed.db_start-l,len=w+l+r;
        int score=0;for(int i=0;i<len;++i)score+=q[startq+i]==d[startd+i]?2:-1;
        if(score>best.score || (score==best.score && len<best.length))best={startq,startd,len,score};
    }
    return best;
}
std::vector<std::string> binary_strings(){
    std::vector<std::string> values{""};
    for(int n=1;n<=4;++n)for(int bits=0;bits<(1<<n);++bits){
        std::string s(n,'A');for(int i=0;i<n;++i)if((bits>>i)&1)s[i]='C';values.push_back(s);
    }
    return values;
}
}
TEST(BlastTrace, SeedsMatchNaiveAllByteWordPairs){
    std::mt19937 rng(94);
    for(int sample=0;sample<1000;++sample){
        std::string q(rng()%12,'\0'),d(rng()%15,'\0');
        for(auto& b:q)b=char(rng()%4);for(auto& b:d)b=char(rng()%4);
        const int w=1+rng()%5;std::vector<std::pair<int,int>> expected;
        for(int i=0;i+w<=int(q.size());++i)for(int j=0;j+w<=int(d.size());++j)
            if(q.substr(i,w)==d.substr(j,w))expected.emplace_back(i,j);
        std::vector<std::pair<int,int>> actual;
        for(const auto& s:blast_seeds(q,d,w))actual.emplace_back(s.query_start,s.db_start);
        EXPECT_EQ(actual,expected);
    }
    EXPECT_TRUE(blast_seeds("a","A",1).empty());
    EXPECT_EQ(blast_seeds("AAA","AAAA",2).size(),6);
}
TEST(BlastTrace, EverySmallBinarySeedMatchesExhaustiveAnchoredSegments){
    const auto strings=binary_strings();
    for(const auto& q:strings)for(const auto& d:strings)for(int w=1;w<=3;++w){
        std::map<int,HSP> expected;
        for(const auto& seed:blast_seeds(q,d,w)){
            const auto reference=brute_anchored(q,d,seed,w);
            const auto trace=blast_extend_trace(q,d,seed,w,2,-1,1000);
            EXPECT_EQ(key(trace.hsp),key(reference))<<q<<' '<<d;
            const int diagonal=seed.db_start-seed.query_start;
            auto found=expected.find(diagonal);
            if(found==expected.end() || reference.score>found->second.score)expected[diagonal]=reference;
        }
        const auto hits=blast_search(q,d,w,2,-1,1000,0);
        ASSERT_EQ(hits.size(),expected.size());
        for(const auto& h:hits){const auto found=expected.find(h.db_start-h.query_start);
            ASSERT_NE(found,expected.end());EXPECT_EQ(key(h),key(found->second));}
    }
}
TEST(BlastTrace, StrictDropAllowsRecoveryAtEqualityInEitherDirection){
    const auto stopped=blast_extend_trace("AAACCAA","AAATTAA",{0,0},3,2,-1,1);
    EXPECT_EQ(stopped.hsp.score,6);EXPECT_EQ(stopped.hsp.length,3);
    ASSERT_EQ(stopped.steps.size(),2);EXPECT_EQ(stopped.steps.back().running_score,-2);
    EXPECT_TRUE(stopped.steps.back().stopped);
    const auto recovered=blast_extend_trace("AAACCAA","AAATTAA",{0,0},3,2,-1,2);
    EXPECT_EQ(recovered.hsp.score,8);EXPECT_EQ(recovered.hsp.length,7);
    ASSERT_EQ(recovered.steps.size(),4);EXPECT_FALSE(recovered.steps[1].stopped);
    EXPECT_EQ(recovered.steps.back().best_score,2);
    const auto left=blast_extend_trace("AACCAAA","AATTAAA",{4,4},3,2,-1,2);
    EXPECT_EQ(left.hsp.score,8);EXPECT_EQ(left.hsp.query_start,0);
    ASSERT_EQ(left.steps.size(),4);EXPECT_TRUE(left.steps.front().left);
    EXPECT_EQ(left.steps.front().query_position,3);EXPECT_EQ(left.steps.back().query_position,0);
}
TEST(BlastTrace, TraceStatesAndReturnedIntervalsUseRealScores){
    const std::string q="TTGGATCCAATT",d="ACGTACGTGGATCCAATTGGCCAAGGTT";
    for(const auto& seed:blast_seeds(q,d,4)){
        const auto trace=blast_extend_trace(q,d,seed,4);
        std::int64_t running[2]{},best[2]{};
        for(const auto& step:trace.steps){
            const int side=step.left?1:0;
            EXPECT_EQ(step.delta,q[step.query_position]==d[step.db_position]?2:-1);
            running[side]+=step.delta;best[side]=std::max(best[side],running[side]);
            EXPECT_EQ(step.running_score,running[side]);EXPECT_EQ(step.best_score,best[side]);
            EXPECT_EQ(step.stopped,best[side]-running[side]>5);
        }
        const auto& h=trace.hsp;int sum=0;
        for(int i=0;i<h.length;++i)sum+=q[h.query_start+i]==d[h.db_start+i]?2:-1;
        EXPECT_EQ(h.score,sum);EXPECT_LE(h.query_start,seed.query_start);
        EXPECT_GE(h.query_start+h.length,seed.query_start+4);
    }
    const auto hits=blast_search(q,d,4);ASSERT_EQ(hits.size(),3);
    EXPECT_EQ(key(hits[0]),(Key{1,7,11,22}));
    EXPECT_EQ(key(hits[1]),(Key{0,16,4,8}));EXPECT_EQ(key(hits[2]),(Key{6,20,4,8}));
}
TEST(BlastTrace, DiagonalReductionTieOrderAndInclusiveThresholdAreExplicit){
    const auto equal=blast_search("AAACCCAAA","AAATTTAAA",3,2,-1,0,6);
    ASSERT_EQ(equal.size(),3);EXPECT_EQ(key(equal[0]),(Key{0,0,3,6}));
    EXPECT_EQ(key(equal[1]),(Key{0,6,3,6}));EXPECT_EQ(key(equal[2]),(Key{6,0,3,6}));
    const auto better=blast_search("AAACCCAAAA","AAATTTAAAA",3,2,-1,0,6);
    ASSERT_FALSE(better.empty());EXPECT_EQ(key(better.front()),(Key{6,6,4,8}));
    EXPECT_EQ(blast_search("AAA","AAA",3,2,-1,0,6).size(),1);
    EXPECT_TRUE(blast_search("AAA","AAA",3,2,-1,0,7).empty());
    EXPECT_TRUE(blast_search("ACACAC","AGAGAG",2,2,-1,5,1).empty());
}
TEST(BlastTrace, InvalidParametersSeedsAndArithmeticBoundariesAreChecked){
    for(int w:{std::numeric_limits<int>::min(),-1,0}){
        EXPECT_THROW(blast_seeds("","",w),std::invalid_argument);
        EXPECT_THROW(blast_search("","",w),std::invalid_argument);
    }
    EXPECT_THROW(blast_search("A","A",1,0),std::invalid_argument);
    EXPECT_THROW(blast_search("A","A",1,2,0),std::invalid_argument);
    EXPECT_THROW(blast_search("A","A",1,2,-1,-1),std::invalid_argument);
    for(const auto& seed:std::vector<BlastSeed>{{-1,0},{0,-1},{1,0},{0,1}})
        EXPECT_THROW(blast_extend_trace("A","A",seed,1),std::invalid_argument);
    EXPECT_THROW(blast_extend_trace("A","T",{0,0},1),std::invalid_argument);
    EXPECT_THROW(blast_extend_trace("A","A",{0,0},2),std::invalid_argument);
    EXPECT_TRUE(blast_search("","AAA",1).empty());
    EXPECT_TRUE(blast_search("A","A",std::numeric_limits<int>::max()).empty());
    const int maximum=std::numeric_limits<int>::max();
    EXPECT_EQ(blast_search("A","A",1,maximum).front().score,maximum);
    EXPECT_THROW(blast_search("AA","AA",1,maximum),std::overflow_error);
    EXPECT_THROW(blast_extend_trace("AA","AA",{0,0},1,maximum),std::overflow_error);
    const auto extreme=blast_extend_trace("AC","AT",{0,0},1,maximum,
                                          std::numeric_limits<int>::min(),maximum);
    EXPECT_EQ(extreme.hsp.score,maximum);ASSERT_EQ(extreme.steps.size(),1);
    EXPECT_TRUE(extreme.steps.front().stopped);
}
