#include <datamunge/algorithms/eclat.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <set>
using namespace datamunge::algorithms;
namespace {
using Rows=std::vector<std::vector<std::size_t>>;
using Supports=std::map<std::vector<std::size_t>,std::size_t>;
Supports supports(const std::vector<FrequentItemset>& sets){Supports out;for(const auto& s:sets){EXPECT_TRUE(out.emplace(s.items,s.support).second);}return out;}
std::vector<FrequentItemset> brute(const Rows& tx,std::size_t minimum){
    std::set<std::size_t> universe;std::vector<std::set<std::size_t>> rows;
    for(const auto& row:tx){rows.emplace_back(row.begin(),row.end());universe.insert(row.begin(),row.end());}
    std::vector<std::size_t> ids(universe.begin(),universe.end());std::vector<FrequentItemset> out;
    for(std::size_t mask=1;mask<(std::size_t{1}<<ids.size());++mask){std::vector<std::size_t> items;
        for(std::size_t i=0;i<ids.size();++i)if(mask&(std::size_t{1}<<i))items.push_back(ids[i]);
        std::size_t support=0;for(const auto& row:rows){bool all=true;for(auto item:items)all&=row.contains(item);support+=all;}
        if(support>=minimum)out.push_back({items,support});}
    std::sort(out.begin(),out.end(),[](const auto& a,const auto& b){return a.items.size()!=b.items.size()?a.items.size()<b.items.size():a.items<b.items;});return out;
}
void certificate(const EclatTrace& t){
    const auto& tx=t.transactions;std::set<std::size_t> universe;std::size_t raw=0,incidence=0;
    for(const auto& row:tx){raw+=row.size();std::set<std::size_t> unique(row.begin(),row.end());incidence+=unique.size();universe.insert(unique.begin(),unique.end());}
    ASSERT_EQ(t.singleton_count,universe.size());ASSERT_GE(t.states.size(),t.singleton_count);
    const auto none=std::numeric_limits<std::size_t>::max();std::size_t count=eclat_trace_base_entries(tx),steps=0,matches=0,frequent_roots=0,maxdepth=0;
    std::vector<std::size_t> roots(universe.begin(),universe.end());std::set<std::vector<std::size_t>> distinct;
    for(std::size_t i=0;i<t.states.size();++i){const auto& s=t.states[i];const auto& items=s.itemset.items;
        ASSERT_FALSE(items.empty());EXPECT_TRUE(distinct.insert(items).second);EXPECT_TRUE(std::is_sorted(items.begin(),items.end()));EXPECT_EQ(std::set<std::size_t>(items.begin(),items.end()).size(),items.size());
        std::vector<std::size_t> tids;
        for(std::size_t row=0;row<tx.size();++row){bool all=true;for(auto item:items)all&=std::find(tx[row].begin(),tx[row].end(),item)!=tx[row].end();if(all)tids.push_back(row);}
        EXPECT_EQ(s.tids,tids);EXPECT_EQ(s.itemset.support,tids.size());EXPECT_EQ(s.frequent,tids.size()>=t.min_support_count);
        count+=1+items.size()+tids.size();
        if(i<t.singleton_count){EXPECT_EQ(items,(std::vector<std::size_t>{roots[i]}));EXPECT_EQ(s.left,none);EXPECT_EQ(s.right,none);EXPECT_EQ(s.steps,0u);frequent_roots+=s.frequent;}
        else{ASSERT_LT(s.left,i);ASSERT_LT(s.right,i);const auto& l=t.states[s.left];const auto& r=t.states[s.right];ASSERT_TRUE(l.frequent);ASSERT_TRUE(r.frequent);ASSERT_EQ(l.itemset.items.size(),r.itemset.items.size());
            EXPECT_TRUE(std::equal(l.itemset.items.begin(),l.itemset.items.end()-1,r.itemset.items.begin()));EXPECT_LT(l.itemset.items.back(),r.itemset.items.back());auto expected=l.itemset.items;expected.push_back(r.itemset.items.back());EXPECT_EQ(items,expected);
            // Independent merge-step identity: distinct union entries through the earlier last tid.
            std::size_t expected_steps=0;if(!l.tids.empty()&&!r.tids.empty()){std::set<std::size_t> joined(l.tids.begin(),l.tids.end());joined.insert(r.tids.begin(),r.tids.end());const auto end=std::min(l.tids.back(),r.tids.back());expected_steps=std::distance(joined.begin(),joined.upper_bound(end));}
            EXPECT_EQ(s.steps,expected_steps);steps+=s.steps;matches+=tids.size();}
    }
    Supports emitted;std::set<std::size_t> visits;
    for(auto id:t.emitted){ASSERT_LT(id,t.states.size());EXPECT_TRUE(visits.insert(id).second);const auto& s=t.states[id];EXPECT_TRUE(s.frequent);EXPECT_TRUE(emitted.emplace(s.itemset.items,s.itemset.support).second);maxdepth=std::max(maxdepth,s.itemset.items.size());}
    for(std::size_t i=0;i<t.states.size();++i)EXPECT_EQ(visits.contains(i),t.states[i].frequent);
    EXPECT_EQ(emitted,supports(t.itemsets));EXPECT_EQ(emitted,supports(brute(tx,t.min_support_count)));
    EXPECT_EQ(t.work.transactions,tx.size());EXPECT_EQ(t.work.raw_items,raw);EXPECT_EQ(t.work.incidences,incidence);EXPECT_EQ(t.work.frequent_singletons,frequent_roots);
    EXPECT_EQ(t.work.intersections,t.states.size()-t.singleton_count);EXPECT_EQ(t.work.intersection_steps,steps);EXPECT_EQ(t.work.matched_tids,matches);EXPECT_EQ(t.work.emitted_itemsets,t.itemsets.size());EXPECT_EQ(t.work.peak_frames,maxdepth);EXPECT_EQ(t.captured_entries,count+t.emitted.size());
}
}
TEST(EclatTrace, ExhaustiveSmallDatabasesAgainstIndependentSubsets){
    std::size_t cases=0;
    for(int n=0;n<=3;++n){int total=1;for(int i=0;i<n;++i)total*=8;
        for(int code=0;code<total;++code){int v=code;Rows tx(n);for(auto& row:tx){int mask=v%8;v/=8;for(int i=0;i<3;++i)if((mask>>i)&1)row.push_back(i);}
            for(int minimum=0;minimum<=n+1;++minimum){const auto expected=brute(tx,minimum);const auto t=eclat_trace(tx,minimum);certificate(t);const auto actual=eclat_frequent_itemsets(tx,minimum);
                ASSERT_EQ(actual.size(),expected.size());for(std::size_t i=0;i<actual.size();++i){EXPECT_EQ(actual[i].items,expected[i].items);EXPECT_EQ(actual[i].support,expected[i].support);}++cases;}}}
    EXPECT_EQ(cases,2842u);
}
TEST(EclatTrace, SeededSparseDuplicateAndPermutedRows){
    std::mt19937 rng(20261107);const auto big=std::numeric_limits<std::size_t>::max();
    const std::array<std::size_t,6> ids{0,2,7,99,big-1,big};
    for(int trial=0;trial<500;++trial){Rows tx(rng()%10);for(auto& row:tx){for(auto id:ids)if(rng()%2){row.push_back(id);if(rng()%3==0)row.push_back(id);}std::shuffle(row.begin(),row.end(),rng);}
        const auto minimum=trial%7==0?big:std::size_t(rng()%(tx.size()+2));const auto t=eclat_trace(tx,minimum);certificate(t);const auto expected=supports(brute(tx,minimum));EXPECT_EQ(supports(eclat_frequent_itemsets(tx,minimum)),expected);
        std::shuffle(tx.begin(),tx.end(),rng);const auto permuted=eclat_trace(tx,minimum);certificate(permuted);EXPECT_EQ(supports(permuted.itemsets),expected);
        if(minimum>0)EXPECT_EQ(supports(fp_growth_frequent_itemsets(tx,minimum)),expected);
        EXPECT_EQ(supports(apriori_frequent_itemsets(tx,minimum)),expected);
    }
}
TEST(EclatTrace, ActualConditionalOperandsAndCreationVersusVisitOrder){
    const Rows tx{{0,1,2},{0,1},{1,2},{0,1,2},{3,4},{0,3,4},{3,4},{0,1,2,3}};
    const auto t=eclat_trace(tx,2);certificate(t);
    EXPECT_EQ(t.states.size(),18u);EXPECT_EQ(t.work.intersections,13u);EXPECT_EQ(t.work.matched_tids,24u);EXPECT_EQ(t.captured_entries,137u);
    ASSERT_EQ(t.states[9].itemset.items,(std::vector<std::size_t>{0,1,2}));EXPECT_EQ(t.states[9].left,5u);EXPECT_EQ(t.states[9].right,6u);
    EXPECT_EQ(t.emitted,(std::vector<std::size_t>{0,5,9,6,7,1,12,2,3,17,4}));
}
TEST(EclatTrace, ExactQuotaRawBaseAndIndependentOwnership){
    Rows tx{{0,1},{0,1},{0},{1}};const auto t=eclat_trace(tx,2,28);EXPECT_EQ(t.captured_entries,28u);
    EXPECT_THROW((void)eclat_trace(tx,2,27),std::length_error);
    EXPECT_THROW((void)eclat_trace(tx,2,9),std::length_error);
    EXPECT_EQ(eclat_trace_base_entries(tx),10u);EXPECT_EQ(eclat_trace(tx,2,0).captured_entries,28u);
    for(std::size_t cap=1;cap<28;++cap){EXPECT_THROW((void)eclat_trace(tx,2,cap),std::length_error);}
    auto copy=t;tx[0].clear();copy.states[0].tids.clear();copy.itemsets.clear();EXPECT_EQ(t.transactions[0].size(),2u);EXPECT_EQ(t.states[0].tids.size(),3u);EXPECT_EQ(t.itemsets.size(),3u);
    EXPECT_EQ(eclat_trace({},0,1).captured_entries,0u);EXPECT_EQ(eclat_trace({{},{}},0,2).captured_entries,2u);
    EXPECT_THROW((void)eclat_trace({{},{}},0,1),std::length_error);
}
TEST(EclatTrace, ZeroThresholdEmptyRowsFullWidthIdsAndVerticalIndex){
    const auto big=std::numeric_limits<std::size_t>::max();const Rows tx{{big,big},{},{0},{big}};
    const auto v=vertical_tidsets(tx);ASSERT_EQ(v.size(),2u);EXPECT_EQ(v[0].item,0u);EXPECT_EQ(v[0].tids,(std::vector<std::size_t>{2}));EXPECT_EQ(v[1].item,big);EXPECT_EQ(v[1].tids,(std::vector<std::size_t>{0,3}));
    const auto t=eclat_trace(tx,0);certificate(t);ASSERT_EQ(t.itemsets.size(),3u);EXPECT_EQ(t.itemsets.back().support,0u);EXPECT_TRUE(eclat_frequent_itemsets(tx,big).empty());
    EXPECT_EQ(fp_growth_frequent_itemsets({{0},{1}},0).size(),2u);EXPECT_EQ(eclat_frequent_itemsets({{0},{1}},0).size(),3u);
}
TEST(EclatTrace, ExactDenseAndDisjointWorkFamilies){
    for(std::size_t u=1;u<=12;++u){Rows dense(5);for(auto& row:dense){row.resize(u);std::iota(row.begin(),row.end(),0);}const auto r=eclat_frequent_itemsets_work(dense,5);const auto f=(std::size_t{1}<<u)-1;EXPECT_EQ(r.itemsets.size(),f);EXPECT_EQ(r.work.intersections,f-u);EXPECT_EQ(r.work.intersection_steps,5*(f-u));EXPECT_EQ(r.work.matched_tids,5*(f-u));EXPECT_EQ(r.work.peak_frames,u);
        Rows disjoint(u);for(std::size_t i=0;i<u;++i)disjoint[i]={i};const auto d=eclat_frequent_itemsets_work(disjoint,1);EXPECT_EQ(d.itemsets.size(),u);EXPECT_EQ(d.work.intersections,u*(u-1)/2);EXPECT_EQ(d.work.intersection_steps,u*(u-1)/2);EXPECT_EQ(d.work.matched_tids,0u);EXPECT_EQ(d.work.peak_frames,1u);}
    Rows wide(1);wide[0].resize(20000);std::iota(wide[0].begin(),wide[0].end(),0);
    // Reject retained input quota before normalization or a combinatorial traversal.
    EXPECT_THROW((void)eclat_trace(wide,1,20000),std::length_error);
}
