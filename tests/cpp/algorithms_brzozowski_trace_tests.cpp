#include <datamunge/algorithms/brzozowski.hpp>
#include <datamunge/algorithms/dfa_minimization.hpp>
#include <gtest/gtest.h>
#include <climits>
#include <random>
#include <set>
namespace fa=datamunge::algorithms;
namespace {
// Independent reference always adds a virtual sink, whether or not it is reachable.
fa::DFA reference_complete(const fa::DFA& d) {
    fa::DFA c{d.num_states+1,d.num_symbols,{},d.start<0?d.num_states:d.start,d.accept};
    c.accept.push_back(0);c.delta.assign(c.num_states,std::vector<int>(c.num_symbols,d.num_states));
    for (int q=0;q<d.num_states;++q) for (int s=0;s<d.num_symbols;++s)
        if (d.delta[q][s]>=0) c.delta[q][s]=d.delta[q][s];
    return c;
}
std::vector<std::vector<bool>> marked_pairs(const fa::DFA& d) {
    const int n=d.num_states;std::vector<std::vector<bool>> marked(n,std::vector<bool>(n,false));
    for (int p=0;p<n;++p) for (int q=0;q<n;++q) marked[p][q]=(d.accept[p]!=0)!=(d.accept[q]!=0);
    bool changed=true;
    while (changed) {
        changed=false;
        for (int p=0;p<n;++p) for (int q=0;q<n;++q) if (!marked[p][q])
            for (int s=0;s<d.num_symbols;++s) if (marked[d.delta[p][s]][d.delta[q][s]]) {
                marked[p][q]=true;changed=true;break;
            }
    }
    return marked;
}
int minimal_count(const fa::DFA& source) {
    const auto d=reference_complete(source);const auto marked=marked_pairs(d);
    std::vector<bool> reachable(d.num_states,false);reachable[d.start]=true;
    bool changed=true;
    while (changed) {changed=false;for (int p=0;p<d.num_states;++p) if (reachable[p]) for (int q:d.delta[p])
        if (!reachable[q]) {reachable[q]=true;changed=true;}}
    int classes=0;
    for (int p=0;p<d.num_states;++p) if (reachable[p]) {
        bool earlier=false;for (int q=0;q<p;++q) if (reachable[q] && !marked[p][q]) earlier=true;
        if (!earlier) ++classes;
    }
    return classes;
}
bool equal_language(const fa::DFA& a,const fa::DFA& b) {
    auto left=reference_complete(a);const auto right=reference_complete(b);const int offset=left.num_states;
    left.num_states+=right.num_states;left.accept.insert(left.accept.end(),right.accept.begin(),right.accept.end());
    for (auto row:right.delta) {for (int& q:row) q+=offset;left.delta.push_back(std::move(row));}
    return !marked_pairs(left)[left.start][right.start+offset];
}
bool reference_accepts(const fa::DFA& d,const std::vector<int>& word) {
    int q=d.start;for (int s:word) {if (q<0) return false;q=d.delta[q][s];}
    return q>=0 && d.accept[q]!=0;
}
void same_dfa(const fa::DFA& a,const fa::DFA& b) {
    EXPECT_EQ(a.num_states,b.num_states);EXPECT_EQ(a.num_symbols,b.num_symbols);EXPECT_EQ(a.start,b.start);
    EXPECT_EQ(a.delta,b.delta);EXPECT_EQ(a.accept,b.accept);
}
void verify_subsets(const fa::NFA& n,const fa::PowersetTrace& trace) {
    const auto& d=trace.dfa;ASSERT_EQ(trace.subsets.size(),d.num_states);
    EXPECT_EQ(d.num_symbols,n.num_symbols);fa::validate_dfa(d);
    std::set<std::vector<int>> unique(trace.subsets.begin(),trace.subsets.end());EXPECT_EQ(unique.size(),trace.subsets.size());
    if (!trace.subsets.empty()) {
        auto initial=n.start_states;std::sort(initial.begin(),initial.end());initial.erase(std::unique(initial.begin(),initial.end()),initial.end());
        EXPECT_EQ(trace.subsets[0],initial);EXPECT_EQ(d.start,0);
    } else EXPECT_EQ(d.start,-1);
    std::size_t arcs=0,unions=0,largest=0;
    for (int q=0;q<d.num_states;++q) {
        const auto& subset=trace.subsets[q];EXPECT_TRUE(std::is_sorted(subset.begin(),subset.end()));
        EXPECT_EQ(std::adjacent_find(subset.begin(),subset.end()),subset.end());largest=std::max(largest,subset.size());
        bool accepting=false;for (int state:subset) accepting=accepting || n.accept[state]!=0;
        EXPECT_EQ(d.accept[q]!=0,accepting);
        for (int s=0;s<n.num_symbols;++s) {
            ++unions;std::set<int> expected;
            for (int state:subset) for (int target:n.delta[state][s]) {expected.insert(target);++arcs;}
            const std::vector<int> target(expected.begin(),expected.end());
            if (target.empty() && !trace.complete) EXPECT_EQ(d.delta[q][s],-1);
            else {ASSERT_GE(d.delta[q][s],0);ASSERT_LT(d.delta[q][s],d.num_states);EXPECT_EQ(trace.subsets[d.delta[q][s]],target);}
        }
    }
    EXPECT_EQ(trace.transition_unions,unions);EXPECT_EQ(trace.nfa_arcs_examined,arcs);EXPECT_EQ(trace.max_subset_size,largest);
}
void verify_pipeline(const fa::DFA& d,const fa::BrzozowskiTrace& trace) {
    verify_subsets(trace.first_reversal,trace.first);verify_subsets(trace.second_reversal,trace.second);
    EXPECT_EQ(trace.second.dfa.num_states,minimal_count(d));EXPECT_TRUE(equal_language(d,trace.second.dfa));
    EXPECT_EQ(fa::dfa_reachable_states(trace.second.dfa).size(),trace.second.dfa.num_states);
    EXPECT_EQ(trace.reachable_input_states,fa::dfa_reachable_states(d));
    const auto exact=fa::dfa_equivalence(d,trace.second.dfa);EXPECT_TRUE(exact.equivalent);EXPECT_TRUE(exact.witness.empty());
    same_dfa(trace.second.dfa,fa::brzozowski_minimize(d));
}
fa::DFA original() {return {6,2,{{1,2},{0,3},{4,5},{4,5},{4,5},{5,5}},0,{0,0,1,1,1,0}};}
}
TEST(BrzozowskiTrace, ExhaustiveCompleteBinaryDfasThroughThreeStatesMatchTableFilling) {
    std::size_t machines=0;
    for (int n=1;n<=3;++n) {
        std::size_t transitions=1;for (int i=0;i<2*n;++i) transitions*=n;
        for (std::size_t code=0;code<transitions;++code) {
            fa::DFA d{n,2,std::vector<std::vector<int>>(n,std::vector<int>(2)),0,std::vector<char>(n)};
            auto digits=code;for (auto& row:d.delta) for (auto& q:row) {q=digits%n;digits/=n;}
            for (int flags=0;flags<(1<<n);++flags) {
                for (int q=0;q<n;++q) d.accept[q]=(flags>>q)&1;
                for (d.start=0;d.start<n;++d.start) {verify_pipeline(d,fa::brzozowski_minimize_trace(d));++machines;}
            }
        }
    }
    EXPECT_EQ(machines,17626u);
}
TEST(BrzozowskiTrace, RandomPartialAndNoStartMachinesMatchIndependentLanguageAndSizeOracle) {
    std::mt19937 rng(1963);
    for (int trial=0;trial<1000;++trial) {
        const int n=rng()%7,k=rng()%4;
        fa::DFA d{n,k,std::vector<std::vector<int>>(n,std::vector<int>(k)),int(rng()%(n+1))-1,std::vector<char>(n)};
        for (auto& row:d.delta) for (int& q:row) q=int(rng()%(n+1))-1;
        for (auto& accepting:d.accept) accepting=rng()%2?2:0;
        const auto saved=d;verify_pipeline(d,fa::brzozowski_minimize_trace(d));same_dfa(d,saved);
    }
}
TEST(BrzozowskiTrace, OriginalSixStateExampleRetainsActualSubsetsAndThreeStateResult) {
    const auto d=original();const auto trace=fa::brzozowski_minimize_trace(d);
    EXPECT_EQ(trace.reachable_input_states,(std::vector<int>{0,1,2,3,4,5}));
    EXPECT_EQ(trace.first.subsets,(std::vector<std::vector<int>>{{2,3,4},{0,1},{}}));
    EXPECT_EQ(trace.second.subsets,(std::vector<std::vector<int>>{{1},{0},{}}));
    EXPECT_EQ(trace.first.dfa.num_states,3);EXPECT_EQ(trace.second.dfa.num_states,3);
    verify_pipeline(d,trace);
    fa::DFA (*legacy_minimize)(const fa::DFA&)=fa::brzozowski_minimize;
    fa::NFA (*legacy_reverse)(const fa::DFA&)=fa::reverse_dfa;
    fa::DFA (*legacy_complete)(const fa::DFA&)=fa::complete_dfa;
    same_dfa(legacy_minimize(d),trace.second.dfa);EXPECT_EQ(legacy_reverse(d).start_states,(std::vector<int>{2,3,4}));
    same_dfa(legacy_complete(d),d);
}
TEST(BrzozowskiTrace, UnreachableAcceptingStateCanMakeFirstPassNonminimal) {
    const fa::DFA d{2,1,{{0},{0}},0,{0,1}};
    const auto trace=fa::brzozowski_minimize_trace(d);
    EXPECT_EQ(trace.reachable_input_states,(std::vector<int>{0}));
    EXPECT_EQ(trace.first.dfa.num_states,2);EXPECT_EQ(minimal_count(trace.first.dfa),1);
    EXPECT_EQ(trace.second.dfa.num_states,1);EXPECT_TRUE(equal_language(d,trace.second.dfa));
    EXPECT_FALSE(fa::dfa_accepts(trace.second.dfa,{}));
}
TEST(BrzozowskiTrace, EmptyDefaultZeroAlphabetAndImplicitSinkHaveDefinedLanguages) {
    const fa::DFA empty;fa::validate_dfa(empty);EXPECT_EQ(empty.start,-1);
    const fa::NFA no_states;fa::validate_nfa(no_states);
    for (const auto& d:std::vector<fa::DFA>{empty,{1,1,{{-1}},0,{1}},{2,0,{{},{}},1,{0,1}},
        {2,2,{{0,1},{1,0}},-1,{1,1}},{1,1,{{0}},0,{7}}}) verify_pipeline(d,fa::brzozowski_minimize_trace(d));
    const auto partial=fa::powerset_construction(no_states,false);EXPECT_EQ(partial.num_states,0);EXPECT_EQ(partial.start,-1);
    EXPECT_FALSE(fa::dfa_accepts(partial,{}));EXPECT_EQ(fa::brzozowski_minimize(partial).num_states,1);
}
TEST(BrzozowskiTrace, BinaryPositionFamilyHasExactExponentialFirstPassAndLinearFinalSize) {
    for (int position=1;position<=10;++position) {
        const auto d=fa::binary_kth_symbol_dfa(position);const auto trace=fa::brzozowski_minimize_trace(d);
        EXPECT_EQ(d.num_states,position+2);EXPECT_EQ(trace.first.dfa.num_states,1<<position);
        EXPECT_EQ(trace.second.dfa.num_states,position+2);EXPECT_TRUE(fa::dfa_equivalence(d,trace.second.dfa).equivalent);
        for (int length=0;length<=position+2;++length) for (int mask=0;mask<(1<<length);++mask) {
            std::vector<int> word(length);for (int i=0;i<length;++i) word[i]=(mask>>(length-1-i))&1;
            const bool expected=length>=position && word[position-1]==1;
            EXPECT_EQ(reference_accepts(d,word),expected);
            const bool reversed=length>=position && word[length-position]==1;
            EXPECT_EQ(reference_accepts(trace.first.dfa,word),reversed);
        }
    }
}
TEST(BrzozowskiTrace, ExactEquivalenceMatchesTableFillingAndShortestLexicographicWordEnumeration) {
    std::mt19937 rng(10105318);
    for (int trial=0;trial<500;++trial) {
        const int alphabet=rng()%3;
        auto random_machine=[&] {
            const int n=rng()%4;
            fa::DFA d{n,alphabet,std::vector<std::vector<int>>(n,std::vector<int>(alphabet)),int(rng()%(n+1))-1,std::vector<char>(n)};
            for (auto& row:d.delta) for (int& q:row) q=int(rng()%(n+1))-1;
            for (char& a:d.accept) a=rng()%2;return d;
        };
        const auto left=random_machine(),right=random_machine();const auto result=fa::dfa_equivalence(left,right);
        const bool expected=equal_language(left,right);EXPECT_EQ(result.equivalent,expected);
        if (expected) {EXPECT_TRUE(result.witness.empty());EXPECT_EQ(result.pairs_reached,result.pairs_examined);continue;}
        EXPECT_NE(reference_accepts(left,result.witness),reference_accepts(right,result.witness));
        ASSERT_LT(result.witness.size(),std::size_t((left.num_states+1)*(right.num_states+1)));
        bool found=false;std::vector<int> first;
        for (std::size_t length=0;length<=result.witness.size() && !found;++length) {
            std::size_t count=1;for (std::size_t i=0;i<length;++i) count*=alphabet;
            for (std::size_t code=0;code<count;++code) {
                std::vector<int> word(length);auto digits=code;
                for (std::size_t i=length;i>0;--i) {word[i-1]=digits%alphabet;digits/=alphabet;}
                if (reference_accepts(left,word)!=reference_accepts(right,word)) {first=word;found=true;break;}
            }
        }
        EXPECT_TRUE(found);EXPECT_EQ(result.witness,first);
    }
}
TEST(BrzozowskiTrace, ProductBudgetsThrowAndSparseComparisonDoesNotNeedCartesianAllocation) {
    const fa::DFA a{3,1,{{1},{2},{0}},0,{0,0,0}},b{4,1,{{1},{2},{3},{0}},0,{0,0,0,0}};
    EXPECT_THROW(fa::dfa_equivalence(a,b,11),std::runtime_error);
    const auto exact=fa::dfa_equivalence(a,b,12);EXPECT_TRUE(exact.equivalent);EXPECT_EQ(exact.pairs_reached,12u);
    fa::DFA cycle{2000,1,std::vector<std::vector<int>>(2000,std::vector<int>(1)),0,std::vector<char>(2000,0)};
    for (int q=0;q<2000;++q) cycle.delta[q][0]=(q+1)%2000;
    EXPECT_EQ(fa::dfa_equivalence(cycle,cycle,2000).pairs_reached,2000u);
    const auto epsilon=fa::dfa_equivalence(fa::DFA{1,1,{{0}},0,{1}},fa::DFA{1,1,{{0}},0,{0}},1);
    EXPECT_FALSE(epsilon.equivalent);EXPECT_TRUE(epsilon.witness.empty());EXPECT_EQ(epsilon.pairs_examined,1u);
}
TEST(BrzozowskiTrace, SubsetCompletionDuplicatesAndOutputStateCapsAreExact) {
    const fa::NFA n{2,2,{{{0,0},{0,1,1}},{{},{}}},{0,0},{0,2}};
    for (bool complete:{false,true}) {
        const auto trace=fa::powerset_construction_trace(n,complete);verify_subsets(n,trace);
        same_dfa(trace.dfa,fa::powerset_construction(n,complete));EXPECT_EQ(trace.dfa.num_states,2);
        EXPECT_THROW(fa::powerset_construction_trace(n,complete,1),std::runtime_error);
        EXPECT_EQ(fa::powerset_construction_trace(n,complete,2).dfa.num_states,2);
    }
    const fa::NFA empty{0,2,{}, {},{}};
    EXPECT_EQ(fa::powerset_construction_trace(empty,false,1).dfa.num_states,0);
    EXPECT_EQ(fa::powerset_construction_trace(empty,true,1).dfa.num_states,1);
}
TEST(BrzozowskiTrace, BothStageCapsFailWithoutReturningPartialMinimisation) {
    const auto d=fa::binary_kth_symbol_dfa(3);const auto saved=d;
    EXPECT_THROW(fa::brzozowski_minimize_trace(d,{7,0}),std::runtime_error);
    EXPECT_THROW(fa::brzozowski_minimize_trace(d,{8,4}),std::runtime_error);
    EXPECT_EQ(fa::brzozowski_minimize_trace(d,{8,5}).second.dfa.num_states,5);same_dfa(d,saved);
}
TEST(BrzozowskiTrace, MalformedDfasFailValidationBeforeAllPublicOperations) {
    const fa::DFA valid{1,1,{{0}},0,{0}};std::vector<fa::DFA> bad;
    auto d=valid;d.num_states=-1;bad.push_back(d);d=valid;d.num_symbols=-1;bad.push_back(d);
    d=valid;d.delta.clear();bad.push_back(d);d=valid;d.accept.clear();bad.push_back(d);
    d=valid;d.delta[0].push_back(0);bad.push_back(d);d=valid;d.start=1;bad.push_back(d);
    d=valid;d.start=-2;bad.push_back(d);d=valid;d.delta[0][0]=1;bad.push_back(d);
    d=valid;d.delta[0][0]=-2;bad.push_back(d);bad.push_back({0,0,{},0,{}});
    for (const auto& machine:bad) {
        EXPECT_THROW(fa::validate_dfa(machine),std::invalid_argument);
        EXPECT_THROW(fa::reverse_dfa(machine),std::invalid_argument);
        EXPECT_THROW(fa::complete_dfa(machine),std::invalid_argument);
        EXPECT_THROW(fa::dfa_accepts(machine,{}),std::invalid_argument);
        EXPECT_THROW(fa::dfa_reachable_states(machine),std::invalid_argument);
        EXPECT_THROW(fa::brzozowski_minimize(machine),std::invalid_argument);
        EXPECT_THROW(fa::brzozowski_minimize_trace(machine),std::invalid_argument);
        EXPECT_THROW(fa::dfa_equivalence(machine,valid),std::invalid_argument);
        EXPECT_THROW(fa::moore_minimize(machine),std::invalid_argument);
        EXPECT_THROW(fa::hopcroft_minimize(machine),std::invalid_argument);
    }
    EXPECT_THROW(fa::dfa_equivalence(valid,fa::DFA{1,0,{{}},0,{0}}),std::invalid_argument);
}
TEST(BrzozowskiTrace, MalformedNfasAndAllInvalidWordSymbolsAreRejected) {
    const fa::NFA valid{1,1,{{{0}}},{0},{1}};std::vector<fa::NFA> bad;
    auto n=valid;n.num_states=-1;bad.push_back(n);n=valid;n.num_symbols=-1;bad.push_back(n);
    n=valid;n.delta.clear();bad.push_back(n);n=valid;n.accept.clear();bad.push_back(n);
    n=valid;n.delta[0].clear();bad.push_back(n);n=valid;n.start_states={-1};bad.push_back(n);
    n=valid;n.delta[0][0]={1};bad.push_back(n);n=valid;n.delta[0][0]={-1};bad.push_back(n);
    for (const auto& machine:bad) {
        EXPECT_THROW(fa::validate_nfa(machine),std::invalid_argument);
        EXPECT_THROW(fa::powerset_construction(machine),std::invalid_argument);
        EXPECT_THROW(fa::powerset_construction_trace(machine),std::invalid_argument);
    }
    const fa::DFA missing{1,1,{{-1}},0,{1}};
    EXPECT_THROW(fa::dfa_accepts(missing,{-1}),std::invalid_argument);
    EXPECT_THROW(fa::dfa_accepts(missing,{1}),std::invalid_argument);
    EXPECT_THROW(fa::dfa_accepts(missing,{0,1}),std::invalid_argument);
}
TEST(BrzozowskiTrace, PositionFixtureChecksDomainAndCountBeforeAllocation) {
    EXPECT_THROW(fa::binary_kth_symbol_dfa(0),std::invalid_argument);
    EXPECT_THROW(fa::binary_kth_symbol_dfa(-1),std::invalid_argument);
    EXPECT_THROW(fa::binary_kth_symbol_dfa(INT_MAX),std::overflow_error);
}
