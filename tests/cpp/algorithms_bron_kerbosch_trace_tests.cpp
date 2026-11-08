#include <datamunge/algorithms/bron_kerbosch.hpp>
#include <datamunge/algorithms/graph_connectivity.hpp>
#include <gtest/gtest.h>
#include <climits>
#include <random>
#include <set>
namespace bk=datamunge::algorithms;
namespace {
using Matrix=std::vector<std::vector<bool>>;
Matrix union_matrix(int n,const std::vector<std::vector<int>>& adjacency) {
    Matrix matrix(n,std::vector<bool>(n,false));
    for (int u=0;u<n;++u) for (int v:adjacency[u]) if (u!=v) matrix[u][v]=matrix[v][u]=true;
    return matrix;
}
std::set<std::vector<int>> subset_oracle(const Matrix& matrix) {
    const int n=matrix.size();std::set<std::vector<int>> maximal;
    for (unsigned mask=0;mask<(1u<<n);++mask) {
        std::vector<int> clique;
        for (int u=0;u<n;++u) if (mask&(1u<<u)) clique.push_back(u);
        bool complete=true;
        for (int u:clique) for (int v:clique) if (u!=v && !matrix[u][v]) complete=false;
        if (!complete) continue;
        bool extendable=false;
        for (int v=0;v<n;++v) if (!(mask&(1u<<v))) {
            bool all=true;for (int u:clique) all=all && matrix[u][v];
            extendable=extendable || all;
        }
        if (!extendable) maximal.insert(clique);
    }
    return maximal;
}
void verify_trace(const bk::BronKerboschTrace& trace,const Matrix& matrix) {
    ASSERT_EQ(trace.nodes.size(),trace.recursive_calls);
    EXPECT_EQ(trace.branches+1,trace.recursive_calls);
    std::size_t pivot_candidates=0,pivot_tests=0,branch_tests=0,intersection_tests=0,max_depth=0;
    std::vector<std::vector<int>> reports;
    std::vector<std::size_t> histogram(matrix.size()+1,0);
    for (const auto& node:trace.nodes) {
        ASSERT_LT(node.id,trace.nodes.size());EXPECT_EQ(&node,&trace.nodes[node.id]);
        max_depth=std::max(max_depth,node.r.size());
        std::set<int> r(node.r.begin(),node.r.end()),p(node.p.begin(),node.p.end()),x(node.x.begin(),node.x.end());
        EXPECT_EQ(r.size(),node.r.size());EXPECT_EQ(p.size(),node.p.size());EXPECT_EQ(x.size(),node.x.size());
        for (int u:r) for (int v:r) if (u!=v) EXPECT_TRUE(matrix[u][v]);
        for (int v=0;v<int(matrix.size());++v) {
            bool eligible=!r.contains(v);for (int u:r) eligible=eligible && matrix[u][v];
            EXPECT_EQ(p.contains(v)||x.contains(v),eligible);
            EXPECT_FALSE(p.contains(v)&&x.contains(v));
        }
        EXPECT_EQ(node.reported,node.p.empty() && node.x.empty());
        if (node.reported) {
            auto sorted=node.r;std::sort(sorted.begin(),sorted.end());reports.push_back(sorted);++histogram[sorted.size()];
            EXPECT_EQ(node.pivot,-1);EXPECT_TRUE(node.branch_vertices.empty());
        } else {
            std::vector<int> possible=node.p;possible.insert(possible.end(),node.x.begin(),node.x.end());
            int chosen=-1,best=-1;
            for (int u:possible) {
                int degree=0;for (int v:node.p) degree+=matrix[u][v];
                if (degree>best) {best=degree;chosen=u;}
            }
            EXPECT_EQ(node.pivot,chosen);EXPECT_EQ(node.pivot_degree,best);
            std::vector<int> expected;
            for (int v:node.p) if (!matrix[chosen][v]) expected.push_back(v);
            EXPECT_EQ(node.branch_vertices,expected);
        }
        pivot_candidates+=node.p.size()+node.x.size();
        pivot_tests+=(node.p.size()+node.x.size())*node.p.size();
        branch_tests+=node.p.size();
        if (node.id==0) {EXPECT_EQ(node.parent,0u);EXPECT_EQ(node.via_vertex,-1);continue;}
        ASSERT_LT(node.parent,node.id);
        const auto& parent=trace.nodes[node.parent];
        auto current_p=parent.p,current_x=parent.x;
        bool found=false;
        for (int previous:parent.branch_vertices) {
            if (previous==node.via_vertex) {found=true;break;}
            current_p.erase(std::remove(current_p.begin(),current_p.end(),previous),current_p.end());
            current_x.push_back(previous);
        }
        EXPECT_TRUE(found);
        auto child_r=parent.r;child_r.push_back(node.via_vertex);EXPECT_EQ(node.r,child_r);
        std::vector<int> child_p,child_x;
        for (int v:current_p) if (matrix[node.via_vertex][v]) child_p.push_back(v);
        for (int v:current_x) if (matrix[node.via_vertex][v]) child_x.push_back(v);
        EXPECT_EQ(node.p,child_p);EXPECT_EQ(node.x,child_x);
        intersection_tests+=parent.p.size()+parent.x.size();
    }
    EXPECT_EQ(trace.cliques,reports);EXPECT_EQ(trace.clique_size_counts,histogram);
    EXPECT_EQ(trace.pivot_candidates,pivot_candidates);EXPECT_EQ(trace.pivot_tests,pivot_tests);
    EXPECT_EQ(trace.branch_tests,branch_tests);EXPECT_EQ(trace.intersection_tests,intersection_tests);
    EXPECT_EQ(trace.max_depth,max_depth);
}
}
TEST(BronKerboschTrace, ExhaustiveGraphsThroughSixVerticesMatchSubsetMaximality) {
    std::size_t graphs=0;
    for (int n=0;n<=6;++n) {
        const unsigned choices=1u<<(n*(n-1)/2);
        for (unsigned mask=0;mask<choices;++mask) {
            std::vector<std::vector<int>> adjacency(n);int bit=0;
            for (int u=0;u<n;++u) for (int v=u+1;v<n;++v,++bit) if (mask&(1u<<bit)) {
                adjacency[u].push_back(v);adjacency[v].push_back(u);
            }
            const auto matrix=union_matrix(n,adjacency);
            const auto trace=bk::bron_kerbosch_trace(n,adjacency);
            const auto expected=subset_oracle(matrix);
            EXPECT_EQ((std::set<std::vector<int>>(trace.cliques.begin(),trace.cliques.end())),expected);
            EXPECT_EQ(trace.cliques.size(),expected.size());
            EXPECT_EQ(bk::bron_kerbosch(n,adjacency),trace.cliques);
            verify_trace(trace,matrix);++graphs;
        }
    }
    EXPECT_EQ(graphs,33868u);
}
TEST(BronKerboschTrace, OriginalDiscoveryOrderAndExcludedNonmaximalFrameAreRetained) {
    const std::vector<std::vector<int>> adjacency{{1,2},{0,2},{0,1,3,4,5},{2,4,5},{2,3,5},{2,3,4,6},{5}};
    const auto trace=bk::bron_kerbosch_trace(7,adjacency);
    EXPECT_EQ(trace.cliques,(std::vector<std::vector<int>>{{0,1,2},{2,3,4,5},{5,6}}));
    EXPECT_EQ(trace.recursive_calls,10u);EXPECT_EQ(trace.edge_count,10u);EXPECT_EQ(trace.input_adjacency_entries,20u);
    EXPECT_EQ(trace.nodes[0].pivot,2);EXPECT_EQ(trace.nodes[0].branch_vertices,(std::vector<int>{2,6}));
    bool excluded=false;
    for (const auto& node:trace.nodes) if (node.r==std::vector<int>{2,1}) {
        EXPECT_TRUE(node.p.empty());EXPECT_EQ(node.x,(std::vector<int>{0}));
        EXPECT_FALSE(node.reported);excluded=true;
    }
    EXPECT_TRUE(excluded);verify_trace(trace,union_matrix(7,adjacency));
    auto inferred=bk::bron_kerbosch;EXPECT_EQ(inferred(7,adjacency),trace.cliques);
}
TEST(BronKerboschTrace, UnionSymmetrisationDuplicatesLoopsAndIgnoredRowsPreserveOutput) {
    std::mt19937 rng(1973);
    for (int trial=0;trial<1000;++trial) {
        const int n=1+rng()%10;std::vector<std::vector<int>> adjacency(n);
        for (int i=0;i<4*n;++i) adjacency[rng()%n].push_back(rng()%n);
        const auto matrix=union_matrix(n,adjacency);
        std::vector<std::vector<int>> canonical(n);
        for (int u=0;u<n;++u) for (int v=0;v<n;++v) if (matrix[u][v]) canonical[u].push_back(v);
        for (int u=0;u<n;++u) {
            const auto old=adjacency[u];adjacency[u].insert(adjacency[u].end(),old.begin(),old.end());
            adjacency[u].push_back(u);std::shuffle(adjacency[u].begin(),adjacency[u].end(),rng);
        }
        adjacency.push_back({-1,INT_MAX});const auto saved=adjacency;
        const auto trace=bk::bron_kerbosch_trace(n,adjacency);
        EXPECT_EQ(trace.cliques,bk::bron_kerbosch(n,canonical));
        EXPECT_EQ((std::set<std::vector<int>>(trace.cliques.begin(),trace.cliques.end())),subset_oracle(matrix));
        EXPECT_EQ(adjacency,saved);verify_trace(trace,matrix);
    }
}
TEST(BronKerboschTrace, EmptyGraphIncludesEmptyCliqueAndIsolatesAreMaximal) {
    const auto empty=bk::bron_kerbosch_trace(0,{{-1}});
    EXPECT_EQ(empty.cliques,(std::vector<std::vector<int>>{{}}));
    EXPECT_EQ(empty.clique_size_counts,(std::vector<std::size_t>{1}));
    EXPECT_EQ(empty.recursive_calls,1u);EXPECT_EQ(empty.max_depth,0u);EXPECT_EQ(empty.input_adjacency_entries,0u);
    const auto isolates=bk::bron_kerbosch_trace(50,std::vector<std::vector<int>>(50));
    EXPECT_EQ(isolates.cliques.size(),50u);EXPECT_EQ(isolates.clique_size_counts[1],50u);
    EXPECT_EQ(isolates.recursive_calls,51u);EXPECT_EQ(isolates.max_depth,1u);
}
TEST(BronKerboschTrace, CompleteGraphHasOneCliqueAndOnePathOfCalls) {
    const auto adjacency=bk::complete_multipartite_graph(std::vector<int>(50,1));
    const auto trace=bk::bron_kerbosch_trace(50,adjacency);
    ASSERT_EQ(trace.cliques.size(),1u);EXPECT_EQ(trace.cliques[0].size(),50u);
    EXPECT_EQ(trace.recursive_calls,51u);EXPECT_EQ(trace.max_depth,50u);EXPECT_EQ(trace.clique_size_counts[50],1u);
    verify_trace(trace,union_matrix(50,adjacency));
}
TEST(BronKerboschTrace, CompleteMultipartiteTriplesHaveExactExponentialOutputAndCallCounts) {
    std::size_t expected=1;
    for (int q=0;q<=8;++q) {
        if (q) expected*=3;
        const auto adjacency=bk::complete_multipartite_graph(std::vector<int>(q,3));
        const auto trace=bk::bron_kerbosch_trace(3*q,adjacency);
        EXPECT_EQ(trace.cliques.size(),expected);EXPECT_EQ(trace.clique_size_counts[q],expected);
        EXPECT_EQ(trace.recursive_calls,(3*expected-1)/2);EXPECT_EQ(trace.max_depth,q);
        std::set<std::vector<int>> unique(trace.cliques.begin(),trace.cliques.end());EXPECT_EQ(unique.size(),expected);
        for (const auto& clique:trace.cliques) {
            ASSERT_EQ(clique.size(),q);
            for (int part=0;part<q;++part) EXPECT_EQ(clique[part]/3,part);
        }
    }
}
TEST(BronKerboschTrace, MultipartiteGeneratorHasSortedSymmetricRowsAndGeneralProductCounts) {
    const auto adjacency=bk::complete_multipartite_graph({1,2,4});
    EXPECT_EQ(adjacency.size(),7u);
    const auto trace=bk::bron_kerbosch_trace(7,adjacency);
    EXPECT_EQ(trace.cliques.size(),8u);EXPECT_EQ(trace.edge_count,14u);EXPECT_EQ(trace.clique_size_counts[3],8u);
    for (int u=0;u<7;++u) {
        EXPECT_TRUE(std::is_sorted(adjacency[u].begin(),adjacency[u].end()));
        EXPECT_EQ(std::adjacent_find(adjacency[u].begin(),adjacency[u].end()),adjacency[u].end());
        for (int v:adjacency[u]) {EXPECT_NE(u,v);EXPECT_TRUE(std::binary_search(adjacency[v].begin(),adjacency[v].end(),u));}
    }
    EXPECT_THROW(bk::complete_multipartite_graph({0}),std::invalid_argument);
    EXPECT_THROW(bk::complete_multipartite_graph({-1,3}),std::invalid_argument);
    EXPECT_THROW(bk::complete_multipartite_graph({INT_MAX,1}),std::overflow_error);
}
TEST(BronKerboschTrace, InvalidInputsAreRejectedBeforeMatrixAllocation) {
    EXPECT_THROW(bk::bron_kerbosch(-1,{}),std::invalid_argument);
    EXPECT_THROW(bk::bron_kerbosch_trace(-1,{}),std::invalid_argument);
    EXPECT_THROW(bk::bron_kerbosch(INT_MAX,{}),std::invalid_argument);
    EXPECT_THROW(bk::bron_kerbosch_trace(3,{{1},{0}}),std::invalid_argument);
    EXPECT_THROW(bk::bron_kerbosch(2,{{-1},{}}),std::out_of_range);
    EXPECT_THROW(bk::bron_kerbosch_trace(2,{{},{2}}),std::out_of_range);
    EXPECT_THROW(bk::bron_kerbosch(1,{{0,INT_MAX}}),std::out_of_range);
}
TEST(BronKerboschTrace, PositiveCallBudgetReturnsOnlyCompleteResultsOrThrows) {
    const std::vector<std::vector<int>> adjacency{{1,2},{0,2},{0,1,3,4,5},{2,4,5},{2,3,5},{2,3,4,6},{5}};
    const auto complete=bk::bron_kerbosch_trace(7,adjacency);
    for (std::size_t budget=1;budget<complete.recursive_calls;++budget) {
        EXPECT_THROW(bk::bron_kerbosch_trace(7,adjacency,budget),std::runtime_error);
    }
    EXPECT_EQ(bk::bron_kerbosch_trace(7,adjacency,complete.recursive_calls).cliques,complete.cliques);
    EXPECT_EQ(bk::bron_kerbosch_trace(0,{},1).cliques,(std::vector<std::vector<int>>{{}}));
}
