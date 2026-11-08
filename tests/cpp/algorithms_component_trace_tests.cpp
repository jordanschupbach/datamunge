#include <datamunge/algorithms/connected_components.hpp>
#include <gtest/gtest.h>
#include <algorithm>
#include <deque>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

namespace {
using namespace datamunge::algorithms;
using Image=std::vector<std::vector<int>>;
// Independent flood fill uses the full undirected graph, no provisional labels/union-find.
ConnectedComponents flood(const Image& image,bool eight) {
    ConnectedComponents out;const auto rows=image.size(),cols=rows?image[0].size():0;
    out.labels.assign(rows,std::vector<int>(cols));
    for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<cols;++c){
        if(!image[r][c]||out.labels[r][c])continue;
        const int label=++out.count;std::deque<std::pair<int,int>> q{{static_cast<int>(r),static_cast<int>(c)}};out.labels[r][c]=label;
        while(!q.empty()){
            const auto [rr,cc]=q.front();q.pop_front();
            for(int dr=-1;dr<=1;++dr)for(int dc=-1;dc<=1;++dc){
                if((!dr&&!dc)||(!eight&&dr&&dc))continue;
                const int nr=rr+dr,nc=cc+dc;
                if(nr<0||nc<0||nr>=static_cast<int>(rows)||nc>=static_cast<int>(cols))continue;
                if(image[nr][nc]&&!out.labels[nr][nc]){out.labels[nr][nc]=label;q.emplace_back(nr,nc);}
            }
        }
    }
    return out;
}
void same(const ConnectedComponents& a,const ConnectedComponents& b){EXPECT_EQ(a.count,b.count);EXPECT_EQ(a.labels,b.labels);}
int root(const std::vector<int>& p,int x){while(p[x]!=x)x=p[x];return x;}
void verify_trace(const Image& image,bool eight,const ComponentLabelTrace& t){
    same(t.analysis.result,flood(image,eight));const auto& w=t.analysis.work;
    EXPECT_EQ(w.foreground,t.cells.size());EXPECT_EQ(w.union_calls,t.unions.size());
    EXPECT_EQ(w.union_calls,w.neighbor_labels);EXPECT_EQ(w.find_calls,2*w.union_calls+w.foreground);
    EXPECT_EQ(w.remapped_cells,w.foreground);EXPECT_EQ(w.first_pass_cells,w.validated_cells);EXPECT_EQ(w.second_pass_cells,w.validated_cells);
    EXPECT_EQ(w.labels_created-w.successful_unions,static_cast<std::size_t>(t.analysis.result.count));
    EXPECT_EQ(w.parent_writes,w.find_steps+w.successful_unions);
    std::vector<int> parents{0};std::vector<unsigned> ranks{0};std::size_t u=0,merges=0;
    Image provisional(image.size(),std::vector<int>(image.empty()?0:image[0].size()));
    for(const auto& e:t.cells){
        ASSERT_NE(image[e.row][e.col],0);EXPECT_EQ(e.union_begin,u);EXPECT_LE(e.earlier_count,4);
        std::vector<int> neighbors;
        const auto consider=[&](std::size_t r,std::size_t c){if(provisional[r][c])neighbors.push_back(provisional[r][c]);};
        if(e.row)consider(e.row-1,e.col);if(e.col)consider(e.row,e.col-1);
        if(eight&&e.row&&e.col)consider(e.row-1,e.col-1);
        if(eight&&e.row&&e.col+1<provisional[0].size())consider(e.row-1,e.col+1);
        EXPECT_EQ(neighbors.size(),e.earlier_count);
        for(std::size_t k=0;k<neighbors.size();++k)EXPECT_EQ(e.earlier_labels[k],neighbors[k]);
        if(neighbors.empty()){
            EXPECT_TRUE(e.created);EXPECT_EQ(e.assigned,parents.size());parents.push_back(e.assigned);ranks.push_back(0);
        }else{
            EXPECT_FALSE(e.created);EXPECT_EQ(e.assigned,*std::min_element(neighbors.begin(),neighbors.end()));
            for(int n:neighbors){
                ASSERT_LT(u,t.unions.size());const auto& a=t.unions[u++];
                EXPECT_EQ(a.a,e.assigned);EXPECT_EQ(a.b,n);EXPECT_EQ(a.root_a,root(parents,a.a));EXPECT_EQ(a.root_b,root(parents,a.b));
                EXPECT_EQ(a.merged,a.root_a!=a.root_b);
                if(!a.merged){EXPECT_EQ(a.loser,0);EXPECT_EQ(a.winner,a.root_a);continue;}
                int winner=a.root_a,loser=a.root_b;
                if(ranks[winner]<ranks[loser]||(ranks[winner]==ranks[loser]&&winner>loser))std::swap(winner,loser);
                EXPECT_EQ(a.winner,winner);EXPECT_EQ(a.loser,loser);parents[loser]=winner;
                if(ranks[winner]==ranks[loser])++ranks[winner];++merges;
            }
        }
        EXPECT_EQ(e.union_end,u);provisional[e.row][e.col]=e.assigned;
    }
    EXPECT_EQ(t.provisional,provisional);EXPECT_EQ(merges,w.successful_unions);EXPECT_EQ(u,t.unions.size());
    ASSERT_EQ(parents.size(),t.parents.size());EXPECT_EQ(t.root_to_dense.size(),parents.size());
    for(std::size_t i=1;i<parents.size();++i)EXPECT_EQ(root(t.parents,i),root(parents,i));
    for(std::size_t r=0;r<provisional.size();++r)for(std::size_t c=0;c<provisional[r].size();++c){
        const int p=provisional[r][c];if(p)EXPECT_EQ(t.root_to_dense[root(parents,p)],t.analysis.result.labels[r][c]);
    }
}
}

TEST(ComponentTrace, BridgeRecordsActualEquivalenceAndDenseRenumbering){
    const Image image{{1,0,1},{1,1,1}};const auto t=connected_components_trace(image);
    EXPECT_EQ(t.provisional,(Image{{1,0,2},{1,1,1}}));EXPECT_EQ(t.analysis.result.labels,(Image{{1,0,1},{1,1,1}}));
    EXPECT_EQ(t.analysis.work.labels_created,2);EXPECT_EQ(t.analysis.work.successful_unions,1);
    EXPECT_EQ(t.analysis.work.union_calls,4);EXPECT_EQ(t.analysis.work.find_calls,13);EXPECT_EQ(t.analysis.work.foreground,5);
    ASSERT_EQ(t.cells.back().earlier_count,2);EXPECT_EQ(t.cells.back().earlier_labels[0],2);EXPECT_EQ(t.cells.back().earlier_labels[1],1);
    ASSERT_EQ(t.unions.size(),4);EXPECT_TRUE(t.unions[2].merged);EXPECT_EQ(t.unions[2].winner,1);EXPECT_EQ(t.unions[2].loser,2);
    verify_trace(image,false,t);
}
TEST(ComponentTrace, RankCanKeepANonminimumRootWithoutChangingFirstOccurrenceIDs){
    const Image image{{1,0,1,0,1},{1,0,1,1,1},{1,1,1,0,0}};
    const auto t=connected_components_trace(image);
    verify_trace(image,false,t);
    EXPECT_EQ(t.analysis.result.count,1);
    EXPECT_EQ(t.parents[1],2);EXPECT_EQ(t.root_to_dense[2],1);
    EXPECT_EQ(t.analysis.result.labels[0][0],1);EXPECT_EQ(t.analysis.result.labels[0][4],1);
    bool observed=false;
    for(const auto& u:t.unions)if(u.merged&&u.winner==2&&u.loser==1)observed=true;
    EXPECT_TRUE(observed);
}
TEST(ComponentTrace, EveryFourByFourBinaryMaskMatchesIndependentFloodFill){
    Image image(4,std::vector<int>(4));
    for(unsigned mask=0;mask<65536;++mask){
        for(unsigned i=0;i<16;++i)image[i/4][i%4]=(mask>>i)&1;
        for(bool eight:{false,true}){const auto expected=flood(image,eight);same(connected_components(image,eight),expected);}
    }
}
TEST(ComponentTrace, RandomActualTraceReplayAndAllNativePathsAgree){
    std::mt19937 rng(9204);
    for(int trial=0;trial<1000;++trial){
        Image image(1+rng()%12,std::vector<int>(1+rng()%12));
        for(auto& row:image)for(auto& v:row)v=(rng()%4==0?0:static_cast<int>(rng()%7)-3);
        for(bool eight:{false,true}){
            const auto t=connected_components_trace(image,eight,{0,0});verify_trace(image,eight,t);
            const auto a=connected_components_analyze(image,eight,{0,1});same(a.result,t.analysis.result);
            EXPECT_EQ(a.work.find_steps,t.analysis.work.find_steps);EXPECT_EQ(a.work.parent_writes,t.analysis.work.parent_writes);
            EXPECT_EQ(a.work.neighbor_probes,t.analysis.work.neighbor_probes);same(connected_components(image,eight),t.analysis.result);
        }
    }
}
TEST(ComponentTrace, ForegroundValuesAndDuplicateNeighborRecords){
    const Image image{{std::numeric_limits<int>::min(),0},{0,std::numeric_limits<int>::max()}};
    EXPECT_EQ(connected_components(image).count,2);EXPECT_EQ(connected_components(image,true).count,1);
    const auto t=connected_components_trace(Image(3,std::vector<int>(3,9)),true);
    const auto& e=t.cells[4];EXPECT_EQ(e.earlier_count,4);EXPECT_EQ(e.earlier_labels,(std::array<int,4>{1,1,1,1}));
    verify_trace(Image(3,std::vector<int>(3,9)),true,t);
}
TEST(ComponentTrace, EmptyZeroWidthBackgroundAndBordersRetainShape){
    for(const Image image:{Image{},Image(3),Image(4,std::vector<int>(5))}){
        const auto t=connected_components_trace(image);same(t.analysis.result,flood(image,false));
        EXPECT_EQ(t.provisional,image);EXPECT_EQ(t.parents,(std::vector<int>{0}));EXPECT_TRUE(t.cells.empty());
        EXPECT_TRUE(connected_component_stats(t.analysis.result).empty());
    }
    for(const Image image:{Image{{1,0,1,1}},Image{{1},{0},{1},{1}}})for(bool eight:{false,true})same(connected_components(image,eight),flood(image,eight));
}
TEST(ComponentTrace, FullShapeValidationAndCompleteActualCaps){
    for(const Image image:{Image{{},{1}},Image{{1,0},{1}},Image{{1},{1},{1,0}}}){
        EXPECT_THROW(connected_components(image),std::invalid_argument);
        EXPECT_THROW(connected_components_trace(image,false,{1,1}),std::invalid_argument);
    }
    const Image bridge{{1,0,1},{1,1,1}};
    EXPECT_EQ(connected_components_trace(bridge,false,{6,9}).cells.size(),5);
    EXPECT_THROW(connected_components_trace(bridge,false,{5,100}),std::length_error);
    EXPECT_THROW(connected_components_trace(bridge,false,{6,8}),std::length_error);
    EXPECT_EQ(connected_components_analyze(bridge,false,{6,1}).result.count,1);
    EXPECT_EQ(connected_components_trace(Image(8,std::vector<int>(8)),false,{64,1}).analysis.result.count,0);
    EXPECT_EQ(connected_components_trace({{1}},false,{1,1}).cells.size(),1);
    EXPECT_EQ(connected_components_trace(bridge,false,{0,0}).cells.size(),5);
}
TEST(ComponentTrace, OriginalRectanglesHaveExactNativeMeasurements){
    const auto image=component_blobs_example();const auto c=connected_components(image);
    EXPECT_EQ(c.count,4);same(c,flood(image,false));same(c,connected_components(image,true));
    const auto s=connected_component_stats(c);ASSERT_EQ(s.size(),4);
    for(std::size_t i=0;i<s.size();++i)EXPECT_EQ(s[i].label,i+1);
    EXPECT_EQ(s[0].area,9);EXPECT_EQ(s[1].area,8);EXPECT_EQ(s[2].area,8);EXPECT_EQ(s[3].area,16);
    EXPECT_EQ(s[0].perimeter4,12);EXPECT_EQ(s[1].perimeter4,12);EXPECT_EQ(s[2].perimeter4,12);EXPECT_EQ(s[3].perimeter4,16);
    EXPECT_EQ(s[0].centroid_row,2);EXPECT_EQ(s[0].centroid_col,2);
    EXPECT_EQ(s[1].centroid_row,1.5);EXPECT_EQ(s[1].centroid_col,8.5);
    EXPECT_EQ(s[2].min_row,6);EXPECT_EQ(s[2].max_row,9);EXPECT_EQ(s[2].centroid_col,2.5);
    EXPECT_EQ(s[3].centroid_row,8.5);EXPECT_EQ(s[3].centroid_col,8.5);
}
TEST(ComponentTrace, CheckerboardsConnectivityAndClosedFormCounts){
    for(std::size_t r=1;r<=24;++r)for(std::size_t c=1;c<=24;++c){
        const auto image=component_checkerboard(r,c);const auto four=connected_components_analyze(image),eight=connected_components_analyze(image,true);
        EXPECT_EQ(four.result.count,(r*c+1)/2);
        EXPECT_EQ(eight.result.count,(r==1||c==1)?(r*c+1)/2:1);
        EXPECT_EQ(four.work.labels_created,four.work.foreground);EXPECT_EQ(four.work.union_calls,0);
        EXPECT_EQ(four.work.first_pass_cells,r*c);EXPECT_EQ(eight.work.second_pass_cells,r*c);
        const auto s=connected_component_stats(four.result);for(const auto& component:s){EXPECT_EQ(component.area,1);EXPECT_EQ(component.perimeter4,4);}
    }
    EXPECT_THROW(component_checkerboard(0,2),std::invalid_argument);
    EXPECT_THROW(component_checkerboard(2,0),std::invalid_argument);
    EXPECT_THROW(component_checkerboard(std::numeric_limits<std::size_t>::max(),2,0),std::length_error);
    EXPECT_EQ(component_checkerboard(2,3,6).size(),2);
    EXPECT_THROW(component_checkerboard(2,3,5),std::length_error);
}
TEST(ComponentTrace, MeasurementsMatchIndependentSetsAndEdgeIdentity){
    std::mt19937 rng(117);
    for(int trial=0;trial<1000;++trial){
        Image image(1+rng()%12,std::vector<int>(1+rng()%12));for(auto& row:image)for(auto& v:row)v=rng()%2;
        const auto c=connected_components(image,trial%2);const auto stats=connected_component_stats(c);
        for(const auto& s:stats){
            std::set<std::pair<std::size_t,std::size_t>> pixels;std::size_t rowsum=0,colsum=0,adjacent=0;
            for(std::size_t r=0;r<image.size();++r)for(std::size_t col=0;col<image[r].size();++col)if(c.labels[r][col]==s.label){pixels.emplace(r,col);rowsum+=r;colsum+=col;}
            std::size_t minr=image.size(),minc=image[0].size(),maxr=0,maxc=0;
            for(const auto& [r,col]:pixels){
                adjacent+=pixels.count({r+1,col})+pixels.count({r,col+1});
                minr=std::min(minr,r);minc=std::min(minc,col);maxr=std::max(maxr,r);maxc=std::max(maxc,col);
            }
            EXPECT_EQ(s.area,pixels.size());EXPECT_EQ(s.perimeter4,4*pixels.size()-2*adjacent);
            EXPECT_EQ(s.min_row,minr);EXPECT_EQ(s.min_col,minc);EXPECT_EQ(s.max_row,maxr);EXPECT_EQ(s.max_col,maxc);
            EXPECT_DOUBLE_EQ(s.centroid_row,double(rowsum)/pixels.size());EXPECT_DOUBLE_EQ(s.centroid_col,double(colsum)/pixels.size());
        }
    }
}
TEST(ComponentTrace, HolesAndDiagonalContactsHaveExplicitGridPerimeter){
    const auto ring=connected_component_stats(connected_components({{1,1,1},{1,0,1},{1,1,1}}));
    ASSERT_EQ(ring.size(),1);EXPECT_EQ(ring[0].area,8);EXPECT_EQ(ring[0].perimeter4,16);
    const auto diagonal=connected_component_stats(connected_components({{1,0},{0,1}},true));
    ASSERT_EQ(diagonal.size(),1);EXPECT_EQ(diagonal[0].area,2);EXPECT_EQ(diagonal[0].perimeter4,8);
    EXPECT_EQ(diagonal[0].centroid_row,.5);EXPECT_EQ(diagonal[0].centroid_col,.5);
}
TEST(ComponentTrace, ExternalMeasurementMapsValidateBeforeHugeAllocations){
    for(const ConnectedComponents c:{ConnectedComponents{{{1}},0},ConnectedComponents{{{-1}},1},ConnectedComponents{{{1}},-1},
        ConnectedComponents{{{1}},std::numeric_limits<int>::max()},ConnectedComponents{{{1,3,3}},3},ConnectedComponents{{{1},{}},1}})
        EXPECT_THROW(connected_component_stats(c),std::invalid_argument);
    const auto disconnected=connected_component_stats(ConnectedComponents{{{1,0,1}},1});
    ASSERT_EQ(disconnected.size(),1);EXPECT_EQ(disconnected[0].area,2);EXPECT_EQ(disconnected[0].perimeter4,8);
}
TEST(ComponentTrace, IndependentOwnedInputSnapshotsAndMeasurements){
    Image image{{1,0,1},{1,1,1}};auto t=connected_components_trace(image);auto other=t;
    auto s=connected_component_stats(t.analysis.result);image[0][0]=0;t.provisional[0][0]=99;t.analysis.result.labels[0][0]=99;
    t.parents[1]=0;t.cells[0].assigned=99;t.unions[0].a=99;
    EXPECT_EQ(other.provisional[0][0],1);EXPECT_EQ(other.analysis.result.labels[0][0],1);
    EXPECT_EQ(other.parents[1],1);EXPECT_EQ(other.cells[0].assigned,1);EXPECT_EQ(s[0].area,5);
}
