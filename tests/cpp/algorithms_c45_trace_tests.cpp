#include <gtest/gtest.h>
#include <datamunge/algorithms/decision_tree_induction.hpp>
#include <random>
#include <datamunge/datasets/gaussian_blobs.hpp>
#include <numeric>
#include <tuple>
namespace {
namespace a=datamunge::algorithms;
using Matrix=std::vector<std::vector<double>>;
// Independent materialized-label oracle: no production entropy/split helpers.
double entropy(const std::vector<int>& y) {
  std::map<int,std::size_t> frequency;
  for (int label:y) ++frequency[label];
  double h=0;
  for (const auto& [label,count]:frequency) {
    const double p=static_cast<double>(count)/y.size();h-=p*std::log2(p);
  }
  return h;
}
int majority(const std::vector<int>& y) {
  std::map<int,int> count;for (int label:y) ++count[label];
  int best=0,maximum=0;for (auto [label,n]:count) if(n>maximum) {best=label;maximum=n;}
  return best;
}
struct Oracle {
  a::DecisionTree tree;
  std::size_t candidates=0,row_tests=0;
  int build(const Matrix& x,const std::vector<int>& y,a::C45Options options,std::size_t depth=0) {
    const int id=tree.nodes.size();tree.nodes.push_back({});tree.nodes[id].prediction=majority(y);
    const bool pure=std::all_of(y.begin(),y.end(),[&](int label){return label==y[0];});
    if (pure || y.size()<options.min_samples || (options.max_depth && depth>=options.max_depth)) return id;
    double best=0,cut=0;std::size_t feature=0;bool found=false;
    for (std::size_t f=0;f<x[0].size();++f) {
      std::set<double> distinct;for (const auto& row:x) distinct.insert(row[f]);
      for (auto lower=distinct.begin();lower!=distinct.end();++lower) {
        auto upper=std::next(lower);if(upper==distinct.end()) break;
        double threshold=(*lower+*upper)/2; // Oracle inputs are small exact integers.
        std::vector<int> left,right;
        for (std::size_t i=0;i<y.size();++i) (x[i][f]<=threshold?left:right).push_back(y[i]);
        const double p=static_cast<double>(left.size())/y.size(),q=static_cast<double>(right.size())/y.size();
        const double gain=entropy(y)-(p*entropy(left)+q*entropy(right));
        const double split=-p*std::log2(p)-q*std::log2(q);
        const double ratio=gain/split;++candidates;row_tests+=y.size();
        if (ratio>best) {best=ratio;cut=threshold;feature=f;found=true;}
      }
    }
    if (!found) return id;
    Matrix left,right;std::vector<int> yl,yr;
    for (std::size_t i=0;i<y.size();++i) {
      const bool low=x[i][feature]<=cut;(low?left:right).push_back(x[i]);(low?yl:yr).push_back(y[i]);
    }
    tree.nodes[id].leaf=false;tree.nodes[id].feature=feature;tree.nodes[id].threshold=cut;
    const int l=build(left,yl,options,depth+1),r=build(right,yr,options,depth+1);
    tree.nodes[id].left=l;tree.nodes[id].right=r;return id;
  }
};
void compare_tree(const a::DecisionTree& actual,const a::DecisionTree& expected) {
  ASSERT_EQ(actual.nodes.size(),expected.nodes.size());
  for (std::size_t i=0;i<actual.nodes.size();++i) {
    const auto& p=actual.nodes[i];const auto& q=expected.nodes[i];
    ASSERT_EQ(p.leaf,q.leaf);EXPECT_EQ(p.prediction,q.prediction);
    if (!p.leaf) {EXPECT_EQ(p.feature,q.feature);EXPECT_EQ(p.threshold,q.threshold);EXPECT_EQ(p.left,q.left);EXPECT_EQ(p.right,q.right);}
  }
}
void verify(const Matrix& x,const std::vector<int>& y,a::C45Options options={}) {
  const auto trace=a::c45_train_trace(x,y,options);
  Oracle expected;if (!x.empty()) expected.build(x,y,options);
  compare_tree(trace.tree,expected.tree);compare_tree(a::c45_train(x,y,options.max_depth,options.min_samples),expected.tree);
  EXPECT_EQ(trace.candidate_count,expected.candidates);EXPECT_EQ(trace.candidate_row_tests,expected.row_tests);
  ASSERT_EQ(trace.nodes.size(),trace.tree.nodes.size());ASSERT_EQ(trace.candidates.size(),trace.candidate_count);
  std::vector<std::vector<std::size_t>> memberships(trace.nodes.size());
  if (!x.empty()) {memberships[0].resize(x.size());std::iota(memberships[0].begin(),memberships[0].end(),std::size_t{0});}
  std::size_t cursor=0,tests=0;
  for (std::size_t id=0;id<trace.nodes.size();++id) {
    const auto& node=trace.nodes[id];const auto& model=trace.tree.nodes[id];
    EXPECT_EQ(node.rows,memberships[id]);EXPECT_EQ(node.candidate_begin,cursor);
    std::vector<int> labels;for (auto row:node.rows) labels.push_back(y[row]);
    EXPECT_EQ(model.prediction,majority(labels));EXPECT_DOUBLE_EQ(node.entropy,entropy(labels));
    std::size_t selected=0;double maximum=0;
    for (std::size_t j=0;j<node.candidate_count;++j,++cursor) {
      const auto& c=trace.candidates[cursor];EXPECT_EQ(c.node,id);
      EXPECT_LE(c.lower,c.threshold);EXPECT_LT(c.threshold,c.upper);EXPECT_TRUE(std::isfinite(c.threshold));
      std::vector<int> lo,hi;
      for (auto row:node.rows) (x[row][c.feature]<=c.threshold?lo:hi).push_back(y[row]);
      EXPECT_EQ(c.left_count,lo.size());EXPECT_EQ(c.right_count,hi.size());ASSERT_FALSE(lo.empty());ASSERT_FALSE(hi.empty());
      const double p=static_cast<double>(lo.size())/labels.size(),q=static_cast<double>(hi.size())/labels.size();
      EXPECT_DOUBLE_EQ(c.conditional_entropy,p*entropy(lo)+q*entropy(hi));
      EXPECT_DOUBLE_EQ(c.gain,node.entropy-c.conditional_entropy);
      EXPECT_DOUBLE_EQ(c.split_info,-p*std::log2(p)-q*std::log2(q));EXPECT_DOUBLE_EQ(c.ratio,c.gain/c.split_info);
      maximum=std::max(maximum,c.ratio);tests+=labels.size();
      if (c.selected) {++selected;EXPECT_EQ(c.feature,model.feature);EXPECT_EQ(c.threshold,model.threshold);}
    }
    EXPECT_EQ(selected,model.leaf?0u:1u);
    if (!model.leaf) {
      EXPECT_EQ(node.reason,a::C45StopReason::split);
      for (auto row:node.rows) memberships[x[row][model.feature]<=model.threshold?model.left:model.right].push_back(row);
      EXPECT_EQ(trace.nodes[model.left].depth,node.depth+1);EXPECT_EQ(trace.nodes[model.right].depth,node.depth+1);
      for (std::size_t j=node.candidate_begin;j<cursor;++j) if(trace.candidates[j].selected) EXPECT_EQ(trace.candidates[j].ratio,maximum);
    }
  }
  EXPECT_EQ(cursor,trace.candidates.size());EXPECT_EQ(tests,trace.candidate_row_tests);
  for(std::size_t row=0;row<x.size();++row) {
    int id=0;while (!expected.tree.nodes[id].leaf) {const auto& n=expected.tree.nodes[id];id=x[row][n.feature]<=n.threshold?n.left:n.right;}
    EXPECT_EQ(a::c45_predict(trace.tree,x[row]),expected.tree.nodes[id].prediction);
  }
}
}
TEST(C45Trace, HandCalculatedThresholdAndStableTies) {
  const Matrix x{{0,0},{1,1},{2,2},{3,3}};const std::vector<int> y{0,0,1,1};
  const auto trace=a::c45_train_trace(x,y,{1,2,0});ASSERT_EQ(trace.candidates.size(),6u);
  EXPECT_EQ(trace.tree.nodes[0].feature,0u);EXPECT_EQ(trace.tree.nodes[0].threshold,1.5);
  EXPECT_DOUBLE_EQ(trace.candidates[1].gain,1);EXPECT_DOUBLE_EQ(trace.candidates[1].split_info,1);EXPECT_DOUBLE_EQ(trace.candidates[1].ratio,1);
  EXPECT_NEAR(trace.candidates[0].gain,0.31127812445913283,1e-14);
  EXPECT_NEAR(trace.candidates[0].split_info,0.8112781244591328,1e-14);
  EXPECT_NEAR(trace.candidates[0].ratio,0.3836885465963443,1e-14);verify(x,y,{1,2,0});
}
TEST(C45Trace, ExhaustiveSmallOneDimensionalTraining) {
  // Every ordered array of (value in {0,1,2}, label in {-3,4}), n<=5.
  std::size_t checked=0;
  for(std::size_t n=0,power=1;n<=5;++n,power*=6) for(std::size_t code=0;code<power;++code) {
    Matrix x(n,std::vector<double>(1));std::vector<int> y(n);auto digits=code;
    for(std::size_t i=0;i<n;++i) {auto atom=digits%6;digits/=6;x[i][0]=atom/2;y[i]=atom%2?4:-3;}
    for(std::size_t depth:{0u,1u}) for(std::size_t minimum:{1u,3u}) {verify(x,y,{depth,minimum,0});++checked;}
  }
  EXPECT_EQ(checked,37324u);
}
TEST(C45Trace, RandomMultifeatureMulticlassOracle) {
  std::mt19937 rng(405);
  for(int trial=0;trial<1000;++trial) {
    const std::size_t n=rng()%31,d=rng()%5;Matrix x(n,std::vector<double>(d));std::vector<int> y(n);
    for(std::size_t i=0;i<n;++i) {y[i]=static_cast<int>(rng()%4)-2;for(double& value:x[i]) value=static_cast<int>(rng()%9)-4;}
    verify(x,y,{rng()%5,rng()%8,0});
  }
}
TEST(C45Trace, StopReasonsAndMinimumParentNotChild) {
  EXPECT_TRUE(a::c45_train_trace({},{}).tree.nodes.empty());
  EXPECT_EQ(a::c45_train_trace({{},{}},{9,3}).nodes[0].reason,a::C45StopReason::no_positive_gain);
  EXPECT_EQ(a::c45_train_trace({{0},{1}},{7,7}).nodes[0].reason,a::C45StopReason::pure);
  EXPECT_EQ(a::c45_train_trace({{0},{1}},{7,8},{1,3,0}).nodes[0].reason,a::C45StopReason::too_small);
  const Matrix x{{0},{1},{2},{3}};const std::vector<int> y{0,1,0,1};
  const auto limited=a::c45_train_trace(x,y,{1,2,0});
  EXPECT_TRUE(std::any_of(limited.nodes.begin(),limited.nodes.end(),[](const auto& node){return node.reason==a::C45StopReason::depth_limit;}));
  const auto singleton=a::c45_train_trace({{0},{1},{2}},{0,0,1},{0,3,0});
  ASSERT_EQ(singleton.tree.nodes.size(),3u);EXPECT_EQ(singleton.nodes[2].rows.size(),1u);
  verify(x,y,{0,0,0});verify(x,y,{0,1,0});
}
TEST(C45Trace, PureXorAndConflictingDuplicatesStop) {
  const auto xor_tree=a::c45_train_trace({{0,0},{0,1},{1,0},{1,1}},{0,1,1,0});
  EXPECT_EQ(xor_tree.nodes[0].reason,a::C45StopReason::no_positive_gain);EXPECT_EQ(xor_tree.candidate_count,2u);
  EXPECT_EQ(xor_tree.tree.nodes[0].prediction,0);
  const auto duplicate=a::c45_train_trace({{1},{1},{1}},{4,-2,4});
  EXPECT_EQ(duplicate.candidate_count,0u);EXPECT_EQ(a::c45_predict(duplicate.tree,{}),4);
}
TEST(C45Trace, FiniteOverflowAdjacentAndSubnormalBoundaries) {
  const double top=std::numeric_limits<double>::max(),tiny=std::numeric_limits<double>::denorm_min();
  for(auto [lo,hi]:std::vector<std::pair<double,double>>{{top/2,top},{-top,top},{0,tiny},{-tiny,0},{1,std::nextafter(1,2)}}) {
    const auto trace=a::c45_train_trace({{lo},{hi}},{3,7});
    ASSERT_EQ(trace.tree.nodes.size(),3u);const double cut=trace.tree.nodes[0].threshold;
    EXPECT_TRUE(std::isfinite(cut));EXPECT_LE(lo,cut);EXPECT_LT(cut,hi);
    EXPECT_EQ(a::c45_predict(trace.tree,{lo}),3);EXPECT_EQ(a::c45_predict(trace.tree,{hi}),7);
  }
}
TEST(C45Trace, CandidateBudgetCountsZeroGainAndIsWholeCall) {
  const Matrix x{{0},{1},{2},{3}};const std::vector<int> y{0,1,0,1};
  const auto expected=a::c45_train_trace(x,y);ASSERT_GT(expected.candidate_count,3u);
  for(std::size_t cap=1;cap<expected.candidate_count;++cap) EXPECT_THROW(a::c45_train_trace(x,y,{0,2,cap}),std::length_error);
  const auto exact=a::c45_train_trace(x,y,{0,2,expected.candidate_count});compare_tree(exact.tree,expected.tree);
  EXPECT_THROW(a::c45_train_trace({{0,0},{0,1},{1,0},{1,1}},{0,1,1,0},{0,2,1}),std::length_error);
  EXPECT_EQ(a::c45_train_trace({{0},{1}},{7,7},{0,2,1}).candidate_count,0u);
}
TEST(C45Trace, RejectsBadTrainingInputsBeforePurityShortcut) {
  EXPECT_THROW(a::c45_train({{0}},{}),std::invalid_argument);
  EXPECT_THROW(a::c45_train({}, {1}),std::invalid_argument);
  EXPECT_THROW(a::c45_train({{0},{0,1}},{1,1}),std::invalid_argument);
  for(double bad:{std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(a::c45_train({{bad}},{1}),std::invalid_argument);
    EXPECT_THROW(a::c45_train_trace({{bad}},{1}),std::invalid_argument);
  }
}
TEST(C45Trace, PredictionChecksOnlyTheVisitedPath) {
  EXPECT_THROW(a::c45_predict({},{}),std::invalid_argument);
  auto tree=a::c45_train({{0},{1}},{3,7});
  EXPECT_THROW(a::c45_predict(tree,{}),std::invalid_argument);
  EXPECT_THROW(a::c45_predict(tree,{std::numeric_limits<double>::quiet_NaN()}),std::invalid_argument);
  EXPECT_EQ(a::c45_predict(tree,{0,std::numeric_limits<double>::quiet_NaN()}),3);
  tree.nodes[0].right=-1;EXPECT_EQ(a::c45_predict(tree,{0}),3);
  EXPECT_THROW(a::c45_predict(tree,{1}),std::invalid_argument);
  tree.nodes[0].left=0;EXPECT_THROW(a::c45_predict(tree,{0}),std::invalid_argument);
  tree.nodes[0].left=999;EXPECT_THROW(a::c45_predict(tree,{0}),std::invalid_argument);
  tree.nodes[0].threshold=std::numeric_limits<double>::infinity();
  EXPECT_THROW(a::c45_predict(tree,{0}),std::invalid_argument);
  const auto categorical=a::id3_train({{0},{1}},{3,7});
  EXPECT_THROW(a::c45_predict(categorical,{0}),std::invalid_argument);
  auto leaf=a::c45_train({{}},{42});EXPECT_EQ(a::c45_predict(leaf,{}),42);
}
TEST(C45Trace, OwnedRecordsAndDeterministicPreorder) {
  Matrix x{{0},{1},{2},{3}};std::vector<int> y{0,1,0,1};const auto trace=a::c45_train_trace(x,y);
  auto copy=trace;x.clear();y.clear();copy.nodes[0].rows.clear();copy.tree.nodes[0].prediction=999;
  EXPECT_EQ(trace.nodes[0].rows.size(),4u);EXPECT_NE(trace.tree.nodes[0].prediction,999);
  for(std::size_t i=0;i<trace.tree.nodes.size();++i) if(!trace.tree.nodes[i].leaf) {
    EXPECT_EQ(trace.tree.nodes[i].left,static_cast<int>(i+1));EXPECT_GT(trace.tree.nodes[i].right,trace.tree.nodes[i].left);
  }
}

TEST(C45Trace, GaussianGeneratorPreservesOriginalDrawOrder) {
  const auto b=datamunge::datasets::gaussian_blobs({{0,0},{6,0},{3,6}});
  ASSERT_EQ(b.features.size(),180u);ASSERT_EQ(b.labels.size(),180u);
  std::mt19937_64 rng(5);std::normal_distribution<double> noise(0,.6);
  const Matrix centres{{0,0},{6,0},{3,6}};
  for(std::size_t c=0;c<3;++c) for(std::size_t i=0;i<60;++i) {
    EXPECT_EQ(b.labels[c*60+i],static_cast<int>(c));
    EXPECT_EQ(b.features[c*60+i][0],centres[c][0]+noise(rng));
    EXPECT_EQ(b.features[c*60+i][1],centres[c][1]+noise(rng));
  }
  const auto repeat=datamunge::datasets::gaussian_blobs(centres);EXPECT_EQ(b.features,repeat.features);
  auto tree=a::c45_train(b.features,b.labels);std::size_t correct=0;
  for(std::size_t i=0;i<180;++i) correct+=a::c45_predict(tree,b.features[i])==b.labels[i];
  EXPECT_EQ(correct,180u);EXPECT_EQ(tree.nodes.size(),5u);
}
TEST(C45Trace, GaussianGeneratorEmptyExactAndCheckedInputs) {
  using datamunge::datasets::gaussian_blobs;
  EXPECT_TRUE(gaussian_blobs({}).features.empty());EXPECT_TRUE(gaussian_blobs({{1,2}},0).features.empty());
  const auto b=gaussian_blobs({{1,2},{3,4}},2,0);
  EXPECT_EQ(b.features,(Matrix{{1,2},{1,2},{3,4},{3,4}}));EXPECT_EQ(b.labels,(std::vector<int>{0,0,1,1}));
  EXPECT_THROW(gaussian_blobs({{}}),std::invalid_argument);
  EXPECT_THROW(gaussian_blobs({{0},{1,2}}),std::invalid_argument);
  EXPECT_THROW(gaussian_blobs({{0}},0,-1),std::invalid_argument);
  EXPECT_THROW(gaussian_blobs({{std::numeric_limits<double>::quiet_NaN()}},0),std::invalid_argument);
  EXPECT_THROW(gaussian_blobs({{0}},0,std::numeric_limits<double>::infinity()),std::invalid_argument);
  EXPECT_THROW(gaussian_blobs({{0},{1}},std::numeric_limits<std::size_t>::max()),std::length_error);
}
