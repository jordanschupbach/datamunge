#include <gtest/gtest.h>
#include <datamunge/stats/decision_tree_classifier.hpp>
#include <datamunge/datasets/datasets.hpp>

using datamunge::stats::DecisionTreeClassifier;
namespace {
datamunge::dstruct::DataFrame tiny() {
    datamunge::dstruct::DataFrame data;
    data.add_column("x",std::vector<double>{1,2,3,4,5,6});
    data.add_column("label",std::vector<std::string>{"A","A","A","B","B","B"});
    return data;
}
}
TEST(DecisionTreeDiagram, ActualTinySplitAndCompleteNodeBudget) {
    const DecisionTreeClassifier model(tiny(),"label ~ x",{.max_depth=1});
    const auto prediction=model.fitted_classes();
    EXPECT_THROW(model.plot_tree(2),std::length_error);
    const auto svg=model.plot_tree(3).to_svg();
    EXPECT_EQ(svg,model.plot_tree(0).to_svg());
    EXPECT_NE(svg.find("x &lt;= 3.5"),std::string::npos);
    EXPECT_NE(svg.find("n=6; Gini=0.500"),std::string::npos);
    EXPECT_NE(svg.find("n=3; Gini=0.000"),std::string::npos);
    EXPECT_EQ(model.fitted_classes(),prediction);
    EXPECT_EQ(model.node_count(),3);
}
TEST(DecisionTreeDiagram, IrisDiagramOwnsActualModelWithoutRefitting) {
    const DecisionTreeClassifier model(datamunge::datasets::iris(),"Species ~ .",{.max_depth=3});
    const auto before=model.summary();
    const auto picture=model.plot_tree();
    const auto svg=picture.to_svg();
    EXPECT_NE(svg.find("Petal.Length"),std::string::npos);
    EXPECT_NE(svg.find("2.45"),std::string::npos);
    EXPECT_NE(svg.find("Petal.Width"),std::string::npos);
    EXPECT_NE(svg.find("n=150; Gini=0.667"),std::string::npos);
    EXPECT_NE(svg.find("setosa"),std::string::npos);
    EXPECT_EQ(model.summary(),before);
    EXPECT_EQ(model.node_count(),9);
    EXPECT_THROW(model.plot_tree(8),std::length_error);
    auto owned=[] {return DecisionTreeClassifier(tiny(),"label ~ x").plot_tree();}();
    EXPECT_NE(owned.to_svg().find("x &lt;= 3.5"),std::string::npos);
}
TEST(DecisionTreeDiagram, EntropyAndEscapedClassLabels) {
    auto data=tiny();
    data=datamunge::dstruct::DataFrame{};
    data.add_column("x",std::vector<double>{1,2,3,4});
    data.add_column("label",std::vector<std::string>{"<A>","<A>","B & C","B & C"});
    const DecisionTreeClassifier model(data,"label ~ x",{.criterion=datamunge::stats::SplitCriterion::Entropy});
    const auto svg=model.plot_tree().to_svg();
    EXPECT_NE(svg.find("entropy=1.000"),std::string::npos);
    EXPECT_NE(svg.find("&lt;A&gt;"),std::string::npos);
    EXPECT_NE(svg.find("B &amp; C"),std::string::npos);
    EXPECT_EQ(svg.find("<A>"),std::string::npos);
}
TEST(DecisionTreeDiagram, UnsplitMixedRootAndDeepTreeAreBothComplete) {
    auto data=tiny();
    const DecisionTreeClassifier root(data,"label ~ x",{.min_samples_split=100});
    EXPECT_EQ(root.node_count(),1);
    const auto svg=root.plot_tree(1).to_svg();
    EXPECT_NE(svg.find("n=6; Gini=0.500"),std::string::npos);
    const DecisionTreeClassifier full(datamunge::datasets::iris(),"Species ~ .",{.max_depth=30});
    EXPECT_EQ(full.node_count(),17);
    EXPECT_NO_THROW(full.plot_tree(17).to_svg());
    EXPECT_THROW(full.plot_tree(16),std::length_error);
}
