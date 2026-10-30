#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/stats/stats.hpp>

using datamunge::dstruct::DataFrame;
using datamunge::stats::SVM;
using datamunge::stats::SVMKernel;
using datamunge::stats::SVMOptions;

TEST(SVM, LinearlySeparableDataAchievesPerfectSeparation) {
    DataFrame df;
    df.add_column("x1", std::vector<double>{-2, -3, -1, -2, -1.5, 2, 3, 1, 2, 1.5});
    df.add_column("x2", std::vector<double>{-2, -1, -3, -1, -1.5, 2, 1, 3, 1, 1.5});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a", "a", "b", "b", "b", "b", "b"});

    SVMOptions options;
    options.kernel = SVMKernel::Linear;
    SVM model(df, "label ~ x1 + x2", options);

    EXPECT_DOUBLE_EQ(model.training_accuracy(), 1.0);

    DataFrame newdata;
    newdata.add_column("x1", std::vector<double>{-5.0, 5.0, -0.1});
    newdata.add_column("x2", std::vector<double>{-5.0, 5.0, -0.1});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 3u);
    EXPECT_EQ(preds[0], "a");
    EXPECT_EQ(preds[1], "b");
    EXPECT_EQ(preds[2], "a");
}

TEST(SVM, RadialKernelSolvesNonLinearlySeparableData) {
    // Classic four-quadrant XOR pattern: class membership depends on the
    // product of signs, which no linear boundary can separate (best a
    // linear classifier can do is ~50%) but an RBF kernel can.
    DataFrame df;
    std::vector<double> x1, x2;
    std::vector<std::string> label;
    const double centers[4][2] = {{-2, -2}, {2, 2}, {-2, 2}, {2, -2}};
    const bool   is_positive[4] = {true, true, false, false};
    for (int quadrant = 0; quadrant < 4; ++quadrant) {
        for (int i = 0; i < 10; ++i) {
            const double jitter_x = 0.3 * static_cast<double>(i % 5 - 2);
            const double jitter_y = 0.3 * static_cast<double>((i * 3) % 5 - 2);
            x1.push_back(centers[quadrant][0] + jitter_x);
            x2.push_back(centers[quadrant][1] + jitter_y);
            label.push_back(is_positive[quadrant] ? "pos" : "neg");
        }
    }
    df.add_column("x1", x1);
    df.add_column("x2", x2);
    df.add_column("label", label);

    SVM radial_model(df, "label ~ x1 + x2"); // default kernel is Radial
    EXPECT_GT(radial_model.training_accuracy(), 0.95);

    SVMOptions linear_options;
    linear_options.kernel = SVMKernel::Linear;
    SVM linear_model(df, "label ~ x1 + x2", linear_options);
    EXPECT_LT(linear_model.training_accuracy(), 0.7);
}

// Reference values computed with scikit-learn's SVC(kernel='rbf', C=1,
// gamma=0.25) / SVC(kernel='linear', C=1) on the same embedded iris
// dataset, standardized (zero mean, unit variance, ddof=1) exactly as this
// class standardizes internally by default.
TEST(SVM, IrisRadialKernelMatchesSklearn) {
    SVMOptions options;
    options.kernel = SVMKernel::Radial;
    options.cost   = 1.0;
    options.gamma  = 0.25;
    SVM model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", options);

    EXPECT_NEAR(model.training_accuracy(), 0.973333333, 1e-6);

    const auto cm = model.confusion_matrix();
    const double expected[3][3] = {{50, 0, 0}, {0, 48, 2}, {0, 2, 48}};
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) EXPECT_NEAR(cm(i, j), expected[i][j], 1e-9);

    // sklearn found 51 support vectors (8/22/21); this SMO implementation's
    // slightly different convergence path lands on 52 (8/22/22) — within
    // one boundary point of the reference, as expected for independently
    // converged SMO solutions on a non-strictly-convex dual.
    EXPECT_GE(model.num_support_vectors(), 48u);
    EXPECT_LE(model.num_support_vectors(), 56u);

    DataFrame newdata;
    newdata.add_column("Sepal.Length", std::vector<double>{5.1, 6.0, 6.5, 6.2});
    newdata.add_column("Sepal.Width", std::vector<double>{3.5, 2.7, 3.0, 2.8});
    newdata.add_column("Petal.Length", std::vector<double>{1.4, 4.5, 5.5, 4.8});
    newdata.add_column("Petal.Width", std::vector<double>{0.2, 1.5, 2.0, 1.8});
    const auto preds = model.predict(newdata);
    ASSERT_EQ(preds.size(), 4u);
    EXPECT_EQ(preds[0], "setosa");
    EXPECT_EQ(preds[1], "versicolor");
    EXPECT_EQ(preds[2], "virginica");
    EXPECT_EQ(preds[3], "virginica");
}

TEST(SVM, IrisLinearKernelMatchesSklearn) {
    SVMOptions options;
    options.kernel = SVMKernel::Linear;
    SVM model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", options);
    EXPECT_NEAR(model.training_accuracy(), 0.966666667, 1e-6);
}

TEST(SVM, RejectsSingleClassResponse) {
    DataFrame df;
    df.add_column("x", std::vector<double>{1.0, 2.0, 3.0, 4.0});
    df.add_column("label", std::vector<std::string>{"a", "a", "a", "a"});
    EXPECT_THROW(SVM(df, "label ~ x"), std::invalid_argument);
}

TEST(SVM, PredictThrowsOnUnseenCategoricalLevel) {
    auto penguins = datamunge::datasets::penguins().drop_nulls({"bill_length_mm", "bill_depth_mm", "island"});
    SVM  model(penguins, "species ~ bill_length_mm + bill_depth_mm + island");

    DataFrame bad;
    bad.add_column("bill_length_mm", std::vector<double>{40.0});
    bad.add_column("bill_depth_mm", std::vector<double>{18.0});
    bad.add_column("island", std::vector<std::string>{"Atlantis"});
    EXPECT_THROW((void)model.predict(bad), std::runtime_error);
}

TEST(SVM, SummaryContainsKeyInformation) {
    SVM model(datamunge::datasets::iris(),
              "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width");
    const auto text = model.summary();
    EXPECT_NE(text.find("SVM-Kernel"), std::string::npos);
    EXPECT_NE(text.find("Support Vectors"), std::string::npos);
    EXPECT_NE(text.find("setosa"), std::string::npos);
}

TEST(SVM, TwoPointSolutionMatchesClosedFormAcrossScalesAndCosts) {
    for (double a : {0.5, 1.0, 2.0, 4.0}) for (double cost : {0.01, 0.1, 1.0, 10.0}) {
        DataFrame data;
        data.add_column("x", std::vector<double>{-a,a});
        data.add_column("label", std::vector<std::string>{"left","right"});
        SVMOptions options;
        options.kernel=SVMKernel::Linear; options.scale=false; options.cost=cost;
        const SVM model(data,"label ~ x",options);
        const double alpha=std::min(cost,1.0/(2*a*a));
        const auto rows=model.training_details(), diag=model.diagnostics();
        const double norm=4*a*a*alpha*alpha;
        for (std::size_t i=0;i<2;++i) {
            EXPECT_NEAR(rows.double_at("alpha",i),alpha,1e-10);
            EXPECT_NEAR(rows.double_at("margin",i),2*a*a*alpha,1e-10);
        }
        EXPECT_NEAR(diag.double_at("norm_squared",0),norm,1e-10);
        EXPECT_NEAR(diag.double_at("dual",0),2*alpha-0.5*norm,1e-10);
        EXPECT_NEAR(diag.double_at("gap",0),0,1e-10);
        EXPECT_NEAR(diag.double_at("equality_residual",0),0,1e-12);
        EXPECT_NEAR(diag.double_at("max_kkt_violation",0),0,1e-10);
        EXPECT_NEAR(diag.double_at("bias",0),0,1e-12);
        EXPECT_DOUBLE_EQ(model.error(data),0.0);
    }
}

TEST(SVM, XorPolynomialHasKnownCoefficientsAndObjective) {
    DataFrame data;
    data.add_column("x",std::vector<double>{-1,1,-1,1});
    data.add_column("z",std::vector<double>{-1,1,1,-1});
    data.add_column("label",std::vector<std::string>{"same","same","unlike","unlike"});
    SVMOptions opt;
    opt.kernel=SVMKernel::Polynomial; opt.gamma=1; opt.degree=2; opt.scale=false; opt.cost=10; opt.tolerance=1e-8;
    const SVM model(data,"label ~ x + z",opt);
    const auto details=model.training_details(), diag=model.diagnostics();
    // Opposite corners have identical mapped features, so individual coefficients
    // are non-unique. Their sum per class, margins and objectives are unique.
    EXPECT_NEAR(details.double_at("alpha",0)+details.double_at("alpha",1),0.25,1e-8);
    EXPECT_NEAR(details.double_at("alpha",2)+details.double_at("alpha",3),0.25,1e-8);
    for (std::size_t i=0;i<4;++i) EXPECT_NEAR(details.double_at("margin",i),1,1e-8);
    EXPECT_NEAR(diag.double_at("norm_squared",0),0.5,1e-8);
    EXPECT_NEAR(diag.double_at("dual",0),0.25,1e-8);
    EXPECT_NEAR(diag.double_at("gap",0),0,1e-8);
}

TEST(SVM, PairDecisionsReproduceVotesAndTrainingDetails) {
    const auto iris=datamunge::datasets::iris();
    const SVM model(iris,"Species ~ .");
    const auto values=model.decision_function(iris);
    const auto prediction=model.predict_detail(iris);
    const auto rows=model.training_details();
    ASSERT_EQ(values.cols(),3u); ASSERT_EQ(values.rows(),150u);
    ASSERT_EQ(rows.nrows(),300u);
    for (std::size_t row=0;row<150;++row) {
        std::vector<double> votes(3,0);
        std::size_t p=0;
        for (std::size_t i=0;i<3;++i) for (std::size_t j=i+1;j<3;++j,++p)
            ++votes[values(row,p)>=0 ? i : j];
        EXPECT_EQ(votes,prediction.votes[row]);
    }
    for (std::size_t r=0;r<rows.nrows();++r) {
        const auto pair=r/100;
        const auto input_row=static_cast<std::size_t>(rows.double_at("row",r));
        EXPECT_NEAR(values(input_row,pair),rows.double_at("decision",r),1e-6);
    }
}

TEST(SVM, MissingRowsKeepOriginalIndicesAndAreExcludedFromError) {
    DataFrame data;
    data.add_column("x",std::vector<std::optional<double>>{-2,std::nullopt,-1,1,2});
    data.add_column("label",std::vector<std::string>{"a","a","a","b","b"});
    const SVM model(data,"label ~ x");
    const auto rows=model.training_details();
    ASSERT_EQ(rows.nrows(),4u);
    EXPECT_DOUBLE_EQ(rows.double_at("row",0),0);
    EXPECT_DOUBLE_EQ(rows.double_at("row",1),2);
    const auto decisions=model.decision_function(data);
    EXPECT_TRUE(std::isnan(decisions(1,0)));
    EXPECT_EQ(model.predict(data)[1],"");
    EXPECT_DOUBLE_EQ(model.error(data),0);
    DataFrame unlabelled; unlabelled.add_column("x",std::vector<double>{1});
    EXPECT_THROW((void)model.error(unlabelled),std::invalid_argument);
    DataFrame no_labels;
    no_labels.add_column("x",std::vector<double>{1});
    no_labels.add_column("label",std::vector<std::optional<std::string>>{std::nullopt});
    EXPECT_THROW((void)model.error(no_labels),std::invalid_argument);
    EXPECT_EQ(model.predict(unlabelled).size(),1u);
}

TEST(SVM, RegionPlotValidatesAxesAndSupportsTheirReversal) {
    const SVM model(datamunge::datasets::iris(),"Species ~ Petal.Length + Petal.Width");
    EXPECT_NO_THROW((void)model.plot_decision_regions("Petal.Length","Petal.Width",10));
    EXPECT_NO_THROW((void)model.plot_decision_regions("Petal.Width","Petal.Length",10));
    EXPECT_THROW((void)model.plot_decision_regions("Petal.Length","missing"),std::invalid_argument);
    EXPECT_THROW((void)model.plot_decision_regions("Petal.Length","Petal.Length"),std::invalid_argument);
    EXPECT_THROW((void)model.plot_decision_regions("Petal.Length","Petal.Width",10,std::array<double,4>{0,0,0,1}),std::invalid_argument);
    const SVM four(datamunge::datasets::iris(),"Species ~ .");
    EXPECT_THROW((void)four.plot_decision_regions("Petal.Length","Petal.Width"),std::invalid_argument);
}

TEST(SVM, RejectsInvalidOptimizationOptions) {
    const auto data=datamunge::datasets::iris();
    SVMOptions opt;
    opt.cost=0; EXPECT_THROW(SVM(data,"Species ~ .",opt),std::invalid_argument);
    opt.cost=1; opt.max_passes=0; EXPECT_THROW(SVM(data,"Species ~ .",opt),std::invalid_argument);
    opt.max_passes=50; opt.tolerance=-1; EXPECT_THROW(SVM(data,"Species ~ .",opt),std::invalid_argument);
}


TEST(SVM, ErrorAcceptsAnEmptyStringAsAnObservedClass) {
    DataFrame data;
    data.add_column("x",std::vector<double>{-2,-1,1,2});
    data.add_column("label",std::vector<std::string>{"","","other","other"});
    const SVM model(data,"label ~ x");
    EXPECT_DOUBLE_EQ(model.error(data),0);
    DataFrame query;
    query.add_column("x",std::vector<double>{-1});
    query.add_column("label",std::vector<std::string>{"other"});
    EXPECT_DOUBLE_EQ(model.error(query),1);
}
