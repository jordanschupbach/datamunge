#include <gtest/gtest.h>

#include <datamunge/datasets/datasets.hpp>
#include <datamunge/stats/mds.hpp>
#include <datamunge/stats/pca.hpp>
#include <limits>
#include <algorithm>

#include <cmath>
#include <stdexcept>

using datamunge::dstruct::DataFrame;
using datamunge::stats::DistanceMetric;
using datamunge::stats::MDS;
using datamunge::stats::MDSOptions;

namespace {

const std::vector<std::string> kIrisFeatures = {"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"};

double euclidean(const DataFrame& df, const std::vector<std::string>& features, const std::size_t i, const std::size_t j) {
  double sum = 0.0;
  for (const auto& f : features) {
    const double d = df.double_at(f, i) - df.double_at(f, j);
    sum += d * d;
  }
  return std::sqrt(sum);
}

} // namespace

TEST(MDS, EmbeddingPreservesPairwiseDistanceOrdering) {
  const auto iris = datamunge::datasets::iris();
  MDS mds(iris, kIrisFeatures);

  EXPECT_EQ(mds.observations(), iris.nrows());
  EXPECT_EQ(mds.n_components(), 2u);
  EXPECT_GT(mds.goodness_of_fit(), 0.9);

  std::vector<double> orig, embedded;
  for (std::size_t i = 0; i < iris.nrows(); i += 3) {
    for (std::size_t j = i + 1; j < iris.nrows(); j += 3) {
      orig.push_back(euclidean(iris, kIrisFeatures, i, j));
      double sum = 0.0;
      for (std::size_t c = 0; c < 2; ++c) {
        const double d = mds.embedding()(i, c) - mds.embedding()(j, c);
        sum += d * d;
      }
      embedded.push_back(std::sqrt(sum));
    }
  }

  const double n = static_cast<double>(orig.size());
  double mean_o = 0.0, mean_e = 0.0;
  for (std::size_t i = 0; i < orig.size(); ++i) {
    mean_o += orig[i];
    mean_e += embedded[i];
  }
  mean_o /= n;
  mean_e /= n;
  double cov = 0.0, var_o = 0.0, var_e = 0.0;
  for (std::size_t i = 0; i < orig.size(); ++i) {
    const double do_ = orig[i] - mean_o;
    const double de_ = embedded[i] - mean_e;
    cov += do_ * de_;
    var_o += do_ * do_;
    var_e += de_ * de_;
  }
  const double correlation = cov / std::sqrt(var_o * var_e);
  EXPECT_GT(correlation, 0.9) << "MDS embedding distances should strongly correlate with original distances";
}

TEST(MDS, EuclideanDistanceMdsClosesMatchesPcaVarianceStructure) {
  const auto iris = datamunge::datasets::iris();
  // Classical MDS on raw Euclidean distances is mathematically closely related to unscaled PCA
  // -- both derive from eigendecomposing a Gram/covariance matrix built from the same centered
  // data -- so their leading-eigenvalue proportions should be in the same ballpark.
  MDS mds(iris, kIrisFeatures, MDSOptions{2, DistanceMetric::Euclidean});
  EXPECT_GT(mds.eigenvalues()[0], mds.eigenvalues()[1]);
  EXPECT_GT(mds.eigenvalues()[1], 0.0);
}

TEST(MDS, ManhattanMetricProducesADifferentEmbeddingThanEuclidean) {
  const auto iris = datamunge::datasets::iris();
  MDS euclidean_mds(iris, kIrisFeatures, MDSOptions{2, DistanceMetric::Euclidean});
  MDS manhattan_mds(iris, kIrisFeatures, MDSOptions{2, DistanceMetric::Manhattan});
  EXPECT_NE(euclidean_mds.eigenvalues()[0], manhattan_mds.eigenvalues()[0]);
}

TEST(MDS, DropsRowsWithNullFeatures) {
  DataFrame frame;
  frame.add_column("x", std::vector<std::optional<double>>{1.0, 2.0, std::nullopt, 4.0, 5.0, 6.0});
  frame.add_column("y", std::vector<std::optional<double>>{2.0, 4.0, 6.0, 8.0, 9.0, 3.0});

  MDS mds(frame, {"x", "y"}, MDSOptions{1, DistanceMetric::Euclidean});
  EXPECT_EQ(mds.observations(), 5u);
  EXPECT_EQ(mds.kept_row_indices(), (std::vector<std::size_t>{0, 1, 3, 4, 5}));
}

TEST(MDS, RejectsInvalidOptionsAndTooFewRows) {
  const auto iris = datamunge::datasets::iris();
  EXPECT_THROW(MDS(iris, {"NotAColumn"}), std::invalid_argument);
  EXPECT_THROW(MDS(iris, {"Species"}), std::invalid_argument);
  EXPECT_THROW(MDS(iris, kIrisFeatures, MDSOptions{0, DistanceMetric::Euclidean}), std::invalid_argument);

  DataFrame tiny;
  tiny.add_column("x", std::vector<double>{1.0, 2.0});
  EXPECT_THROW(MDS(tiny, {"x"}, MDSOptions{2, DistanceMetric::Euclidean}), std::invalid_argument);
}

TEST(MDS, PlotEmbeddingGroupedMatchesUngrouped) {
  const auto iris = datamunge::datasets::iris();
  MDS mds(iris, kIrisFeatures);

  std::vector<std::string> species(mds.observations());
  for (std::size_t i = 0; i < mds.observations(); ++i) species[i] = iris.string_at("Species", mds.kept_row_indices()[i]);

  const auto grouped = mds.plot_embedding(species);
  EXPECT_EQ(grouped.series().size(), 3u);

  std::size_t total_points = 0;
  for (const auto& series : grouped.series()) total_points += series.x.size();
  EXPECT_EQ(total_points, mds.observations());

  const auto ungrouped = mds.plot_embedding();
  EXPECT_EQ(ungrouped.series().size(), 1u);

  EXPECT_THROW((void)mds.plot_embedding(std::vector<std::string>{"only-one-label"}), std::invalid_argument);
  EXPECT_THROW((void)mds.plot_embedding(10, 0), std::out_of_range);
}

namespace {

datamunge::linalg::DenseMatrix<double> cycle_distances(double diagonal) {
  return {4,4,{0,1,diagonal,1, 1,0,1,diagonal, diagonal,1,0,1, 1,diagonal,1,0}};
}

} // namespace

TEST(MDS, RectangleExposesDoubleCentringAndExactRankTwoReconstruction) {
  DataFrame rectangle;
  rectangle.add_column("x",std::vector<double>{0,4,4,0});
  rectangle.add_column("y",std::vector<double>{0,0,3,3});
  const MDS model(rectangle,{"x","y"});
  EXPECT_NEAR(model.eigenvalues()[0],16,1e-10);
  EXPECT_NEAR(model.eigenvalues()[1],9,1e-10);
  const double first_row[]={6.25,-1.75,-6.25,1.75};
  for (std::size_t j=0;j<4;++j) EXPECT_NEAR(model.gram_matrix()(0,j),first_row[j],1e-10);
  const auto reconstructed=model.embedding()*model.embedding().transpose();
  EXPECT_LT((model.gram_matrix()-reconstructed).norm_frobenius(),1e-9);
  const auto path=model.dimension_diagnostics();
  EXPECT_NEAR(path.double_at("strain_squared",0),81,1e-9);
  EXPECT_NEAR(path.double_at("relative_strain",0),9/std::sqrt(337.0),1e-10);
  EXPECT_NEAR(path.double_at("gof_positive",0),16.0/25,1e-10);
  EXPECT_LT(path.double_at("relative_distance_error",1),1e-10);
  const auto precomputed=MDS::from_distances(model.dissimilarities());
  EXPECT_TRUE(precomputed.feature_names().empty());
  EXPECT_EQ(precomputed.kept_row_indices(),(std::vector<std::size_t>{0,1,2,3}));
  EXPECT_LT((precomputed.gram_matrix()-model.gram_matrix()).norm_frobenius(),1e-10);
}

TEST(MDS, NonEuclideanCycleHasPerfectPositiveGofButNonzeroStress) {
  const auto model=MDS::from_distances(cycle_distances(2),3);
  const auto path=model.dimension_diagnostics(), spectrum=model.spectrum();
  EXPECT_NEAR(model.eigenvalues()[0],2,1e-10);
  EXPECT_NEAR(model.eigenvalues()[1],2,1e-10);
  EXPECT_NEAR(model.eigenvalues()[3],-1,1e-10);
  EXPECT_NEAR(path.double_at("gof_positive",1),1,1e-10);
  EXPECT_NEAR(path.double_at("gof_absolute",1),0.8,1e-10);
  EXPECT_NEAR(path.double_at("strain_squared",1),1,1e-10);
  EXPECT_NEAR(path.double_at("relative_strain",1),1.0/3,1e-10);
  EXPECT_NEAR(path.double_at("relative_distance_error",1),(std::sqrt(2.0)-1)/std::sqrt(3.0),1e-10);
  EXPECT_NEAR(path.double_at("relative_distance_error",2),path.double_at("relative_distance_error",1),1e-10);
  EXPECT_DOUBLE_EQ(spectrum.double_at("significant_negative",3),1);
  // Classical scaling is not a distance-loss minimiser: shrinking this square improves stress.
  const double classical_sse=12*std::pow(path.double_at("relative_distance_error",1),2);
  const double side=(1+std::sqrt(2.0))/2;
  const double alternative_sse=4*std::pow(side-1,2)+2*std::pow(std::sqrt(2.0)*side-2,2);
  EXPECT_LT(alternative_sse,classical_sse);
}

TEST(MDS, CycleSpectrumMatchesClosedFormAcrossEuclideanBoundary) {
  for (double d : {1.0,1.2,std::sqrt(2.0),1.7,2.0}) {
    const auto model=MDS::from_distances(cycle_distances(d),3);
    std::vector<double> expected{d*d/2,d*d/2,1-d*d/2,0};
    std::sort(expected.begin(),expected.end(),std::greater<double>());
    for (std::size_t i=0;i<4;++i) EXPECT_NEAR(model.eigenvalues()[i],expected[i],1e-10);
    const auto path=model.dimension_diagnostics();
    EXPECT_NEAR(path.double_at("strain_squared",2),std::pow(std::min(0.0,1-d*d/2),2),1e-10);
  }
}

TEST(MDS, DistanceAndStrainDiagnosticsMatchDirectMatrixAndPairCalculations) {
  const MDS model(datamunge::datasets::iris(),kIrisFeatures,MDSOptions{4,DistanceMetric::Euclidean});
  const auto path=model.dimension_diagnostics();
  for (std::size_t k=1;k<=4;++k) {
    const auto pairs=model.pair_distances(k);
    double numerator=0,denominator=0;
    for (std::size_t i=0;i<pairs.nrows();++i) {
      numerator += std::pow(pairs.double_at("residual",i),2);
      denominator += std::pow(pairs.double_at("dissimilarity",i),2);
    }
    EXPECT_NEAR(path.double_at("relative_distance_error",k-1),std::sqrt(numerator/denominator),1e-12);
    EXPECT_NEAR(path.double_at("distance_rmse",k-1),std::sqrt(numerator/pairs.nrows()),1e-12);
    datamunge::linalg::DenseMatrix<double> coordinates(150,k,0.0);
    for (std::size_t i=0;i<150;++i) for (std::size_t j=0;j<k;++j) coordinates(i,j)=model.embedding()(i,j);
    const double actual=(model.gram_matrix()-coordinates*coordinates.transpose()).norm_frobenius();
    EXPECT_NEAR(path.double_at("strain_squared",k-1),actual*actual,1e-7);
  }
}

TEST(MDS, UnscaledPcaAndEuclideanMdsHaveTheSameGramMatrix) {
  const auto iris=datamunge::datasets::iris();
  const MDS model(iris,kIrisFeatures,MDSOptions{4,DistanceMetric::Euclidean});
  const datamunge::stats::PCA pca(iris,kIrisFeatures,{true,false});
  for (std::size_t k=0;k<4;++k)
    EXPECT_NEAR(model.eigenvalues()[k],149*pca.explained_variance()[k],1e-8);
  const auto mds_gram=model.embedding()*model.embedding().transpose();
  const auto pca_gram=pca.scores()*pca.scores().transpose();
  EXPECT_LT((mds_gram-pca_gram).norm_frobenius(),1e-8);
}

TEST(MDS, ScalingDistancesHasThePredictedUnits) {
  const auto base=MDS::from_distances(cycle_distances(2),2);
  const auto ref=base.dimension_diagnostics();
  for (double scale : {1e-4,0.1,10.0,1e4}) {
    const auto model=MDS::from_distances(cycle_distances(2)*scale,2);
    const auto path=model.dimension_diagnostics();
    for (std::size_t i=0;i<4;++i) EXPECT_NEAR(model.eigenvalues()[i]/(scale*scale),base.eigenvalues()[i],1e-9);
    EXPECT_NEAR(path.double_at("relative_distance_error",1),ref.double_at("relative_distance_error",1),1e-10);
    EXPECT_NEAR(path.double_at("distance_rmse",1)/scale,ref.double_at("distance_rmse",1),1e-10);
    EXPECT_NEAR(path.double_at("strain_squared",1)/std::pow(scale,4),1,1e-9);
  }
}

TEST(MDS, PairIndicesRetainOriginalRowNumbersAfterNullDropping) {
  DataFrame data;
  data.add_column("x",std::vector<std::optional<double>>{0,std::nullopt,1,2});
  const MDS model(data,{"x"},{1});
  const auto pairs=model.pair_distances();
  EXPECT_EQ(pairs.nrows(),3u);
  EXPECT_DOUBLE_EQ(pairs.double_at("row_i",0),0);
  EXPECT_DOUBLE_EQ(pairs.double_at("row_j",0),2);
  EXPECT_DOUBLE_EQ(pairs.double_at("row_i",2),2);
  EXPECT_DOUBLE_EQ(pairs.double_at("row_j",2),3);
}

TEST(MDS, DegenerateZeroDistancesHaveFiniteZeroDiagnostics) {
  const auto model=MDS::from_distances(datamunge::linalg::DenseMatrix<double>(4,4,0.0),3);
  const auto path=model.dimension_diagnostics();
  for (std::size_t i=0;i<3;++i) {
    EXPECT_DOUBLE_EQ(path.double_at("gof_positive",i),0);
    EXPECT_DOUBLE_EQ(path.double_at("relative_strain",i),0);
    EXPECT_DOUBLE_EQ(path.double_at("relative_distance_error",i),0);
  }
  EXPECT_NO_THROW((void)model.plot_shepard());
  EXPECT_THROW((void)model.pair_distances(4),std::out_of_range);
}

TEST(MDS, RejectsInvalidDissimilarityMatricesAndNonfiniteFeatures) {
  using Matrix=datamunge::linalg::DenseMatrix<double>;
  EXPECT_THROW((void)MDS::from_distances(Matrix(2,3,0.0),1),std::invalid_argument);
  EXPECT_THROW((void)MDS::from_distances(Matrix(1,1,0.0),1),std::invalid_argument);
  for (int issue=0;issue<4;++issue) {
    auto bad=cycle_distances(2);
    if (issue==0) bad(0,0)=1;
    if (issue==1) bad(0,1)=-1;
    if (issue==2) bad(0,1)=0;
    if (issue==3) bad(0,1)=std::numeric_limits<double>::infinity();
    EXPECT_THROW((void)MDS::from_distances(bad),std::invalid_argument);
  }
  EXPECT_THROW((void)MDS::from_distances(cycle_distances(2),0),std::invalid_argument);
  EXPECT_THROW((void)MDS::from_distances(cycle_distances(2),4),std::invalid_argument);
  DataFrame bad;
  bad.add_column("x",std::vector<double>{0,1,std::numeric_limits<double>::quiet_NaN()});
  EXPECT_THROW(MDS(bad,{"x"}),std::invalid_argument);
}

TEST(MDS, SmallAsymmetriesAreAveragedWithoutMutatingInput) {
  auto distances=cycle_distances(2);
  distances(0,1)+=1e-13; distances(0,0)=1e-13;
  const auto model=MDS::from_distances(distances);
  EXPECT_DOUBLE_EQ(model.dissimilarities()(0,0),0);
  EXPECT_DOUBLE_EQ(model.dissimilarities()(0,1),model.dissimilarities()(1,0));
  EXPECT_DOUBLE_EQ(distances(0,0),1e-13);
}
