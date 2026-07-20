%module(directors="1") datamunge

%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%ignore datamunge::dstruct::NullableColumn;
%ignore datamunge::dstruct::DataFrame;
%ignore datamunge::stats::LM;
%ignore datamunge::stats::Formula;
%ignore datamunge::stats::DesignInfo;
%ignore datamunge::stats::ResolvedColumn;
%ignore datamunge::stats::ResolvedColumnKind;
%ignore datamunge::stats::ExprNode;
%ignore datamunge::stats::ExprOp;
%ignore datamunge::stats::TermGroup;
%ignore datamunge::stats::DesignMatrixData;
%ignore datamunge::stats::LmOptions;
%ignore datamunge::stats::LmPrediction;
%ignore datamunge::stats::PredictionInterval;
%ignore datamunge::stats::AnovaRow;
%ignore datamunge::stats::LMM;
%ignore datamunge::stats::LMMOptions;
%ignore datamunge::stats::GLMM;
%ignore datamunge::stats::GLMMOptions;
%ignore datamunge::stats::GLMMFamily;
%ignore datamunge::stats::LDA;
%ignore datamunge::stats::LDAOptions;
%ignore datamunge::stats::LDAPrediction;
%ignore datamunge::datasets::iris;
%ignore datamunge::datasets::penguins;
%ignore datamunge::stats::SVM;
%ignore datamunge::stats::SVMOptions;
%ignore datamunge::stats::SVMPrediction;
%ignore datamunge::stats::SVMKernel;
%ignore datamunge::stats::DecisionTreeClassifier;
%ignore datamunge::stats::DecisionTreeClassifierOptions;
%ignore datamunge::stats::DecisionTreeClassifierPrediction;
%ignore datamunge::stats::DecisionTreeRegressor;
%ignore datamunge::stats::DecisionTreeRegressorOptions;
%ignore datamunge::stats::SplitCriterion;
%ignore datamunge::stats::RandomForestClassifier;
%ignore datamunge::stats::RandomForestClassifierOptions;
%ignore datamunge::stats::RandomForestClassifierPrediction;
%ignore datamunge::stats::RandomForestRegressor;
%ignore datamunge::stats::RandomForestRegressorOptions;
%ignore datamunge::stats::ElasticNet;
%ignore datamunge::stats::ElasticNetOptions;
%ignore datamunge::stats::Ridge;
%ignore datamunge::stats::RidgeOptions;
%ignore datamunge::stats::Lasso;
%ignore datamunge::stats::LassoOptions;
%ignore datamunge::stats::KNNClassifier;
%ignore datamunge::stats::KNNClassifierOptions;
%ignore datamunge::stats::KNNClassifierPrediction;
%ignore datamunge::stats::KNNRegressor;
%ignore datamunge::stats::KNNRegressorOptions;
%ignore datamunge::stats::KMeans;
%ignore datamunge::stats::KMeansOptions;
%ignore datamunge::stats::AgglomerativeClustering;
%ignore datamunge::stats::AgglomerativeClusteringOptions;
%ignore datamunge::stats::LinkageCriterion;
%ignore datamunge::stats::DBSCAN;
%ignore datamunge::stats::DBSCANOptions;
%ignore datamunge::stats::DistanceMetric;
%ignore datamunge::stats::GBMClassifier;
%ignore datamunge::stats::GBMClassifierOptions;
%ignore datamunge::stats::GBMClassifierPrediction;
%ignore datamunge::stats::GBMRegressor;
%ignore datamunge::stats::GBMRegressorOptions;
%ignore datamunge::stats::XGBoostClassifier;
%ignore datamunge::stats::XGBoostClassifierOptions;
%ignore datamunge::stats::XGBoostClassifierPrediction;
%ignore datamunge::stats::XGBoostRegressor;
%ignore datamunge::stats::XGBoostRegressorOptions;
%ignore datamunge::stats::KernelRegression;
%ignore datamunge::stats::KernelRegressionOptions;
%ignore datamunge::stats::KernelRegressionKernel;
%ignore datamunge::stats::GaussianProcessRegression;
%ignore datamunge::stats::GaussianProcessRegressionOptions;
%ignore datamunge::stats::GaussianProcessRegressionPrediction;
%ignore datamunge::stats::NaiveBayesClassifier;
%ignore datamunge::stats::NaiveBayesClassifierOptions;
%ignore datamunge::stats::NaiveBayesClassifierPrediction;
%ignore datamunge::stats::GLM;
%ignore datamunge::stats::GLMOptions;
%ignore datamunge::stats::GLMPrediction;
%ignore datamunge::stats::GLMFamily;
%ignore datamunge::stats::GLMPredictionInterval;
%ignore datamunge::linalg::Tensor;
%ignore datamunge::linalg::TensorDType;
%ignore datamunge::autodiff::Dual;
%ignore datamunge::autodiff::HyperDual;
%ignore datamunge::autodiff::Tape;
%ignore datamunge::autodiff::Var;
%ignore datamunge::bayes::AutodiffModel;
%ignore datamunge::bayes::normal_lpdf;
%ignore datamunge::bayes::cauchy_lpdf;
%ignore datamunge::bayes::exponential_lpdf;
%ignore datamunge::bayes::gamma_lpdf;
%ignore datamunge::bayes::beta_lpdf;
%ignore datamunge::bayes::student_t_lpdf;
%ignore datamunge::bayes::bernoulli_logit_lpmf;
%ignore datamunge::bayes::poisson_log_lpmf;
%template(IPair) std::pair<int, int>;
%template(DPair) std::pair<double, double>;
%template(DVectorPair) std::pair<std::vector<double>, std::vector<double> >;
%template(SPair) std::pair<std::string, std::string>;
%template(IVector) std::vector<int>;
%template(DVector) std::vector<double>;
%template(DVectorVector) std::vector<std::vector<double> >;
%template(SizeVector) std::vector<size_t>;
%apply std::vector<size_t> { std::vector<std::size_t> };
%apply std::vector<size_t>& { std::vector<std::size_t>& };
%apply const std::vector<size_t>& { const std::vector<std::size_t>& };
%template(SVector) std::vector<std::string>;

%feature("director") datamunge::Callback;
%feature("director") datamunge::optim::ArbitraryFunction;
%feature("director") datamunge::optim::DifferentiableFunction;
%feature("director") datamunge::optim::SeparableFunction;
%feature("director") datamunge::optim::DifferentiableSeparableFunction;

%{
  #include "datamunge/datamunge.hpp"
%}
%include "datamunge/datamunge.hpp"
%include "datamunge/plot/plot.hpp"
%include "datamunge/stats/arima.hpp"
%include "datamunge/stats/exponential_smoothing.hpp"
%include "datamunge/stats/hypothesis_test_result.hpp"
%include "datamunge/stats/t_test.hpp"
%include "datamunge/stats/wilcoxon_test.hpp"
%include "datamunge/stats/ks_test.hpp"
%include "datamunge/stats/chi_squared_test.hpp"
%include "datamunge/stats/anova_test.hpp"
%include "datamunge/stats/correlation_test.hpp"
%include "datamunge/stats/variance_test.hpp"
%include "datamunge/stats/proportion_test.hpp"
%include "datamunge/stats/fisher_exact_test.hpp"
%include "datamunge/stats/normality_test.hpp"
%include "datamunge/stats/p_adjust.hpp"
%include "datamunge/optim/function_types.hpp"
%include "datamunge/optim/gradient_descent.hpp"
%include "datamunge/optim/adam.hpp"
%include "datamunge/optim/lbfgs.hpp"
%include "datamunge/optim/sgd.hpp"
%include "datamunge/optim/simulated_annealing.hpp"
%include "datamunge/optim/pso.hpp"
%include "datamunge/optim/differential_evolution.hpp"
%include "datamunge/optim/genetic_algorithm.hpp"
%include "datamunge/bayes/map.hpp"
%include "datamunge/bayes/hmc.hpp"
%include "datamunge/bayes/nuts.hpp"
%include "datamunge/bayes/rwm.hpp"
%include "datamunge/bayes/gibbs.hpp"
%include "datamunge/bayes/importance_sampling.hpp"

%include "datamunge/geometry/point2d.hpp"
%template(Point2DVector) std::vector<datamunge::geometry::Point2D>;
%include "datamunge/geometry/convex_hull.hpp"
%include "datamunge/geometry/closest_pair.hpp"
%include "datamunge/geometry/polygon.hpp"
%include "datamunge/geometry/segment_intersection.hpp"
%include "datamunge/geometry/kdtree.hpp"

%template(TriangleVector) std::vector<datamunge::geometry::Triangle>;
%include "datamunge/geometry/delaunay.hpp"

%template(Point2DVectorVector) std::vector<std::vector<datamunge::geometry::Point2D> >;
%include "datamunge/geometry/voronoi.hpp"

%include "datamunge/geometry/dtw.hpp"

%include "datamunge/geometry/frechet.hpp"

%include "datamunge/geometry/bounding_box.hpp"
%include "datamunge/geometry/rotating_calipers.hpp"
%include "datamunge/geometry/min_enclosing_circle.hpp"
%include "datamunge/geometry/simplify_polyline.hpp"
%include "datamunge/geometry/polygon_clip.hpp"
%include "datamunge/geometry/polygon_triangulation.hpp"

%template(StringVectorVector) std::vector<std::vector<std::string> >;

%include "datamunge/dstruct/directed_graph.hpp"
%template(DirectedGraph) datamunge::dstruct::DirectedGraph<std::string>;
%include "datamunge/dstruct/undirected_graph.hpp"
%template(UndirectedGraph) datamunge::dstruct::UndirectedGraph<std::string>;

// The map-based dijkstra()/bellman_ford()/floyd_warshall() (and their nested
// ShortestPaths/BellmanFordResult result types) return std::unordered_map, which isn't
// cleanly bindable everywhere -- ignored here in favor of the vector<pair<>>-based
// dijkstra_distances()/bellman_ford_distances()/bellman_ford_has_negative_cycle()
// convenience methods, which use the same already-proven std_pair/std_vector machinery.
%ignore datamunge::dstruct::WeightedGraph::dijkstra;
%ignore datamunge::dstruct::WeightedGraph::bellman_ford;
%ignore datamunge::dstruct::WeightedGraph::floyd_warshall;
%ignore datamunge::dstruct::WeightedGraph::ShortestPaths;
%ignore datamunge::dstruct::WeightedGraph::BellmanFordResult;
%include "datamunge/dstruct/weighted_graph.hpp"
%template(WeightedGraph) datamunge::dstruct::WeightedGraph<std::string, double>;
%template(VertexWeightPair) std::pair<std::string, double>;
%template(VertexWeightPairVector) std::vector<std::pair<std::string, double> >;

%template(ByteVector) std::vector<std::uint8_t>;
%include "datamunge/image/image.hpp"
%template(ImageVector) std::vector<datamunge::image::Image>;
%include "datamunge/image/color.hpp"
%include "datamunge/image/transform.hpp"
%include "datamunge/image/filters.hpp"
%template(IPairVector) std::vector<std::pair<int, int> >;
%include "datamunge/image/draw.hpp"
%include "datamunge/image/netpbm.hpp"
%include "datamunge/image/bmp.hpp"
%include "datamunge/image/png.hpp"

%template(CornerVector) std::vector<datamunge::cv::Corner>;
%include "datamunge/cv/corners.hpp"

%template(KeyPointVector) std::vector<datamunge::cv::KeyPoint>;
%include "datamunge/cv/blob.hpp"

%template(HoughLineVector) std::vector<datamunge::cv::HoughLine>;
%template(HoughCircleVector) std::vector<datamunge::cv::HoughCircle>;
%include "datamunge/cv/hough.hpp"

%include "datamunge/cv/morphology.hpp"
%include "datamunge/cv/segmentation.hpp"

%include "datamunge/cv/pyramid.hpp"

%template(FlowVectorVector) std::vector<datamunge::cv::FlowVector>;
%include "datamunge/cv/optical_flow.hpp"

// datamunge::cv::conv2d/max_pool2d/avg_pool2d/relu/sigmoid/softmax operate directly on
// datamunge::linalg::Tensor, which is deliberately %ignore'd for Python in favor of the
// hand-written SWIG-friendly datamunge::Tensor facade (see datamunge.hpp) -- so cv/nn.hpp
// itself is NOT %include'd here; its functionality is instead exposed via new facade methods
// on datamunge::Tensor (Tensor::conv2d/max_pool2d/avg_pool2d/relu/sigmoid/softmax,
// Tensor::from_image), declared and implemented alongside the rest of that facade.

// Unlike datamunge::cv::detail (which only ever touches already-bound types), these
// datamunge::filter::detail helpers take/return linalg::DenseMatrix -- a type that (per this
// codebase's established rule) can NEVER be a SWIG-bound parameter or return type anywhere.
// %include follows a header's own #include chain, so SWIG sees (and would otherwise try, and
// fail, to wrap) these even though they're implementation details never meant to be called
// from bound languages at all.
%ignore datamunge::filter::detail::to_dense;
%ignore datamunge::filter::detail::to_vector2d;
%ignore datamunge::filter::detail::to_column;
%ignore datamunge::filter::detail::to_vector;
%ignore datamunge::filter::detail::numerical_jacobian;

%feature("director") datamunge::filter::VectorFunction;
%include "datamunge/filter/vector_function.hpp"

%template(KalmanStateVector) std::vector<datamunge::filter::KalmanState>;
%include "datamunge/filter/kalman_filter.hpp"

%include "datamunge/filter/extended_kalman_filter.hpp"
%include "datamunge/filter/unscented_kalman_filter.hpp"
%include "datamunge/filter/information_filter.hpp"
%include "datamunge/filter/ensemble_kalman_filter.hpp"
%include "datamunge/filter/particle_filter.hpp"

%template(AlphaBetaStateVector) std::vector<datamunge::filter::AlphaBetaState>;
%template(AlphaBetaGammaStateVector) std::vector<datamunge::filter::AlphaBetaGammaState>;
%include "datamunge/filter/alpha_beta_filter.hpp"
