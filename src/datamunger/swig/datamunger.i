%module(directors="1") datamunger

%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%ignore datamunge::dstruct::NullableColumn;
%ignore datamunge::dstruct::DataFrame;
%ignore datamunge::dstruct::GroupedDataFrame;
%ignore datamunge::gis::ShapeType;
%ignore datamunge::gis::Shape;
%ignore datamunge::gis::ShapefileData;
%ignore datamunge::gis::ShapeLayer;
%ignore datamunge::gis::to_string;
%ignore datamunge::gis::read_shp;
%ignore datamunge::gis::read_dbf;
%ignore datamunge::stats::PCA;
%ignore datamunge::stats::PCAOptions;
%ignore datamunge::stats::MDS;
%ignore datamunge::stats::MDSOptions;
%ignore datamunge::stats::Isomap;
%ignore datamunge::stats::IsomapOptions;
%ignore datamunge::stats::LLE;
%ignore datamunge::stats::LLEOptions;
%ignore datamunge::stats::TSNE;
%ignore datamunge::stats::TSNEOptions;
%ignore datamunge::stats::LaplacianEigenmaps;
%ignore datamunge::stats::LaplacianEigenmapsOptions;
%ignore datamunge::stats::DiffusionMaps;
%ignore datamunge::stats::DiffusionMapsOptions;
%ignore datamunge::stats::KernelPCA;
%ignore datamunge::stats::KernelPCAOptions;
%ignore datamunge::stats::SammonMapping;
%ignore datamunge::stats::SammonMappingOptions;
%ignore datamunge::stats::UMAP;
%ignore datamunge::stats::UMAPOptions;
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
// Polynomial::divmod returns std::pair<Polynomial, Polynomial> -- this codebase avoids
// exposing custom-class-pair returns to SWIG bindings directly (see poly_quotient()/
// poly_remainder(), the binding-friendly free-function alternatives, in polynomial.hpp).
%ignore datamunge::algebra::Polynomial::divmod;
// berlekamp_factor() takes/returns std::vector<long long>, a container type with no proven
// SWIG binding template in this codebase (the size_t vector traits-collision bugs elsewhere
// make new fixed-width-integer vector templates risky) -- berlekamp_factor_mod() is the
// double-based binding-friendly wrapper, reusing the already-proven DVector/DVectorVector.
%ignore datamunge::algebra::berlekamp_factor;
%ignore datamunge::algebra::detail::gf_mod;
%ignore datamunge::algebra::detail::gf_trim;
%ignore datamunge::algebra::detail::gf_degree;
%ignore datamunge::algebra::detail::gf_is_zero;
%ignore datamunge::algebra::detail::gf_mod_inverse;
%ignore datamunge::algebra::detail::gf_divmod;
%ignore datamunge::algebra::detail::gf_gcd;
%ignore datamunge::algebra::detail::gf_mul_x_mod;
// Expr's node tree (detail::ExprNode/ExprOp) is a private implementation detail never touched
// by the public Expr API directly -- ExprOp is a namespace-level enum class, which hits the
// same swig-jse R-backend doubled-prefix bug as TrendType/MonomialOrder/etc (see the
// perl-fix block in justfile's prebuild-r), so it's simplest to just not expose it at all.
%ignore datamunge::algebra::detail::ExprOp;
%ignore datamunge::algebra::detail::ExprNode;
%template(IPair) std::pair<int, int>;
%template(DPair) std::pair<double, double>;
%template(DVectorPair) std::pair<std::vector<double>, std::vector<double> >;
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
%feature("director") datamunge::optim::ProximalFunction;
%feature("director") datamunge::optim::HessianFunction;
%feature("director") datamunge::optim::EqualityConstrainedFunction;
%feature("director") datamunge::optim::InequalityConstrainedFunction;
%feature("director") datamunge::optim::ResidualFunction;
%feature("director") datamunge::optim::BayesianSurrogate;

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
%include "datamunge/optim/optim.hpp"
%include "datamunge/optim/gradient_descent.hpp"
%include "datamunge/optim/adam.hpp"
%include "datamunge/optim/adagrad.hpp"
%include "datamunge/optim/adadelta.hpp"
%include "datamunge/optim/amsgrad.hpp"
%include "datamunge/optim/nadam.hpp"
%include "datamunge/optim/rmsprop.hpp"
%include "datamunge/optim/lbfgs.hpp"
%include "datamunge/optim/nelder_mead.hpp"
%include "datamunge/optim/sgd.hpp"
%include "datamunge/optim/svrg.hpp"
%include "datamunge/optim/saga.hpp"
%include "datamunge/optim/coordinate_descent.hpp"
%include "datamunge/optim/randomized_block_coordinate_descent.hpp"
%include "datamunge/optim/nesterov_accelerated_gradient.hpp"
%include "datamunge/optim/conjugate_gradient.hpp"
%include "datamunge/optim/cma_es.hpp"
%include "datamunge/optim/simulated_annealing.hpp"
%include "datamunge/optim/pso.hpp"
%include "datamunge/optim/differential_evolution.hpp"
%include "datamunge/optim/genetic_algorithm.hpp"
%include "datamunge/optim/acor.hpp"
%include "datamunge/optim/artificial_bee_colony.hpp"
%include "datamunge/optim/cross_entropy_method.hpp"
%include "datamunge/optim/cuckoo_search.hpp"
%include "datamunge/optim/estimation_of_distribution.hpp"
%include "datamunge/optim/evolution_strategy.hpp"
%include "datamunge/optim/firefly_algorithm.hpp"
%include "datamunge/optim/grey_wolf_optimizer.hpp"
%include "datamunge/optim/harmony_search.hpp"
%include "datamunge/optim/parallel_tempering.hpp"
%include "datamunge/optim/whale_optimization.hpp"
%include "datamunge/optim/fista.hpp"
%include "datamunge/optim/proximal_gradient.hpp"
%include "datamunge/optim/levenberg_marquardt.hpp"
%include "datamunge/optim/newton.hpp"
%include "datamunge/optim/trust_region_newton.hpp"
%include "datamunge/optim/augmented_lagrangian.hpp"
%include "datamunge/optim/sqp.hpp"
%include "datamunge/optim/interior_point.hpp"
%include "datamunge/optim/bayesian_optimization.hpp"
%include "datamunge/optim/rbf_gaussian_process_surrogate.hpp"
%include "datamunge/bayes/map.hpp"
%include "datamunge/bayes/hmc.hpp"
%include "datamunge/bayes/nuts.hpp"
%include "datamunge/bayes/rwm.hpp"
%include "datamunge/bayes/gibbs.hpp"
%include "datamunge/bayes/importance_sampling.hpp"
%ignore datamunge::ode::ODESolution::t;
%ignore datamunge::ode::ODESolution::y;
%include "datamunge/ode/rhs.hpp"
%include "datamunge/ode/ode_solver.hpp"

%include "datamunge/algebra/polynomial.hpp"
%template(PolynomialVector) std::vector<datamunge::algebra::Polynomial>;

%include "datamunge/algebra/poly_gcd.hpp"
%include "datamunge/algebra/interpolation.hpp"
%include "datamunge/algebra/resultant.hpp"
%include "datamunge/algebra/rational_function.hpp"
%include "datamunge/algebra/sturm.hpp"
%include "datamunge/algebra/descartes.hpp"

%template(SquareFreeFactorVector) std::vector<datamunge::algebra::SquareFreeFactor>;
%include "datamunge/algebra/square_free.hpp"

%include "datamunge/algebra/modular.hpp"
%include "datamunge/algebra/poly_factor_gf_p.hpp"

%include "datamunge/algebra/monomial_order.hpp"
%template(IVectorVector) std::vector<std::vector<int> >;
%include "datamunge/algebra/multivariate_polynomial.hpp"
%template(MultivariatePolynomialVector) std::vector<datamunge::algebra::MultivariatePolynomial>;
%include "datamunge/algebra/groebner.hpp"

%include "datamunge/algebra/expression.hpp"
