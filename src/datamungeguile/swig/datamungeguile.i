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
// Guile's std_vector.i scalar fast-path (specialize_stl_vector) is keyed to a fixed list of
// exact literal spellings (unsigned long, double, std::string, ...) and doesn't textually
// match the "std::size_t" typedef even though they're the same underlying type on this
// platform -- so BY-VALUE std::vector<std::size_t> returns fall back to the generic
// (non-specialized) class-attached template typemap, which wraps EACH element as an
// individually pointer-boxed collectable-swig-pointer object instead of a plain Scheme vector
// of integers. Unlike Tcl, a freestanding "%typemap(out) std::vector<std::size_t>" override
// declared before %template(SizeVector) does NOT take precedence here -- Guile's generic
// template typemap is attached to the template class body itself (inside "namespace std {
// template<class T> class vector { %typemap(out) vector<T> {...} }; }") and appears to win
// regardless of file order. The correct fix is to invoke the Lib's own specialize_stl_vector
// macro directly for std::size_t, mirroring exactly how it already covers unsigned long/double/
// etc -- this must be done inside "namespace std" so the resulting "template<> class
// vector<std::size_t>" is generated in the right namespace, and before %template(SizeVector)
// below.
namespace std {
  specialize_stl_vector(std::size_t, scm_is_number, scm_to_ulong, scm_from_ulong);
}

%template(DPair) std::pair<double, double>;
%template(IPair) std::pair<int, int>;
%template(DVectorPair) std::pair<std::vector<double>, std::vector<double> >;
%template(DVector) std::vector<double>;
%template(DVectorVector) std::vector<std::vector<double> >;
%template(IVector) std::vector<int>;
// Declared using the exact spelling (std::size_t) the facade header actually uses for every
// shape/index parameter -- avoids the %apply-based type-aliasing bug where SWIG treats
// std::vector<size_t> and std::vector<std::size_t> as distinct, incompatible identities
// (see datamunge_lua_bindings.md / datamunge_perl_bindings.md / datamunge_r_dollar_dispatch_bug.md
// / datamunge_d_bindings.md / datamunge_ocaml_bindings.md / datamunge_tcl_bindings.md memory for
// the same bug recurring in Lua, R, Perl, D, OCaml, and Tcl).
%template(SizeVector) std::vector<std::size_t>;
%template(SVector) std::vector<std::string>;

%feature("director") datamunge::Callback;
%feature("director") datamunge::optim::ArbitraryFunction;
%feature("director") datamunge::optim::DifferentiableFunction;
%feature("director") datamunge::optim::SeparableFunction;
%feature("director") datamunge::optim::DifferentiableSeparableFunction;

%{
#include "datamunge/datamunge.hpp"
%}

%include <datamunge/datamunge.hpp>
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
