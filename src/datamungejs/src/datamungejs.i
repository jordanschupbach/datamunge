%module(directors="1") datamungejs

%include <std_shared_ptr.i>
%include <stdint.i>
%include <std_vector.i>
%include <std_string.i>
%include <std_pair.i>

%template(IPair) std::pair<int, int>;
%template(DPair) std::pair<double, double>;
%template(DVectorPair) std::pair<std::vector<double>, std::vector<double> >;
%template(SPair) std::pair<std::string, std::string>;
%template(IVector) std::vector<int>;
%template(DVector) std::vector<double>;
%template(DVectorVector) std::vector<std::vector<double> >;
// Declared using the exact spelling (std::size_t) the facade header actually uses for every
// shape/index parameter -- avoids the %apply-based type-aliasing bug where SWIG treats
// std::vector<size_t> and std::vector<std::size_t> as distinct, incompatible identities
// (see datamunge_lua_bindings.md / datamunge_perl_bindings.md / datamunge_r_dollar_dispatch_bug.md
// / datamunge_d_bindings.md / datamunge_ocaml_bindings.md / datamunge_tcl_bindings.md /
// datamunge_guile_bindings.md memory for the same bug recurring in Lua, R, Perl, D, OCaml,
// Tcl, and Guile).
%template(SizeVector) std::vector<std::size_t>;
%template(SVector) std::vector<std::string>;

%feature("director") datamunge::Callback;

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
