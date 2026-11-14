classdef GLMM < datamunge.NativeObject
  % GLMM  SWIG-friendly facade for datamunge::stats::GLMM — a generalized linear mixed model fit by penalized quasi-likelihood, with a single grouping factor, e.g. "success ~ x1 + (1 | school)" with family="binomial".
  %
  %   obj = datamunge.GLMM (data, formula)
  %   obj = datamunge.GLMM (data, formula, family)
  %   obj = datamunge.GLMM (data, formula, family, max_iterations)
  %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol)
  %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size)
  %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size, de_max_generations)
  %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size, de_max_generations, theta_bound)
  %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size, de_max_generations, theta_bound, seed)
  %
  %   Methods:
  %     formula_text  C++ datamunge::GLMM::formula_text.
  %     family  C++ datamunge::GLMM::family.
  %     group_variable  C++ datamunge::GLMM::group_variable.
  %     has_random_intercept  C++ datamunge::GLMM::has_random_intercept.
  %     random_effect_names  C++ datamunge::GLMM::random_effect_names.
  %     observations  C++ datamunge::GLMM::observations.
  %     num_groups  C++ datamunge::GLMM::num_groups.
  %     rank  C++ datamunge::GLMM::rank.
  %     iterations  C++ datamunge::GLMM::iterations.
  %     coefficients  C++ datamunge::GLMM::coefficients.
  %     coefficient_names  C++ datamunge::GLMM::coefficient_names.
  %     standard_errors  C++ datamunge::GLMM::standard_errors.
  %     z_values  C++ datamunge::GLMM::z_values.
  %     p_values  C++ datamunge::GLMM::p_values.
  %     fitted_values  C++ datamunge::GLMM::fitted_values.
  %     random_effect_std_devs  C++ datamunge::GLMM::random_effect_std_devs.
  %     random_effect_correlation  C++ datamunge::GLMM::random_effect_correlation.
  %     group_labels  C++ datamunge::GLMM::group_labels.
  %     random_effects_for_group  The BLUP random-effect vector for the group at `group_index` (see group_labels()), in the same order as random_effect_names().
  %     deviance  C++ datamunge::GLMM::deviance.
  %     aic  C++ datamunge::GLMM::aic.
  %     bic  C++ datamunge::GLMM::bic.
  %     summary  C++ datamunge::GLMM::summary.
  %     print_summary  C++ datamunge::GLMM::print_summary.
  %     predict  C++ datamunge::GLMM::predict.
  %
  %   C++: class datamunge::GLMM
  methods
    function obj = GLMM (varargin)
      % GLMM  Construct a C++ datamunge::GLMM.
      %
      %   obj = datamunge.GLMM (data, formula)
      %   obj = datamunge.GLMM (data, formula, family)
      %   obj = datamunge.GLMM (data, formula, family, max_iterations)
      %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol)
      %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size)
      %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size, de_max_generations)
      %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size, de_max_generations, theta_bound)
      %   obj = datamunge.GLMM (data, formula, family, max_iterations, tol, de_population_size, de_max_generations, theta_bound, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     family  char row, default 'binomial': One of "binomial" (logit link) or "poisson" (log link).
      %     max_iterations  integer, default 20
      %     tol  double, default 1e-06
      %     de_population_size  integer, default 30
      %     de_max_generations  integer, default 150
      %     theta_bound  double, default 5.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::GLMM::GLMM(const DataFrame & data, const std::string & formula, const std::string & family = "binomial", std::size_t max_iterations = 20, double tol = 9.9999999999999995E-7, std::size_t de_population_size = 30, std::size_t de_max_generations = 150, double theta_bound = 5.0, std::size_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2266, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::GLMM::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLMM::formula_text() const
      r = __datamunge__(2241, obj);
    end
    function r = family (obj)
      % family  C++ datamunge::GLMM::family.
      %
      %   r = obj.family ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLMM::family() const
      r = __datamunge__(2242, obj);
    end
    function r = group_variable (obj)
      % group_variable  C++ datamunge::GLMM::group_variable.
      %
      %   r = obj.group_variable ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLMM::group_variable() const
      r = __datamunge__(2243, obj);
    end
    function r = has_random_intercept (obj)
      % has_random_intercept  C++ datamunge::GLMM::has_random_intercept.
      %
      %   r = obj.has_random_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::GLMM::has_random_intercept() const
      r = __datamunge__(2244, obj);
    end
    function r = random_effect_names (obj)
      % random_effect_names  C++ datamunge::GLMM::random_effect_names.
      %
      %   r = obj.random_effect_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GLMM::random_effect_names() const
      r = __datamunge__(2245, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::GLMM::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLMM::observations() const
      r = __datamunge__(2246, obj);
    end
    function r = num_groups (obj)
      % num_groups  C++ datamunge::GLMM::num_groups.
      %
      %   r = obj.num_groups ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLMM::num_groups() const
      r = __datamunge__(2247, obj);
    end
    function r = rank (obj)
      % rank  C++ datamunge::GLMM::rank.
      %
      %   r = obj.rank ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLMM::rank() const
      r = __datamunge__(2248, obj);
    end
    function r = iterations (obj)
      % iterations  C++ datamunge::GLMM::iterations.
      %
      %   r = obj.iterations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLMM::iterations() const
      r = __datamunge__(2249, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::GLMM::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::coefficients() const
      r = __datamunge__(2250, obj);
    end
    function r = coefficient_names (obj)
      % coefficient_names  C++ datamunge::GLMM::coefficient_names.
      %
      %   r = obj.coefficient_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GLMM::coefficient_names() const
      r = __datamunge__(2251, obj);
    end
    function r = standard_errors (obj)
      % standard_errors  C++ datamunge::GLMM::standard_errors.
      %
      %   r = obj.standard_errors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::standard_errors() const
      r = __datamunge__(2252, obj);
    end
    function r = z_values (obj)
      % z_values  C++ datamunge::GLMM::z_values.
      %
      %   r = obj.z_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::z_values() const
      r = __datamunge__(2253, obj);
    end
    function r = p_values (obj)
      % p_values  C++ datamunge::GLMM::p_values.
      %
      %   r = obj.p_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::p_values() const
      r = __datamunge__(2254, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::GLMM::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::fitted_values() const
      r = __datamunge__(2255, obj);
    end
    function r = random_effect_std_devs (obj)
      % random_effect_std_devs  C++ datamunge::GLMM::random_effect_std_devs.
      %
      %   r = obj.random_effect_std_devs ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::random_effect_std_devs() const
      r = __datamunge__(2256, obj);
    end
    function r = random_effect_correlation (obj, i, j)
      % random_effect_correlation  C++ datamunge::GLMM::random_effect_correlation.
      %
      %   r = obj.random_effect_correlation (i, j)
      %
      %   Inputs:
      %     i  integer
      %     j  integer
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLMM::random_effect_correlation(std::size_t i, std::size_t j) const
      r = __datamunge__(2257, obj, i, j);
    end
    function r = group_labels (obj)
      % group_labels  C++ datamunge::GLMM::group_labels.
      %
      %   r = obj.group_labels ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GLMM::group_labels() const
      r = __datamunge__(2258, obj);
    end
    function r = random_effects_for_group (obj, group_index)
      % random_effects_for_group  The BLUP random-effect vector for the group at `group_index` (see group_labels()), in the same order as random_effect_names().
      %
      %   r = obj.random_effects_for_group (group_index)
      %
      %   Inputs:
      %     group_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::random_effects_for_group(std::size_t group_index) const
      r = __datamunge__(2259, obj, group_index);
    end
    function r = deviance (obj)
      % deviance  C++ datamunge::GLMM::deviance.
      %
      %   r = obj.deviance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLMM::deviance() const
      r = __datamunge__(2260, obj);
    end
    function r = aic (obj)
      % aic  C++ datamunge::GLMM::aic.
      %
      %   r = obj.aic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLMM::aic() const
      r = __datamunge__(2261, obj);
    end
    function r = bic (obj)
      % bic  C++ datamunge::GLMM::bic.
      %
      %   r = obj.bic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLMM::bic() const
      r = __datamunge__(2262, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::GLMM::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLMM::summary() const
      r = __datamunge__(2263, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::GLMM::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::GLMM::print_summary() const
      __datamunge__(2264, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::GLMM::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLMM::predict(const DataFrame & newdata) const
      r = __datamunge__(2265, obj, newdata);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::GLMM. Do not edit.
