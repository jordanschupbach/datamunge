classdef LMM < datamunge.NativeObject
  % LMM  SWIG-friendly facade for datamunge::stats::LMM — a linear mixed model fit by (RE)ML, with a single grouping factor, e.g. "score ~ x1 + (1 + x1 | school)".
  %
  %   obj = datamunge.LMM (data, formula)
  %   obj = datamunge.LMM (data, formula, reml)
  %   obj = datamunge.LMM (data, formula, reml, de_population_size)
  %   obj = datamunge.LMM (data, formula, reml, de_population_size, de_max_generations)
  %   obj = datamunge.LMM (data, formula, reml, de_population_size, de_max_generations, theta_bound)
  %   obj = datamunge.LMM (data, formula, reml, de_population_size, de_max_generations, theta_bound, seed)
  %
  %   Methods:
  %     formula_text  C++ datamunge::LMM::formula_text.
  %     group_variable  C++ datamunge::LMM::group_variable.
  %     has_random_intercept  C++ datamunge::LMM::has_random_intercept.
  %     random_effect_names  C++ datamunge::LMM::random_effect_names.
  %     is_reml  C++ datamunge::LMM::is_reml.
  %     observations  C++ datamunge::LMM::observations.
  %     num_groups  C++ datamunge::LMM::num_groups.
  %     rank  C++ datamunge::LMM::rank.
  %     coefficients  C++ datamunge::LMM::coefficients.
  %     coefficient_names  C++ datamunge::LMM::coefficient_names.
  %     standard_errors  C++ datamunge::LMM::standard_errors.
  %     z_values  C++ datamunge::LMM::z_values.
  %     p_values  C++ datamunge::LMM::p_values.
  %     fitted_values  C++ datamunge::LMM::fitted_values.
  %     residuals  C++ datamunge::LMM::residuals.
  %     residual_variance  C++ datamunge::LMM::residual_variance.
  %     residual_std_dev  C++ datamunge::LMM::residual_std_dev.
  %     random_effect_std_devs  C++ datamunge::LMM::random_effect_std_devs.
  %     random_effect_correlation  C++ datamunge::LMM::random_effect_correlation.
  %     group_labels  C++ datamunge::LMM::group_labels.
  %     random_effects_for_group  The BLUP random-effect vector for the group at `group_index` (see group_labels()), in the same order as random_effect_names().
  %     log_likelihood  C++ datamunge::LMM::log_likelihood.
  %     deviance  C++ datamunge::LMM::deviance.
  %     aic  C++ datamunge::LMM::aic.
  %     bic  C++ datamunge::LMM::bic.
  %     summary  C++ datamunge::LMM::summary.
  %     print_summary  C++ datamunge::LMM::print_summary.
  %     predict  C++ datamunge::LMM::predict.
  %
  %   C++: class datamunge::LMM
  methods
    function obj = LMM (varargin)
      % LMM  Construct a C++ datamunge::LMM.
      %
      %   obj = datamunge.LMM (data, formula)
      %   obj = datamunge.LMM (data, formula, reml)
      %   obj = datamunge.LMM (data, formula, reml, de_population_size)
      %   obj = datamunge.LMM (data, formula, reml, de_population_size, de_max_generations)
      %   obj = datamunge.LMM (data, formula, reml, de_population_size, de_max_generations, theta_bound)
      %   obj = datamunge.LMM (data, formula, reml, de_population_size, de_max_generations, theta_bound, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     reml  logical, default true: REML (default) or maximum likelihood.
      %     de_population_size  integer, default 40
      %     de_max_generations  integer, default 300
      %     theta_bound  double, default 5.0: Box-constraint magnitude (relative to the residual SD) for the DifferentialEvolution search over variance-component parameters.
      %     seed  integer, default 42
      %
      %   C++: datamunge::LMM::LMM(const DataFrame & data, const std::string & formula, bool reml = true, std::size_t de_population_size = 40, std::size_t de_max_generations = 300, double theta_bound = 5.0, std::size_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2240, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::LMM::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LMM::formula_text() const
      r = __datamunge__(2212, obj);
    end
    function r = group_variable (obj)
      % group_variable  C++ datamunge::LMM::group_variable.
      %
      %   r = obj.group_variable ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LMM::group_variable() const
      r = __datamunge__(2213, obj);
    end
    function r = has_random_intercept (obj)
      % has_random_intercept  C++ datamunge::LMM::has_random_intercept.
      %
      %   r = obj.has_random_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::LMM::has_random_intercept() const
      r = __datamunge__(2214, obj);
    end
    function r = random_effect_names (obj)
      % random_effect_names  C++ datamunge::LMM::random_effect_names.
      %
      %   r = obj.random_effect_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LMM::random_effect_names() const
      r = __datamunge__(2215, obj);
    end
    function r = is_reml (obj)
      % is_reml  C++ datamunge::LMM::is_reml.
      %
      %   r = obj.is_reml ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::LMM::is_reml() const
      r = __datamunge__(2216, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::LMM::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LMM::observations() const
      r = __datamunge__(2217, obj);
    end
    function r = num_groups (obj)
      % num_groups  C++ datamunge::LMM::num_groups.
      %
      %   r = obj.num_groups ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LMM::num_groups() const
      r = __datamunge__(2218, obj);
    end
    function r = rank (obj)
      % rank  C++ datamunge::LMM::rank.
      %
      %   r = obj.rank ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LMM::rank() const
      r = __datamunge__(2219, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::LMM::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::coefficients() const
      r = __datamunge__(2220, obj);
    end
    function r = coefficient_names (obj)
      % coefficient_names  C++ datamunge::LMM::coefficient_names.
      %
      %   r = obj.coefficient_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LMM::coefficient_names() const
      r = __datamunge__(2221, obj);
    end
    function r = standard_errors (obj)
      % standard_errors  C++ datamunge::LMM::standard_errors.
      %
      %   r = obj.standard_errors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::standard_errors() const
      r = __datamunge__(2222, obj);
    end
    function r = z_values (obj)
      % z_values  C++ datamunge::LMM::z_values.
      %
      %   r = obj.z_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::z_values() const
      r = __datamunge__(2223, obj);
    end
    function r = p_values (obj)
      % p_values  C++ datamunge::LMM::p_values.
      %
      %   r = obj.p_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::p_values() const
      r = __datamunge__(2224, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::LMM::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::fitted_values() const
      r = __datamunge__(2225, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::LMM::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::residuals() const
      r = __datamunge__(2226, obj);
    end
    function r = residual_variance (obj)
      % residual_variance  C++ datamunge::LMM::residual_variance.
      %
      %   r = obj.residual_variance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LMM::residual_variance() const
      r = __datamunge__(2227, obj);
    end
    function r = residual_std_dev (obj)
      % residual_std_dev  C++ datamunge::LMM::residual_std_dev.
      %
      %   r = obj.residual_std_dev ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LMM::residual_std_dev() const
      r = __datamunge__(2228, obj);
    end
    function r = random_effect_std_devs (obj)
      % random_effect_std_devs  C++ datamunge::LMM::random_effect_std_devs.
      %
      %   r = obj.random_effect_std_devs ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::random_effect_std_devs() const
      r = __datamunge__(2229, obj);
    end
    function r = random_effect_correlation (obj, i, j)
      % random_effect_correlation  C++ datamunge::LMM::random_effect_correlation.
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
      %   C++: double datamunge::LMM::random_effect_correlation(std::size_t i, std::size_t j) const
      r = __datamunge__(2230, obj, i, j);
    end
    function r = group_labels (obj)
      % group_labels  C++ datamunge::LMM::group_labels.
      %
      %   r = obj.group_labels ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LMM::group_labels() const
      r = __datamunge__(2231, obj);
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
      %   C++: std::vector<double> datamunge::LMM::random_effects_for_group(std::size_t group_index) const
      r = __datamunge__(2232, obj, group_index);
    end
    function r = log_likelihood (obj)
      % log_likelihood  C++ datamunge::LMM::log_likelihood.
      %
      %   r = obj.log_likelihood ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LMM::log_likelihood() const
      r = __datamunge__(2233, obj);
    end
    function r = deviance (obj)
      % deviance  C++ datamunge::LMM::deviance.
      %
      %   r = obj.deviance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LMM::deviance() const
      r = __datamunge__(2234, obj);
    end
    function r = aic (obj)
      % aic  C++ datamunge::LMM::aic.
      %
      %   r = obj.aic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LMM::aic() const
      r = __datamunge__(2235, obj);
    end
    function r = bic (obj)
      % bic  C++ datamunge::LMM::bic.
      %
      %   r = obj.bic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LMM::bic() const
      r = __datamunge__(2236, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::LMM::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LMM::summary() const
      r = __datamunge__(2237, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::LMM::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::LMM::print_summary() const
      __datamunge__(2238, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::LMM::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LMM::predict(const DataFrame & newdata) const
      r = __datamunge__(2239, obj, newdata);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::LMM. Do not edit.
