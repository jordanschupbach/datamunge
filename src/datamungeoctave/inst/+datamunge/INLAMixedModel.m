classdef INLAMixedModel < datamunge.NativeObject
  % INLAMixedModel  SWIG-friendly facade for datamunge::stats::INLAMixedModel — a mixed model (same formula grammar as GLMM/LMM) fit by Integrated Nested Laplace Approximation instead of penalized quasi-likelihood/REML: a genuinely Bayesian alternative returning real posterior means/sds (including for the variance components) rather than point estimates + asymptotic standard errors. See datamunge::bayes::INLA (C++-only, not exposed to bindings -- same reasoning as AutodiffModel) for the underlying algorithm.
  %
  %   obj = datamunge.INLAMixedModel (data, formula)
  %   obj = datamunge.INLAMixedModel (data, formula, family)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span, mode_population_size)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span, mode_population_size, mode_max_generations)
  %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span, mode_population_size, mode_max_generations, seed)
  %
  %   Methods:
  %     formula_text  C++ datamunge::INLAMixedModel::formula_text.
  %     family  C++ datamunge::INLAMixedModel::family.
  %     group_variable  C++ datamunge::INLAMixedModel::group_variable.
  %     random_effect_names  C++ datamunge::INLAMixedModel::random_effect_names.
  %     observations  C++ datamunge::INLAMixedModel::observations.
  %     num_groups  C++ datamunge::INLAMixedModel::num_groups.
  %     fixed_effects_mean  C++ datamunge::INLAMixedModel::fixed_effects_mean.
  %     fixed_effects_sd  C++ datamunge::INLAMixedModel::fixed_effects_sd.
  %     coefficient_names  C++ datamunge::INLAMixedModel::coefficient_names.
  %     random_effect_std_devs  C++ datamunge::INLAMixedModel::random_effect_std_devs.
  %     residual_std_dev  Gaussian family only; throws for binomial/poisson (dispersion fixed at 1).
  %     group_labels  C++ datamunge::INLAMixedModel::group_labels.
  %     random_effects_mean_for_group  The posterior-mean/sd BLUP-like random-effect vector for the group at `group_index` (see group_labels()), in random_effect_names() order.
  %     random_effects_sd_for_group  C++ datamunge::INLAMixedModel::random_effects_sd_for_group.
  %     log_marginal_likelihood  log p(y), approximated by the same INLA machinery used to fit the model.
  %     summary  C++ datamunge::INLAMixedModel::summary.
  %     print_summary  C++ datamunge::INLAMixedModel::print_summary.
  %     predict  C++ datamunge::INLAMixedModel::predict.
  %
  %   C++: class datamunge::INLAMixedModel
  methods
    function obj = INLAMixedModel (varargin)
      % INLAMixedModel  Construct a C++ datamunge::INLAMixedModel.
      %
      %   obj = datamunge.INLAMixedModel (data, formula)
      %   obj = datamunge.INLAMixedModel (data, formula, family)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span, mode_population_size)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span, mode_population_size, mode_max_generations)
      %   obj = datamunge.INLAMixedModel (data, formula, family, strategy, fixed_effect_prior_sd, grid_points_per_dim, grid_span, mode_population_size, mode_max_generations, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     family  char row, default 'gaussian': One of "gaussian" (identity link), "binomial" (logit link), or "poisson" (log link).
      %     strategy  char row, default 'grid': One of "grid" (integrate over hyperparameter uncertainty) or "eb" (empirical Bayes: fix hyperparameters at their posterior mode).
      %     fixed_effect_prior_sd  double, default 1e+03
      %     grid_points_per_dim  integer, default 7
      %     grid_span  double, default 4.0
      %     mode_population_size  integer, default 40
      %     mode_max_generations  integer, default 200
      %     seed  integer, default 42
      %
      %   C++: datamunge::INLAMixedModel::INLAMixedModel(const DataFrame & data, const std::string & formula, const std::string & family = "gaussian", const std::string & strategy = "grid", double fixed_effect_prior_sd = 1000.0, std::size_t grid_points_per_dim = 7, double grid_span = 4.0, std::size_t mode_population_size = 40, std::size_t mode_max_generations = 200, std::size_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2285, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::INLAMixedModel::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::INLAMixedModel::formula_text() const
      r = __datamunge__(2267, obj);
    end
    function r = family (obj)
      % family  C++ datamunge::INLAMixedModel::family.
      %
      %   r = obj.family ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::INLAMixedModel::family() const
      r = __datamunge__(2268, obj);
    end
    function r = group_variable (obj)
      % group_variable  C++ datamunge::INLAMixedModel::group_variable.
      %
      %   r = obj.group_variable ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::INLAMixedModel::group_variable() const
      r = __datamunge__(2269, obj);
    end
    function r = random_effect_names (obj)
      % random_effect_names  C++ datamunge::INLAMixedModel::random_effect_names.
      %
      %   r = obj.random_effect_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::INLAMixedModel::random_effect_names() const
      r = __datamunge__(2270, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::INLAMixedModel::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::INLAMixedModel::observations() const
      r = __datamunge__(2271, obj);
    end
    function r = num_groups (obj)
      % num_groups  C++ datamunge::INLAMixedModel::num_groups.
      %
      %   r = obj.num_groups ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::INLAMixedModel::num_groups() const
      r = __datamunge__(2272, obj);
    end
    function r = fixed_effects_mean (obj)
      % fixed_effects_mean  C++ datamunge::INLAMixedModel::fixed_effects_mean.
      %
      %   r = obj.fixed_effects_mean ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::INLAMixedModel::fixed_effects_mean() const
      r = __datamunge__(2273, obj);
    end
    function r = fixed_effects_sd (obj)
      % fixed_effects_sd  C++ datamunge::INLAMixedModel::fixed_effects_sd.
      %
      %   r = obj.fixed_effects_sd ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::INLAMixedModel::fixed_effects_sd() const
      r = __datamunge__(2274, obj);
    end
    function r = coefficient_names (obj)
      % coefficient_names  C++ datamunge::INLAMixedModel::coefficient_names.
      %
      %   r = obj.coefficient_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::INLAMixedModel::coefficient_names() const
      r = __datamunge__(2275, obj);
    end
    function r = random_effect_std_devs (obj)
      % random_effect_std_devs  C++ datamunge::INLAMixedModel::random_effect_std_devs.
      %
      %   r = obj.random_effect_std_devs ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::INLAMixedModel::random_effect_std_devs() const
      r = __datamunge__(2276, obj);
    end
    function r = residual_std_dev (obj)
      % residual_std_dev  Gaussian family only; throws for binomial/poisson (dispersion fixed at 1).
      %
      %   r = obj.residual_std_dev ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::INLAMixedModel::residual_std_dev() const
      r = __datamunge__(2277, obj);
    end
    function r = group_labels (obj)
      % group_labels  C++ datamunge::INLAMixedModel::group_labels.
      %
      %   r = obj.group_labels ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::INLAMixedModel::group_labels() const
      r = __datamunge__(2278, obj);
    end
    function r = random_effects_mean_for_group (obj, group_index)
      % random_effects_mean_for_group  The posterior-mean/sd BLUP-like random-effect vector for the group at `group_index` (see group_labels()), in random_effect_names() order.
      %
      %   r = obj.random_effects_mean_for_group (group_index)
      %
      %   Inputs:
      %     group_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::INLAMixedModel::random_effects_mean_for_group(std::size_t group_index) const
      r = __datamunge__(2279, obj, group_index);
    end
    function r = random_effects_sd_for_group (obj, group_index)
      % random_effects_sd_for_group  C++ datamunge::INLAMixedModel::random_effects_sd_for_group.
      %
      %   r = obj.random_effects_sd_for_group (group_index)
      %
      %   Inputs:
      %     group_index  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::INLAMixedModel::random_effects_sd_for_group(std::size_t group_index) const
      r = __datamunge__(2280, obj, group_index);
    end
    function r = log_marginal_likelihood (obj)
      % log_marginal_likelihood  log p(y), approximated by the same INLA machinery used to fit the model.
      %
      %   r = obj.log_marginal_likelihood ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::INLAMixedModel::log_marginal_likelihood() const
      r = __datamunge__(2281, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::INLAMixedModel::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::INLAMixedModel::summary() const
      r = __datamunge__(2282, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::INLAMixedModel::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::INLAMixedModel::print_summary() const
      __datamunge__(2283, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::INLAMixedModel::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::INLAMixedModel::predict(const DataFrame & newdata) const
      r = __datamunge__(2284, obj, newdata);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::INLAMixedModel. Do not edit.
