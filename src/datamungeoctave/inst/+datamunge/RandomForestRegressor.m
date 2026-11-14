classdef RandomForestRegressor < datamunge.NativeObject
  % RandomForestRegressor  SWIG-friendly facade for datamunge::stats::RandomForestRegressor — a bagged ensemble of CART regression trees, each fit on a bootstrap sample with a random subset of predictors considered at every split.
  %
  %   obj = datamunge.RandomForestRegressor (data, formula)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, sample_fraction)
  %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, sample_fraction, seed)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::RandomForestRegressor::predictor_names.
  %     observations  C++ datamunge::RandomForestRegressor::observations.
  %     n_trees  C++ datamunge::RandomForestRegressor::n_trees.
  %     max_features_used  C++ datamunge::RandomForestRegressor::max_features_used.
  %     feature_importance  C++ datamunge::RandomForestRegressor::feature_importance.
  %     fitted_values  C++ datamunge::RandomForestRegressor::fitted_values.
  %     r_squared  C++ datamunge::RandomForestRegressor::r_squared.
  %     rmse  C++ datamunge::RandomForestRegressor::rmse.
  %     oob_r_squared  C++ datamunge::RandomForestRegressor::oob_r_squared.
  %     oob_rmse  C++ datamunge::RandomForestRegressor::oob_rmse.
  %     summary  C++ datamunge::RandomForestRegressor::summary.
  %     print_summary  C++ datamunge::RandomForestRegressor::print_summary.
  %     predict  C++ datamunge::RandomForestRegressor::predict.
  %     plot_predicted_vs_actual  C++ datamunge::RandomForestRegressor::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::RandomForestRegressor::plot_residuals_vs_fitted.
  %
  %   C++: class datamunge::RandomForestRegressor
  methods
    function obj = RandomForestRegressor (varargin)
      % RandomForestRegressor  Construct a C++ datamunge::RandomForestRegressor.
      %
      %   obj = datamunge.RandomForestRegressor (data, formula)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, sample_fraction)
      %   obj = datamunge.RandomForestRegressor (data, formula, n_trees, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap, sample_fraction, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     n_trees  integer, default 100
      %     max_depth  integer, default 10
      %     min_samples_split  integer, default 2
      %     min_samples_leaf  integer, default 1
      %     max_features  integer, default 0: 0 = auto (floor(number of predictors / 3)).
      %     bootstrap  logical, default true
      %     sample_fraction  double, default 1.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::RandomForestRegressor::RandomForestRegressor(const DataFrame & data, const std::string & formula, std::size_t n_trees = 100, std::size_t max_depth = 10, std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1, std::size_t max_features = 0, bool bootstrap = true, double sample_fraction = 1.0, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2378, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::RandomForestRegressor::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::RandomForestRegressor::predictor_names() const
      r = __datamunge__(2363, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::RandomForestRegressor::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::RandomForestRegressor::observations() const
      r = __datamunge__(2364, obj);
    end
    function r = n_trees (obj)
      % n_trees  C++ datamunge::RandomForestRegressor::n_trees.
      %
      %   r = obj.n_trees ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::RandomForestRegressor::n_trees() const
      r = __datamunge__(2365, obj);
    end
    function r = max_features_used (obj)
      % max_features_used  C++ datamunge::RandomForestRegressor::max_features_used.
      %
      %   r = obj.max_features_used ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::RandomForestRegressor::max_features_used() const
      r = __datamunge__(2366, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::RandomForestRegressor::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::RandomForestRegressor::feature_importance() const
      r = __datamunge__(2367, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::RandomForestRegressor::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::RandomForestRegressor::fitted_values() const
      r = __datamunge__(2368, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::RandomForestRegressor::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::RandomForestRegressor::r_squared() const
      r = __datamunge__(2369, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::RandomForestRegressor::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::RandomForestRegressor::rmse() const
      r = __datamunge__(2370, obj);
    end
    function r = oob_r_squared (obj)
      % oob_r_squared  C++ datamunge::RandomForestRegressor::oob_r_squared.
      %
      %   r = obj.oob_r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::RandomForestRegressor::oob_r_squared() const
      r = __datamunge__(2371, obj);
    end
    function r = oob_rmse (obj)
      % oob_rmse  C++ datamunge::RandomForestRegressor::oob_rmse.
      %
      %   r = obj.oob_rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::RandomForestRegressor::oob_rmse() const
      r = __datamunge__(2372, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::RandomForestRegressor::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::RandomForestRegressor::summary() const
      r = __datamunge__(2373, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::RandomForestRegressor::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::RandomForestRegressor::print_summary() const
      __datamunge__(2374, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::RandomForestRegressor::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::RandomForestRegressor::predict(const DataFrame & newdata) const
      r = __datamunge__(2375, obj, newdata);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::RandomForestRegressor::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::RandomForestRegressor::plot_predicted_vs_actual() const
      r = __datamunge__(2376, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::RandomForestRegressor::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::RandomForestRegressor::plot_residuals_vs_fitted() const
      r = __datamunge__(2377, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::RandomForestRegressor. Do not edit.
