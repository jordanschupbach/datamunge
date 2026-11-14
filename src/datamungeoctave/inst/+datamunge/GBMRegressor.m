classdef GBMRegressor < datamunge.NativeObject
  % GBMRegressor  SWIG-friendly facade for datamunge::stats::GBMRegressor — a sequence of shallow regression trees, each fit to the residuals of the current ensemble (gradient boosting on squared error).
  %
  %   obj = datamunge.GBMRegressor (data, formula)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample)
  %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::GBMRegressor::predictor_names.
  %     observations  C++ datamunge::GBMRegressor::observations.
  %     n_trees  C++ datamunge::GBMRegressor::n_trees.
  %     feature_importance  C++ datamunge::GBMRegressor::feature_importance.
  %     fitted_values  C++ datamunge::GBMRegressor::fitted_values.
  %     r_squared  C++ datamunge::GBMRegressor::r_squared.
  %     rmse  C++ datamunge::GBMRegressor::rmse.
  %     training_deviance  Mean squared error on the training set after each boosting round (length n_trees()).
  %     summary  C++ datamunge::GBMRegressor::summary.
  %     print_summary  C++ datamunge::GBMRegressor::print_summary.
  %     predict  C++ datamunge::GBMRegressor::predict.
  %     plot_predicted_vs_actual  C++ datamunge::GBMRegressor::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::GBMRegressor::plot_residuals_vs_fitted.
  %     plot_training_deviance  C++ datamunge::GBMRegressor::plot_training_deviance.
  %
  %   C++: class datamunge::GBMRegressor
  methods
    function obj = GBMRegressor (varargin)
      % GBMRegressor  Construct a C++ datamunge::GBMRegressor.
      %
      %   obj = datamunge.GBMRegressor (data, formula)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample)
      %   obj = datamunge.GBMRegressor (data, formula, n_trees, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     n_trees  integer, default 100
      %     learning_rate  double, default 0.1
      %     max_depth  integer, default 3
      %     min_samples_split  integer, default 2
      %     min_samples_leaf  integer, default 1
      %     subsample  double, default 1.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::GBMRegressor::GBMRegressor(const DataFrame & data, const std::string & formula, std::size_t n_trees = 100, double learning_rate = 0.10000000000000001, std::size_t max_depth = 3, std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1, double subsample = 1.0, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2663, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::GBMRegressor::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GBMRegressor::predictor_names() const
      r = __datamunge__(2649, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::GBMRegressor::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GBMRegressor::observations() const
      r = __datamunge__(2650, obj);
    end
    function r = n_trees (obj)
      % n_trees  C++ datamunge::GBMRegressor::n_trees.
      %
      %   r = obj.n_trees ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GBMRegressor::n_trees() const
      r = __datamunge__(2651, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::GBMRegressor::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GBMRegressor::feature_importance() const
      r = __datamunge__(2652, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::GBMRegressor::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GBMRegressor::fitted_values() const
      r = __datamunge__(2653, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::GBMRegressor::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GBMRegressor::r_squared() const
      r = __datamunge__(2654, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::GBMRegressor::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GBMRegressor::rmse() const
      r = __datamunge__(2655, obj);
    end
    function r = training_deviance (obj)
      % training_deviance  Mean squared error on the training set after each boosting round (length n_trees()).
      %
      %   r = obj.training_deviance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GBMRegressor::training_deviance() const
      r = __datamunge__(2656, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::GBMRegressor::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GBMRegressor::summary() const
      r = __datamunge__(2657, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::GBMRegressor::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::GBMRegressor::print_summary() const
      __datamunge__(2658, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::GBMRegressor::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GBMRegressor::predict(const DataFrame & newdata) const
      r = __datamunge__(2659, obj, newdata);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::GBMRegressor::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GBMRegressor::plot_predicted_vs_actual() const
      r = __datamunge__(2660, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::GBMRegressor::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GBMRegressor::plot_residuals_vs_fitted() const
      r = __datamunge__(2661, obj);
    end
    function r = plot_training_deviance (obj)
      % plot_training_deviance  C++ datamunge::GBMRegressor::plot_training_deviance.
      %
      %   r = obj.plot_training_deviance ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GBMRegressor::plot_training_deviance() const
      r = __datamunge__(2662, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::GBMRegressor. Do not edit.
