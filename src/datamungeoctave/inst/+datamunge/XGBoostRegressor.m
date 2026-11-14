classdef XGBoostRegressor < datamunge.NativeObject
  % XGBoostRegressor  SWIG-friendly facade for datamunge::stats::XGBoostRegressor — regularized, second-order gradient boosting regression with the same tree-growing objective as the XGBoost algorithm.
  %
  %   obj = datamunge.XGBoostRegressor (data, formula)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree)
  %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::XGBoostRegressor::predictor_names.
  %     observations  C++ datamunge::XGBoostRegressor::observations.
  %     n_trees  C++ datamunge::XGBoostRegressor::n_trees.
  %     feature_importance  C++ datamunge::XGBoostRegressor::feature_importance.
  %     fitted_values  C++ datamunge::XGBoostRegressor::fitted_values.
  %     r_squared  C++ datamunge::XGBoostRegressor::r_squared.
  %     rmse  C++ datamunge::XGBoostRegressor::rmse.
  %     training_deviance  C++ datamunge::XGBoostRegressor::training_deviance.
  %     summary  C++ datamunge::XGBoostRegressor::summary.
  %     print_summary  C++ datamunge::XGBoostRegressor::print_summary.
  %     predict  C++ datamunge::XGBoostRegressor::predict.
  %     plot_predicted_vs_actual  C++ datamunge::XGBoostRegressor::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::XGBoostRegressor::plot_residuals_vs_fitted.
  %     plot_training_deviance  C++ datamunge::XGBoostRegressor::plot_training_deviance.
  %
  %   C++: class datamunge::XGBoostRegressor
  methods
    function obj = XGBoostRegressor (varargin)
      % XGBoostRegressor  Construct a C++ datamunge::XGBoostRegressor.
      %
      %   obj = datamunge.XGBoostRegressor (data, formula)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree)
      %   obj = datamunge.XGBoostRegressor (data, formula, n_trees, learning_rate, max_depth, lambda, alpha, gamma, min_child_weight, min_samples_leaf, subsample, colsample_bytree, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     n_trees  integer, default 100
      %     learning_rate  double, default 0.3
      %     max_depth  integer, default 6
      %     lambda  double, default 1.0
      %     alpha  double, default 0.0
      %     gamma  double, default 0.0
      %     min_child_weight  double, default 1.0
      %     min_samples_leaf  integer, default 1
      %     subsample  double, default 1.0
      %     colsample_bytree  double, default 1.0
      %     seed  integer, default 42
      %
      %   C++: datamunge::XGBoostRegressor::XGBoostRegressor(const DataFrame & data, const std::string & formula, std::size_t n_trees = 100, double learning_rate = 0.29999999999999999, std::size_t max_depth = 6, double lambda = 1.0, double alpha = 0.0, double gamma = 0.0, double min_child_weight = 1.0, std::size_t min_samples_leaf = 1, double subsample = 1.0, double colsample_bytree = 1.0, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2694, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::XGBoostRegressor::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::XGBoostRegressor::predictor_names() const
      r = __datamunge__(2680, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::XGBoostRegressor::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::XGBoostRegressor::observations() const
      r = __datamunge__(2681, obj);
    end
    function r = n_trees (obj)
      % n_trees  C++ datamunge::XGBoostRegressor::n_trees.
      %
      %   r = obj.n_trees ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::XGBoostRegressor::n_trees() const
      r = __datamunge__(2682, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::XGBoostRegressor::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::XGBoostRegressor::feature_importance() const
      r = __datamunge__(2683, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::XGBoostRegressor::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::XGBoostRegressor::fitted_values() const
      r = __datamunge__(2684, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::XGBoostRegressor::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::XGBoostRegressor::r_squared() const
      r = __datamunge__(2685, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::XGBoostRegressor::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::XGBoostRegressor::rmse() const
      r = __datamunge__(2686, obj);
    end
    function r = training_deviance (obj)
      % training_deviance  C++ datamunge::XGBoostRegressor::training_deviance.
      %
      %   r = obj.training_deviance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::XGBoostRegressor::training_deviance() const
      r = __datamunge__(2687, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::XGBoostRegressor::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::XGBoostRegressor::summary() const
      r = __datamunge__(2688, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::XGBoostRegressor::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::XGBoostRegressor::print_summary() const
      __datamunge__(2689, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::XGBoostRegressor::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::XGBoostRegressor::predict(const DataFrame & newdata) const
      r = __datamunge__(2690, obj, newdata);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::XGBoostRegressor::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::XGBoostRegressor::plot_predicted_vs_actual() const
      r = __datamunge__(2691, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::XGBoostRegressor::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::XGBoostRegressor::plot_residuals_vs_fitted() const
      r = __datamunge__(2692, obj);
    end
    function r = plot_training_deviance (obj)
      % plot_training_deviance  C++ datamunge::XGBoostRegressor::plot_training_deviance.
      %
      %   r = obj.plot_training_deviance ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::XGBoostRegressor::plot_training_deviance() const
      r = __datamunge__(2693, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::XGBoostRegressor. Do not edit.
