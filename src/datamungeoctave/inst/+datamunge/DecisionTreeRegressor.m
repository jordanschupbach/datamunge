classdef DecisionTreeRegressor < datamunge.NativeObject
  % DecisionTreeRegressor  SWIG-friendly facade for datamunge::stats::DecisionTreeRegressor — a CART-style regression tree.
  %
  %   obj = datamunge.DecisionTreeRegressor (data, formula)
  %   obj = datamunge.DecisionTreeRegressor (data, formula, max_depth)
  %   obj = datamunge.DecisionTreeRegressor (data, formula, max_depth, min_samples_split)
  %   obj = datamunge.DecisionTreeRegressor (data, formula, max_depth, min_samples_split, min_samples_leaf)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::DecisionTreeRegressor::predictor_names.
  %     observations  C++ datamunge::DecisionTreeRegressor::observations.
  %     node_count  C++ datamunge::DecisionTreeRegressor::node_count.
  %     leaf_count  C++ datamunge::DecisionTreeRegressor::leaf_count.
  %     depth  C++ datamunge::DecisionTreeRegressor::depth.
  %     feature_importance  C++ datamunge::DecisionTreeRegressor::feature_importance.
  %     fitted_values  C++ datamunge::DecisionTreeRegressor::fitted_values.
  %     r_squared  C++ datamunge::DecisionTreeRegressor::r_squared.
  %     rmse  C++ datamunge::DecisionTreeRegressor::rmse.
  %     summary  C++ datamunge::DecisionTreeRegressor::summary.
  %     print_summary  C++ datamunge::DecisionTreeRegressor::print_summary.
  %     predict  C++ datamunge::DecisionTreeRegressor::predict.
  %     plot_predicted_vs_actual  C++ datamunge::DecisionTreeRegressor::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::DecisionTreeRegressor::plot_residuals_vs_fitted.
  %
  %   C++: class datamunge::DecisionTreeRegressor
  methods
    function obj = DecisionTreeRegressor (varargin)
      % DecisionTreeRegressor  Construct a C++ datamunge::DecisionTreeRegressor.
      %
      %   obj = datamunge.DecisionTreeRegressor (data, formula)
      %   obj = datamunge.DecisionTreeRegressor (data, formula, max_depth)
      %   obj = datamunge.DecisionTreeRegressor (data, formula, max_depth, min_samples_split)
      %   obj = datamunge.DecisionTreeRegressor (data, formula, max_depth, min_samples_split, min_samples_leaf)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     max_depth  integer, default 5
      %     min_samples_split  integer, default 2
      %     min_samples_leaf  integer, default 1
      %
      %   C++: datamunge::DecisionTreeRegressor::DecisionTreeRegressor(const DataFrame & data, const std::string & formula, std::size_t max_depth = 5, std::size_t min_samples_split = 2, std::size_t min_samples_leaf = 1)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2346, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::DecisionTreeRegressor::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::DecisionTreeRegressor::predictor_names() const
      r = __datamunge__(2332, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::DecisionTreeRegressor::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeRegressor::observations() const
      r = __datamunge__(2333, obj);
    end
    function r = node_count (obj)
      % node_count  C++ datamunge::DecisionTreeRegressor::node_count.
      %
      %   r = obj.node_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeRegressor::node_count() const
      r = __datamunge__(2334, obj);
    end
    function r = leaf_count (obj)
      % leaf_count  C++ datamunge::DecisionTreeRegressor::leaf_count.
      %
      %   r = obj.leaf_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeRegressor::leaf_count() const
      r = __datamunge__(2335, obj);
    end
    function r = depth (obj)
      % depth  C++ datamunge::DecisionTreeRegressor::depth.
      %
      %   r = obj.depth ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::DecisionTreeRegressor::depth() const
      r = __datamunge__(2336, obj);
    end
    function r = feature_importance (obj)
      % feature_importance  C++ datamunge::DecisionTreeRegressor::feature_importance.
      %
      %   r = obj.feature_importance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DecisionTreeRegressor::feature_importance() const
      r = __datamunge__(2337, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::DecisionTreeRegressor::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DecisionTreeRegressor::fitted_values() const
      r = __datamunge__(2338, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::DecisionTreeRegressor::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DecisionTreeRegressor::r_squared() const
      r = __datamunge__(2339, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::DecisionTreeRegressor::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::DecisionTreeRegressor::rmse() const
      r = __datamunge__(2340, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::DecisionTreeRegressor::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::DecisionTreeRegressor::summary() const
      r = __datamunge__(2341, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::DecisionTreeRegressor::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::DecisionTreeRegressor::print_summary() const
      __datamunge__(2342, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::DecisionTreeRegressor::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::DecisionTreeRegressor::predict(const DataFrame & newdata) const
      r = __datamunge__(2343, obj, newdata);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::DecisionTreeRegressor::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::DecisionTreeRegressor::plot_predicted_vs_actual() const
      r = __datamunge__(2344, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::DecisionTreeRegressor::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::DecisionTreeRegressor::plot_residuals_vs_fitted() const
      r = __datamunge__(2345, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::DecisionTreeRegressor. Do not edit.
