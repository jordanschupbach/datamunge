classdef KNNRegressor < datamunge.NativeObject
  % KNNRegressor  SWIG-friendly facade for datamunge::stats::KNNRegressor — k-nearest-neighbors regression. As with KNNClassifier, fitted_values()/r_squared()/rmse() report leave-one-out performance.
  %
  %   obj = datamunge.KNNRegressor (data, formula)
  %   obj = datamunge.KNNRegressor (data, formula, k)
  %   obj = datamunge.KNNRegressor (data, formula, k, metric)
  %   obj = datamunge.KNNRegressor (data, formula, k, metric, weighted)
  %   obj = datamunge.KNNRegressor (data, formula, k, metric, weighted, standardize)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::KNNRegressor::predictor_names.
  %     observations  C++ datamunge::KNNRegressor::observations.
  %     k  C++ datamunge::KNNRegressor::k.
  %     fitted_values  C++ datamunge::KNNRegressor::fitted_values.
  %     r_squared  C++ datamunge::KNNRegressor::r_squared.
  %     rmse  C++ datamunge::KNNRegressor::rmse.
  %     summary  C++ datamunge::KNNRegressor::summary.
  %     print_summary  C++ datamunge::KNNRegressor::print_summary.
  %     predict  C++ datamunge::KNNRegressor::predict.
  %     plot_predicted_vs_actual  C++ datamunge::KNNRegressor::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::KNNRegressor::plot_residuals_vs_fitted.
  %
  %   C++: class datamunge::KNNRegressor
  methods
    function obj = KNNRegressor (varargin)
      % KNNRegressor  Construct a C++ datamunge::KNNRegressor.
      %
      %   obj = datamunge.KNNRegressor (data, formula)
      %   obj = datamunge.KNNRegressor (data, formula, k)
      %   obj = datamunge.KNNRegressor (data, formula, k, metric)
      %   obj = datamunge.KNNRegressor (data, formula, k, metric, weighted)
      %   obj = datamunge.KNNRegressor (data, formula, k, metric, weighted, standardize)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     k  integer, default 5
      %     metric  char row, default 'euclidean': "euclidean" (default) or "manhattan".
      %     weighted  logical, default false
      %     standardize  logical, default true
      %
      %   C++: datamunge::KNNRegressor::KNNRegressor(const DataFrame & data, const std::string & formula, std::size_t k = 5, const std::string & metric = "euclidean", bool weighted = false, bool standardize = true)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2472, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::KNNRegressor::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KNNRegressor::predictor_names() const
      r = __datamunge__(2461, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::KNNRegressor::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KNNRegressor::observations() const
      r = __datamunge__(2462, obj);
    end
    function r = k (obj)
      % k  C++ datamunge::KNNRegressor::k.
      %
      %   r = obj.k ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KNNRegressor::k() const
      r = __datamunge__(2463, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::KNNRegressor::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KNNRegressor::fitted_values() const
      r = __datamunge__(2464, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::KNNRegressor::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KNNRegressor::r_squared() const
      r = __datamunge__(2465, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::KNNRegressor::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KNNRegressor::rmse() const
      r = __datamunge__(2466, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::KNNRegressor::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::KNNRegressor::summary() const
      r = __datamunge__(2467, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::KNNRegressor::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::KNNRegressor::print_summary() const
      __datamunge__(2468, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::KNNRegressor::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KNNRegressor::predict(const DataFrame & newdata) const
      r = __datamunge__(2469, obj, newdata);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::KNNRegressor::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::KNNRegressor::plot_predicted_vs_actual() const
      r = __datamunge__(2470, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::KNNRegressor::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::KNNRegressor::plot_residuals_vs_fitted() const
      r = __datamunge__(2471, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::KNNRegressor. Do not edit.
