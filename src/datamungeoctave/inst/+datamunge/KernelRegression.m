classdef KernelRegression < datamunge.NativeObject
  % KernelRegression  SWIG-friendly facade for datamunge::stats::KernelRegression — Nadaraya-Watson kernel regression, a nonparametric fit where each prediction is a kernel-weighted average of training responses.
  %
  %   obj = datamunge.KernelRegression (data, formula)
  %   obj = datamunge.KernelRegression (data, formula, kernel)
  %   obj = datamunge.KernelRegression (data, formula, kernel, bandwidth)
  %   obj = datamunge.KernelRegression (data, formula, kernel, bandwidth, n_bandwidth)
  %   obj = datamunge.KernelRegression (data, formula, kernel, bandwidth, n_bandwidth, standardize)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::KernelRegression::predictor_names.
  %     observations  C++ datamunge::KernelRegression::observations.
  %     bandwidth  The bandwidth actually used -- either the caller-supplied value or the cross-validation-selected one.
  %     bandwidth_was_selected  C++ datamunge::KernelRegression::bandwidth_was_selected.
  %     bandwidth_grid  C++ datamunge::KernelRegression::bandwidth_grid.
  %     cv_mean_squared_error  C++ datamunge::KernelRegression::cv_mean_squared_error.
  %     fitted_values  C++ datamunge::KernelRegression::fitted_values.
  %     r_squared  C++ datamunge::KernelRegression::r_squared.
  %     rmse  C++ datamunge::KernelRegression::rmse.
  %     summary  C++ datamunge::KernelRegression::summary.
  %     print_summary  C++ datamunge::KernelRegression::print_summary.
  %     predict  C++ datamunge::KernelRegression::predict.
  %     plot_fit  Scatter of `data` plus the fitted kernel-regression curve; only valid for a single-predictor model.
  %     plot_predicted_vs_actual  C++ datamunge::KernelRegression::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::KernelRegression::plot_residuals_vs_fitted.
  %     plot_cv_curve  C++ datamunge::KernelRegression::plot_cv_curve.
  %
  %   C++: class datamunge::KernelRegression
  methods
    function obj = KernelRegression (varargin)
      % KernelRegression  Construct a C++ datamunge::KernelRegression.
      %
      %   obj = datamunge.KernelRegression (data, formula)
      %   obj = datamunge.KernelRegression (data, formula, kernel)
      %   obj = datamunge.KernelRegression (data, formula, kernel, bandwidth)
      %   obj = datamunge.KernelRegression (data, formula, kernel, bandwidth, n_bandwidth)
      %   obj = datamunge.KernelRegression (data, formula, kernel, bandwidth, n_bandwidth, standardize)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     kernel  char row, default 'gaussian': "gaussian" (default), "epanechnikov", "uniform", or "triangular".
      %     bandwidth  double, default -1.0: Bandwidth in standardized-predictor units; pass a negative value (the default) to select it automatically via leave-one-out cross-validation.
      %     n_bandwidth  integer, default 50
      %     standardize  logical, default true
      %
      %   C++: datamunge::KernelRegression::KernelRegression(const DataFrame & data, const std::string & formula, const std::string & kernel = "gaussian", double bandwidth = -1.0, std::size_t n_bandwidth = 50, bool standardize = true)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2711, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::KernelRegression::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::KernelRegression::predictor_names() const
      r = __datamunge__(2695, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::KernelRegression::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::KernelRegression::observations() const
      r = __datamunge__(2696, obj);
    end
    function r = bandwidth (obj)
      % bandwidth  The bandwidth actually used -- either the caller-supplied value or the cross-validation-selected one.
      %
      %   r = obj.bandwidth ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KernelRegression::bandwidth() const
      r = __datamunge__(2697, obj);
    end
    function r = bandwidth_was_selected (obj)
      % bandwidth_was_selected  C++ datamunge::KernelRegression::bandwidth_was_selected.
      %
      %   r = obj.bandwidth_was_selected ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::KernelRegression::bandwidth_was_selected() const
      r = __datamunge__(2698, obj);
    end
    function r = bandwidth_grid (obj)
      % bandwidth_grid  C++ datamunge::KernelRegression::bandwidth_grid.
      %
      %   r = obj.bandwidth_grid ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KernelRegression::bandwidth_grid() const
      r = __datamunge__(2699, obj);
    end
    function r = cv_mean_squared_error (obj)
      % cv_mean_squared_error  C++ datamunge::KernelRegression::cv_mean_squared_error.
      %
      %   r = obj.cv_mean_squared_error ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KernelRegression::cv_mean_squared_error() const
      r = __datamunge__(2700, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::KernelRegression::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KernelRegression::fitted_values() const
      r = __datamunge__(2701, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::KernelRegression::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KernelRegression::r_squared() const
      r = __datamunge__(2702, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::KernelRegression::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::KernelRegression::rmse() const
      r = __datamunge__(2703, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::KernelRegression::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::KernelRegression::summary() const
      r = __datamunge__(2704, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::KernelRegression::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::KernelRegression::print_summary() const
      __datamunge__(2705, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::KernelRegression::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::KernelRegression::predict(const DataFrame & newdata) const
      r = __datamunge__(2706, obj, newdata);
    end
    function r = plot_fit (obj, varargin)
      % plot_fit  Scatter of `data` plus the fitted kernel-regression curve; only valid for a single-predictor model.
      %
      %   r = obj.plot_fit (data)
      %   r = obj.plot_fit (data, grid_resolution)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     grid_resolution  integer, default 200
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::KernelRegression::plot_fit(const DataFrame & data, std::size_t grid_resolution = 200) const
      r = __datamunge__(2707, obj, varargin{:});
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::KernelRegression::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::KernelRegression::plot_predicted_vs_actual() const
      r = __datamunge__(2708, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::KernelRegression::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::KernelRegression::plot_residuals_vs_fitted() const
      r = __datamunge__(2709, obj);
    end
    function r = plot_cv_curve (obj)
      % plot_cv_curve  C++ datamunge::KernelRegression::plot_cv_curve.
      %
      %   r = obj.plot_cv_curve ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::KernelRegression::plot_cv_curve() const
      r = __datamunge__(2710, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::KernelRegression. Do not edit.
