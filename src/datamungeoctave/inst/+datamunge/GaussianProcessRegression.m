classdef GaussianProcessRegression < datamunge.NativeObject
  % GaussianProcessRegression  SWIG-friendly facade for datamunge::stats::GaussianProcessRegression — exact Gaussian process regression with an RBF kernel, fit via Cholesky decomposition. Unlike every other regressor here, predictions come with a principled posterior confidence interval; see predict_frame().
  %
  %   obj = datamunge.GaussianProcessRegression (data, formula)
  %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale)
  %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio)
  %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio, n_length_scale_grid)
  %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio, n_length_scale_grid, n_noise_grid)
  %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio, n_length_scale_grid, n_noise_grid, standardize)
  %
  %   Methods:
  %     predictor_names  C++ datamunge::GaussianProcessRegression::predictor_names.
  %     observations  C++ datamunge::GaussianProcessRegression::observations.
  %     length_scale  C++ datamunge::GaussianProcessRegression::length_scale.
  %     signal_variance  C++ datamunge::GaussianProcessRegression::signal_variance.
  %     noise_variance  C++ datamunge::GaussianProcessRegression::noise_variance.
  %     log_marginal_likelihood  C++ datamunge::GaussianProcessRegression::log_marginal_likelihood.
  %     length_scale_was_selected  C++ datamunge::GaussianProcessRegression::length_scale_was_selected.
  %     noise_ratio_was_selected  C++ datamunge::GaussianProcessRegression::noise_ratio_was_selected.
  %     length_scale_grid  C++ datamunge::GaussianProcessRegression::length_scale_grid.
  %     length_scale_profile_log_likelihood  C++ datamunge::GaussianProcessRegression::length_scale_profile_log_likelihood.
  %     fitted_values  C++ datamunge::GaussianProcessRegression::fitted_values.
  %     r_squared  C++ datamunge::GaussianProcessRegression::r_squared.
  %     rmse  C++ datamunge::GaussianProcessRegression::rmse.
  %     summary  C++ datamunge::GaussianProcessRegression::summary.
  %     print_summary  C++ datamunge::GaussianProcessRegression::print_summary.
  %     predict  C++ datamunge::GaussianProcessRegression::predict.
  %     predict_frame  C++ datamunge::GaussianProcessRegression::predict_frame.
  %     plot_fit  C++ datamunge::GaussianProcessRegression::plot_fit.
  %     plot_predicted_vs_actual  C++ datamunge::GaussianProcessRegression::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::GaussianProcessRegression::plot_residuals_vs_fitted.
  %     plot_length_scale_profile  C++ datamunge::GaussianProcessRegression::plot_length_scale_profile.
  %
  %   C++: class datamunge::GaussianProcessRegression
  methods
    function obj = GaussianProcessRegression (varargin)
      % GaussianProcessRegression  Construct a C++ datamunge::GaussianProcessRegression.
      %
      %   obj = datamunge.GaussianProcessRegression (data, formula)
      %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale)
      %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio)
      %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio, n_length_scale_grid)
      %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio, n_length_scale_grid, n_noise_grid)
      %   obj = datamunge.GaussianProcessRegression (data, formula, length_scale, noise_ratio, n_length_scale_grid, n_noise_grid, standardize)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     length_scale  double, default -1.0: RBF kernel length scale in standardized-predictor units; pass a negative value (the default) to select it automatically by maximizing the log marginal likelihood.
      %     noise_ratio  double, default -1.0: noise_variance / signal_variance; pass a negative value (the default) to select it automatically the same way (0 is a legal fixed value: a noiseless/interpolating GP).
      %     n_length_scale_grid  integer, default 20
      %     n_noise_grid  integer, default 15
      %     standardize  logical, default true
      %
      %   C++: datamunge::GaussianProcessRegression::GaussianProcessRegression(const DataFrame & data, const std::string & formula, double length_scale = -1.0, double noise_ratio = -1.0, std::size_t n_length_scale_grid = 20, std::size_t n_noise_grid = 15, bool standardize = true)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2733, false, varargin{:});
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::GaussianProcessRegression::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GaussianProcessRegression::predictor_names() const
      r = __datamunge__(2712, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::GaussianProcessRegression::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GaussianProcessRegression::observations() const
      r = __datamunge__(2713, obj);
    end
    function r = length_scale (obj)
      % length_scale  C++ datamunge::GaussianProcessRegression::length_scale.
      %
      %   r = obj.length_scale ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GaussianProcessRegression::length_scale() const
      r = __datamunge__(2714, obj);
    end
    function r = signal_variance (obj)
      % signal_variance  C++ datamunge::GaussianProcessRegression::signal_variance.
      %
      %   r = obj.signal_variance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GaussianProcessRegression::signal_variance() const
      r = __datamunge__(2715, obj);
    end
    function r = noise_variance (obj)
      % noise_variance  C++ datamunge::GaussianProcessRegression::noise_variance.
      %
      %   r = obj.noise_variance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GaussianProcessRegression::noise_variance() const
      r = __datamunge__(2716, obj);
    end
    function r = log_marginal_likelihood (obj)
      % log_marginal_likelihood  C++ datamunge::GaussianProcessRegression::log_marginal_likelihood.
      %
      %   r = obj.log_marginal_likelihood ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GaussianProcessRegression::log_marginal_likelihood() const
      r = __datamunge__(2717, obj);
    end
    function r = length_scale_was_selected (obj)
      % length_scale_was_selected  C++ datamunge::GaussianProcessRegression::length_scale_was_selected.
      %
      %   r = obj.length_scale_was_selected ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::GaussianProcessRegression::length_scale_was_selected() const
      r = __datamunge__(2718, obj);
    end
    function r = noise_ratio_was_selected (obj)
      % noise_ratio_was_selected  C++ datamunge::GaussianProcessRegression::noise_ratio_was_selected.
      %
      %   r = obj.noise_ratio_was_selected ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::GaussianProcessRegression::noise_ratio_was_selected() const
      r = __datamunge__(2719, obj);
    end
    function r = length_scale_grid (obj)
      % length_scale_grid  C++ datamunge::GaussianProcessRegression::length_scale_grid.
      %
      %   r = obj.length_scale_grid ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GaussianProcessRegression::length_scale_grid() const
      r = __datamunge__(2720, obj);
    end
    function r = length_scale_profile_log_likelihood (obj)
      % length_scale_profile_log_likelihood  C++ datamunge::GaussianProcessRegression::length_scale_profile_log_likelihood.
      %
      %   r = obj.length_scale_profile_log_likelihood ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GaussianProcessRegression::length_scale_profile_log_likelihood() const
      r = __datamunge__(2721, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::GaussianProcessRegression::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GaussianProcessRegression::fitted_values() const
      r = __datamunge__(2722, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::GaussianProcessRegression::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GaussianProcessRegression::r_squared() const
      r = __datamunge__(2723, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::GaussianProcessRegression::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GaussianProcessRegression::rmse() const
      r = __datamunge__(2724, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::GaussianProcessRegression::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GaussianProcessRegression::summary() const
      r = __datamunge__(2725, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::GaussianProcessRegression::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::GaussianProcessRegression::print_summary() const
      __datamunge__(2726, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::GaussianProcessRegression::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GaussianProcessRegression::predict(const DataFrame & newdata) const
      r = __datamunge__(2727, obj, newdata);
    end
    function r = predict_frame (obj, varargin)
      % predict_frame  C++ datamunge::GaussianProcessRegression::predict_frame.
      %
      %   r = obj.predict_frame (newdata)
      %   r = obj.predict_frame (newdata, interval_kind)
      %   r = obj.predict_frame (newdata, interval_kind, level)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %     interval_kind  char row, default 'none': One of "none" or "confidence". Returned DataFrame has a "fit" column, plus "se_fit"/"lwr"/"upr" when an interval is requested.
      %     level  double, default 0.95
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::GaussianProcessRegression::predict_frame(const DataFrame & newdata, const std::string & interval_kind = "none", double level = 0.94999999999999996) const
      r = __datamunge__(2728, obj, varargin{:});
    end
    function r = plot_fit (obj, varargin)
      % plot_fit  C++ datamunge::GaussianProcessRegression::plot_fit.
      %
      %   r = obj.plot_fit (data)
      %   r = obj.plot_fit (data, grid_resolution)
      %   r = obj.plot_fit (data, grid_resolution, level)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     grid_resolution  integer, default 200
      %     level  double, default 0.95
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GaussianProcessRegression::plot_fit(const DataFrame & data, std::size_t grid_resolution = 200, double level = 0.94999999999999996) const
      r = __datamunge__(2729, obj, varargin{:});
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::GaussianProcessRegression::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GaussianProcessRegression::plot_predicted_vs_actual() const
      r = __datamunge__(2730, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::GaussianProcessRegression::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GaussianProcessRegression::plot_residuals_vs_fitted() const
      r = __datamunge__(2731, obj);
    end
    function r = plot_length_scale_profile (obj)
      % plot_length_scale_profile  C++ datamunge::GaussianProcessRegression::plot_length_scale_profile.
      %
      %   r = obj.plot_length_scale_profile ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GaussianProcessRegression::plot_length_scale_profile() const
      r = __datamunge__(2732, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::GaussianProcessRegression. Do not edit.
