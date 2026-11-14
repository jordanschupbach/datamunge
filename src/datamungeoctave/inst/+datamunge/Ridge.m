classdef Ridge < datamunge.NativeObject
  % Ridge  SWIG-friendly facade for datamunge::stats::Ridge — pure L2-penalized ("ridge") regression, a special case of ElasticNet with alpha fixed to 0. Shrinks coefficients toward zero without ever zeroing them.
  %
  %   obj = datamunge.Ridge (data, formula)
  %   obj = datamunge.Ridge (data, formula, lambda)
  %   obj = datamunge.Ridge (data, formula, lambda, n_lambda)
  %   obj = datamunge.Ridge (data, formula, lambda, n_lambda, cv_folds)
  %   obj = datamunge.Ridge (data, formula, lambda, n_lambda, cv_folds, standardize)
  %   obj = datamunge.Ridge (data, formula, lambda, n_lambda, cv_folds, standardize, seed)
  %
  %   Methods:
  %     formula_text  C++ datamunge::Ridge::formula_text.
  %     has_intercept  C++ datamunge::Ridge::has_intercept.
  %     predictor_names  C++ datamunge::Ridge::predictor_names.
  %     observations  C++ datamunge::Ridge::observations.
  %     lambda  C++ datamunge::Ridge::lambda.
  %     lambda_was_selected  C++ datamunge::Ridge::lambda_was_selected.
  %     lambda_path  C++ datamunge::Ridge::lambda_path.
  %     cv_mean_squared_error  C++ datamunge::Ridge::cv_mean_squared_error.
  %     coefficients  C++ datamunge::Ridge::coefficients.
  %     intercept  C++ datamunge::Ridge::intercept.
  %     fitted_values  C++ datamunge::Ridge::fitted_values.
  %     residuals  C++ datamunge::Ridge::residuals.
  %     r_squared  C++ datamunge::Ridge::r_squared.
  %     rmse  C++ datamunge::Ridge::rmse.
  %     summary  C++ datamunge::Ridge::summary.
  %     print_summary  C++ datamunge::Ridge::print_summary.
  %     predict  C++ datamunge::Ridge::predict.
  %     plot_coefficient_path  C++ datamunge::Ridge::plot_coefficient_path.
  %     plot_cv_curve  C++ datamunge::Ridge::plot_cv_curve.
  %     plot_predicted_vs_actual  C++ datamunge::Ridge::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::Ridge::plot_residuals_vs_fitted.
  %
  %   C++: class datamunge::Ridge
  methods
    function obj = Ridge (varargin)
      % Ridge  Construct a C++ datamunge::Ridge.
      %
      %   obj = datamunge.Ridge (data, formula)
      %   obj = datamunge.Ridge (data, formula, lambda)
      %   obj = datamunge.Ridge (data, formula, lambda, n_lambda)
      %   obj = datamunge.Ridge (data, formula, lambda, n_lambda, cv_folds)
      %   obj = datamunge.Ridge (data, formula, lambda, n_lambda, cv_folds, standardize)
      %   obj = datamunge.Ridge (data, formula, lambda, n_lambda, cv_folds, standardize, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     lambda  double, default -1.0
      %     n_lambda  integer, default 100
      %     cv_folds  integer, default 5
      %     standardize  logical, default true
      %     seed  integer, default 42
      %
      %   C++: datamunge::Ridge::Ridge(const DataFrame & data, const std::string & formula, double lambda = -1.0, std::size_t n_lambda = 100, std::size_t cv_folds = 5, bool standardize = true, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2424, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::Ridge::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Ridge::formula_text() const
      r = __datamunge__(2403, obj);
    end
    function r = has_intercept (obj)
      % has_intercept  C++ datamunge::Ridge::has_intercept.
      %
      %   r = obj.has_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::Ridge::has_intercept() const
      r = __datamunge__(2404, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::Ridge::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::Ridge::predictor_names() const
      r = __datamunge__(2405, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::Ridge::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Ridge::observations() const
      r = __datamunge__(2406, obj);
    end
    function r = lambda (obj)
      % lambda  C++ datamunge::Ridge::lambda.
      %
      %   r = obj.lambda ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Ridge::lambda() const
      r = __datamunge__(2407, obj);
    end
    function r = lambda_was_selected (obj)
      % lambda_was_selected  C++ datamunge::Ridge::lambda_was_selected.
      %
      %   r = obj.lambda_was_selected ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::Ridge::lambda_was_selected() const
      r = __datamunge__(2408, obj);
    end
    function r = lambda_path (obj)
      % lambda_path  C++ datamunge::Ridge::lambda_path.
      %
      %   r = obj.lambda_path ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Ridge::lambda_path() const
      r = __datamunge__(2409, obj);
    end
    function r = cv_mean_squared_error (obj)
      % cv_mean_squared_error  C++ datamunge::Ridge::cv_mean_squared_error.
      %
      %   r = obj.cv_mean_squared_error ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Ridge::cv_mean_squared_error() const
      r = __datamunge__(2410, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::Ridge::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Ridge::coefficients() const
      r = __datamunge__(2411, obj);
    end
    function r = intercept (obj)
      % intercept  C++ datamunge::Ridge::intercept.
      %
      %   r = obj.intercept ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Ridge::intercept() const
      r = __datamunge__(2412, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::Ridge::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Ridge::fitted_values() const
      r = __datamunge__(2413, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::Ridge::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Ridge::residuals() const
      r = __datamunge__(2414, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::Ridge::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Ridge::r_squared() const
      r = __datamunge__(2415, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::Ridge::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Ridge::rmse() const
      r = __datamunge__(2416, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::Ridge::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Ridge::summary() const
      r = __datamunge__(2417, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::Ridge::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::Ridge::print_summary() const
      __datamunge__(2418, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::Ridge::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Ridge::predict(const DataFrame & newdata) const
      r = __datamunge__(2419, obj, newdata);
    end
    function r = plot_coefficient_path (obj)
      % plot_coefficient_path  C++ datamunge::Ridge::plot_coefficient_path.
      %
      %   r = obj.plot_coefficient_path ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Ridge::plot_coefficient_path() const
      r = __datamunge__(2420, obj);
    end
    function r = plot_cv_curve (obj)
      % plot_cv_curve  C++ datamunge::Ridge::plot_cv_curve.
      %
      %   r = obj.plot_cv_curve ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Ridge::plot_cv_curve() const
      r = __datamunge__(2421, obj);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::Ridge::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Ridge::plot_predicted_vs_actual() const
      r = __datamunge__(2422, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::Ridge::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Ridge::plot_residuals_vs_fitted() const
      r = __datamunge__(2423, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Ridge. Do not edit.
