classdef Lasso < datamunge.NativeObject
  % Lasso  SWIG-friendly facade for datamunge::stats::Lasso — pure L1-penalized ("lasso") regression, a special case of ElasticNet with alpha fixed to 1. Can shrink coefficients exactly to zero, performing variable selection.
  %
  %   obj = datamunge.Lasso (data, formula)
  %   obj = datamunge.Lasso (data, formula, lambda)
  %   obj = datamunge.Lasso (data, formula, lambda, n_lambda)
  %   obj = datamunge.Lasso (data, formula, lambda, n_lambda, cv_folds)
  %   obj = datamunge.Lasso (data, formula, lambda, n_lambda, cv_folds, standardize)
  %   obj = datamunge.Lasso (data, formula, lambda, n_lambda, cv_folds, standardize, seed)
  %
  %   Methods:
  %     formula_text  C++ datamunge::Lasso::formula_text.
  %     has_intercept  C++ datamunge::Lasso::has_intercept.
  %     predictor_names  C++ datamunge::Lasso::predictor_names.
  %     observations  C++ datamunge::Lasso::observations.
  %     lambda  C++ datamunge::Lasso::lambda.
  %     lambda_was_selected  C++ datamunge::Lasso::lambda_was_selected.
  %     lambda_path  C++ datamunge::Lasso::lambda_path.
  %     cv_mean_squared_error  C++ datamunge::Lasso::cv_mean_squared_error.
  %     coefficients  C++ datamunge::Lasso::coefficients.
  %     intercept  C++ datamunge::Lasso::intercept.
  %     non_zero_coefficients  C++ datamunge::Lasso::non_zero_coefficients.
  %     fitted_values  C++ datamunge::Lasso::fitted_values.
  %     residuals  C++ datamunge::Lasso::residuals.
  %     r_squared  C++ datamunge::Lasso::r_squared.
  %     rmse  C++ datamunge::Lasso::rmse.
  %     summary  C++ datamunge::Lasso::summary.
  %     print_summary  C++ datamunge::Lasso::print_summary.
  %     predict  C++ datamunge::Lasso::predict.
  %     plot_coefficient_path  C++ datamunge::Lasso::plot_coefficient_path.
  %     plot_cv_curve  C++ datamunge::Lasso::plot_cv_curve.
  %     plot_predicted_vs_actual  C++ datamunge::Lasso::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::Lasso::plot_residuals_vs_fitted.
  %
  %   C++: class datamunge::Lasso
  methods
    function obj = Lasso (varargin)
      % Lasso  Construct a C++ datamunge::Lasso.
      %
      %   obj = datamunge.Lasso (data, formula)
      %   obj = datamunge.Lasso (data, formula, lambda)
      %   obj = datamunge.Lasso (data, formula, lambda, n_lambda)
      %   obj = datamunge.Lasso (data, formula, lambda, n_lambda, cv_folds)
      %   obj = datamunge.Lasso (data, formula, lambda, n_lambda, cv_folds, standardize)
      %   obj = datamunge.Lasso (data, formula, lambda, n_lambda, cv_folds, standardize, seed)
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
      %   C++: datamunge::Lasso::Lasso(const DataFrame & data, const std::string & formula, double lambda = -1.0, std::size_t n_lambda = 100, std::size_t cv_folds = 5, bool standardize = true, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2447, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::Lasso::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Lasso::formula_text() const
      r = __datamunge__(2425, obj);
    end
    function r = has_intercept (obj)
      % has_intercept  C++ datamunge::Lasso::has_intercept.
      %
      %   r = obj.has_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::Lasso::has_intercept() const
      r = __datamunge__(2426, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::Lasso::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::Lasso::predictor_names() const
      r = __datamunge__(2427, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::Lasso::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Lasso::observations() const
      r = __datamunge__(2428, obj);
    end
    function r = lambda (obj)
      % lambda  C++ datamunge::Lasso::lambda.
      %
      %   r = obj.lambda ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Lasso::lambda() const
      r = __datamunge__(2429, obj);
    end
    function r = lambda_was_selected (obj)
      % lambda_was_selected  C++ datamunge::Lasso::lambda_was_selected.
      %
      %   r = obj.lambda_was_selected ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::Lasso::lambda_was_selected() const
      r = __datamunge__(2430, obj);
    end
    function r = lambda_path (obj)
      % lambda_path  C++ datamunge::Lasso::lambda_path.
      %
      %   r = obj.lambda_path ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Lasso::lambda_path() const
      r = __datamunge__(2431, obj);
    end
    function r = cv_mean_squared_error (obj)
      % cv_mean_squared_error  C++ datamunge::Lasso::cv_mean_squared_error.
      %
      %   r = obj.cv_mean_squared_error ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Lasso::cv_mean_squared_error() const
      r = __datamunge__(2432, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::Lasso::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Lasso::coefficients() const
      r = __datamunge__(2433, obj);
    end
    function r = intercept (obj)
      % intercept  C++ datamunge::Lasso::intercept.
      %
      %   r = obj.intercept ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Lasso::intercept() const
      r = __datamunge__(2434, obj);
    end
    function r = non_zero_coefficients (obj)
      % non_zero_coefficients  C++ datamunge::Lasso::non_zero_coefficients.
      %
      %   r = obj.non_zero_coefficients ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Lasso::non_zero_coefficients() const
      r = __datamunge__(2435, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::Lasso::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Lasso::fitted_values() const
      r = __datamunge__(2436, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::Lasso::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Lasso::residuals() const
      r = __datamunge__(2437, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::Lasso::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Lasso::r_squared() const
      r = __datamunge__(2438, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::Lasso::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Lasso::rmse() const
      r = __datamunge__(2439, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::Lasso::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Lasso::summary() const
      r = __datamunge__(2440, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::Lasso::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::Lasso::print_summary() const
      __datamunge__(2441, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::Lasso::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::Lasso::predict(const DataFrame & newdata) const
      r = __datamunge__(2442, obj, newdata);
    end
    function r = plot_coefficient_path (obj)
      % plot_coefficient_path  C++ datamunge::Lasso::plot_coefficient_path.
      %
      %   r = obj.plot_coefficient_path ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Lasso::plot_coefficient_path() const
      r = __datamunge__(2443, obj);
    end
    function r = plot_cv_curve (obj)
      % plot_cv_curve  C++ datamunge::Lasso::plot_cv_curve.
      %
      %   r = obj.plot_cv_curve ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Lasso::plot_cv_curve() const
      r = __datamunge__(2444, obj);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::Lasso::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Lasso::plot_predicted_vs_actual() const
      r = __datamunge__(2445, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::Lasso::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::Lasso::plot_residuals_vs_fitted() const
      r = __datamunge__(2446, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Lasso. Do not edit.
