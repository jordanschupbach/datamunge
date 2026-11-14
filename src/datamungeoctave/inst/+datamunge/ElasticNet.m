classdef ElasticNet < datamunge.NativeObject
  % ElasticNet  SWIG-friendly facade for datamunge::stats::ElasticNet — regularized linear regression fit by coordinate descent (alpha=0 is ridge, alpha=1 is lasso); see also the Ridge and Lasso convenience facades below.
  %
  %   obj = datamunge.ElasticNet (data, formula)
  %   obj = datamunge.ElasticNet (data, formula, alpha)
  %   obj = datamunge.ElasticNet (data, formula, alpha, lambda)
  %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda)
  %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda, cv_folds)
  %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda, cv_folds, standardize)
  %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
  %
  %   Methods:
  %     formula_text  C++ datamunge::ElasticNet::formula_text.
  %     has_intercept  C++ datamunge::ElasticNet::has_intercept.
  %     predictor_names  C++ datamunge::ElasticNet::predictor_names.
  %     observations  C++ datamunge::ElasticNet::observations.
  %     alpha  C++ datamunge::ElasticNet::alpha.
  %     lambda  The lambda actually used to produce coefficients() -- either the caller-supplied value or the cross-validation-selected one.
  %     lambda_was_selected  C++ datamunge::ElasticNet::lambda_was_selected.
  %     lambda_path  C++ datamunge::ElasticNet::lambda_path.
  %     cv_mean_squared_error  C++ datamunge::ElasticNet::cv_mean_squared_error.
  %     coefficients  C++ datamunge::ElasticNet::coefficients.
  %     intercept  C++ datamunge::ElasticNet::intercept.
  %     non_zero_coefficients  C++ datamunge::ElasticNet::non_zero_coefficients.
  %     fitted_values  C++ datamunge::ElasticNet::fitted_values.
  %     residuals  C++ datamunge::ElasticNet::residuals.
  %     r_squared  C++ datamunge::ElasticNet::r_squared.
  %     rmse  C++ datamunge::ElasticNet::rmse.
  %     summary  C++ datamunge::ElasticNet::summary.
  %     print_summary  C++ datamunge::ElasticNet::print_summary.
  %     predict  C++ datamunge::ElasticNet::predict.
  %     plot_coefficient_path  Coefficient trace (one series per predictor) across the lambda path; throws unless lambda was auto-selected.
  %     plot_cv_curve  Cross-validated MSE across the lambda path with the selected lambda marked; throws unless lambda was auto-selected.
  %     plot_predicted_vs_actual  C++ datamunge::ElasticNet::plot_predicted_vs_actual.
  %     plot_residuals_vs_fitted  C++ datamunge::ElasticNet::plot_residuals_vs_fitted.
  %
  %   C++: class datamunge::ElasticNet
  methods
    function obj = ElasticNet (varargin)
      % ElasticNet  Construct a C++ datamunge::ElasticNet.
      %
      %   obj = datamunge.ElasticNet (data, formula)
      %   obj = datamunge.ElasticNet (data, formula, alpha)
      %   obj = datamunge.ElasticNet (data, formula, alpha, lambda)
      %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda)
      %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda, cv_folds)
      %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda, cv_folds, standardize)
      %   obj = datamunge.ElasticNet (data, formula, alpha, lambda, n_lambda, cv_folds, standardize, seed)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     alpha  double, default 0.5: L1/L2 mixing: 0 = ridge, 1 = lasso, in between = elastic net.
      %     lambda  double, default -1.0: Regularization strength; pass a negative value (the default) to select it automatically via cross-validation.
      %     n_lambda  integer, default 100
      %     cv_folds  integer, default 5
      %     standardize  logical, default true
      %     seed  integer, default 42
      %
      %   C++: datamunge::ElasticNet::ElasticNet(const DataFrame & data, const std::string & formula, double alpha = 0.5, double lambda = -1.0, std::size_t n_lambda = 100, std::size_t cv_folds = 5, bool standardize = true, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2402, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::ElasticNet::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::ElasticNet::formula_text() const
      r = __datamunge__(2379, obj);
    end
    function r = has_intercept (obj)
      % has_intercept  C++ datamunge::ElasticNet::has_intercept.
      %
      %   r = obj.has_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::ElasticNet::has_intercept() const
      r = __datamunge__(2380, obj);
    end
    function r = predictor_names (obj)
      % predictor_names  C++ datamunge::ElasticNet::predictor_names.
      %
      %   r = obj.predictor_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::ElasticNet::predictor_names() const
      r = __datamunge__(2381, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::ElasticNet::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::ElasticNet::observations() const
      r = __datamunge__(2382, obj);
    end
    function r = alpha (obj)
      % alpha  C++ datamunge::ElasticNet::alpha.
      %
      %   r = obj.alpha ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::ElasticNet::alpha() const
      r = __datamunge__(2383, obj);
    end
    function r = lambda (obj)
      % lambda  The lambda actually used to produce coefficients() -- either the caller-supplied value or the cross-validation-selected one.
      %
      %   r = obj.lambda ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::ElasticNet::lambda() const
      r = __datamunge__(2384, obj);
    end
    function r = lambda_was_selected (obj)
      % lambda_was_selected  C++ datamunge::ElasticNet::lambda_was_selected.
      %
      %   r = obj.lambda_was_selected ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::ElasticNet::lambda_was_selected() const
      r = __datamunge__(2385, obj);
    end
    function r = lambda_path (obj)
      % lambda_path  C++ datamunge::ElasticNet::lambda_path.
      %
      %   r = obj.lambda_path ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ElasticNet::lambda_path() const
      r = __datamunge__(2386, obj);
    end
    function r = cv_mean_squared_error (obj)
      % cv_mean_squared_error  C++ datamunge::ElasticNet::cv_mean_squared_error.
      %
      %   r = obj.cv_mean_squared_error ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ElasticNet::cv_mean_squared_error() const
      r = __datamunge__(2387, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::ElasticNet::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ElasticNet::coefficients() const
      r = __datamunge__(2388, obj);
    end
    function r = intercept (obj)
      % intercept  C++ datamunge::ElasticNet::intercept.
      %
      %   r = obj.intercept ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::ElasticNet::intercept() const
      r = __datamunge__(2389, obj);
    end
    function r = non_zero_coefficients (obj)
      % non_zero_coefficients  C++ datamunge::ElasticNet::non_zero_coefficients.
      %
      %   r = obj.non_zero_coefficients ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::ElasticNet::non_zero_coefficients() const
      r = __datamunge__(2390, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::ElasticNet::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ElasticNet::fitted_values() const
      r = __datamunge__(2391, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::ElasticNet::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ElasticNet::residuals() const
      r = __datamunge__(2392, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::ElasticNet::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::ElasticNet::r_squared() const
      r = __datamunge__(2393, obj);
    end
    function r = rmse (obj)
      % rmse  C++ datamunge::ElasticNet::rmse.
      %
      %   r = obj.rmse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::ElasticNet::rmse() const
      r = __datamunge__(2394, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::ElasticNet::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::ElasticNet::summary() const
      r = __datamunge__(2395, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::ElasticNet::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::ElasticNet::print_summary() const
      __datamunge__(2396, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::ElasticNet::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::ElasticNet::predict(const DataFrame & newdata) const
      r = __datamunge__(2397, obj, newdata);
    end
    function r = plot_coefficient_path (obj)
      % plot_coefficient_path  Coefficient trace (one series per predictor) across the lambda path; throws unless lambda was auto-selected.
      %
      %   r = obj.plot_coefficient_path ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::ElasticNet::plot_coefficient_path() const
      r = __datamunge__(2398, obj);
    end
    function r = plot_cv_curve (obj)
      % plot_cv_curve  Cross-validated MSE across the lambda path with the selected lambda marked; throws unless lambda was auto-selected.
      %
      %   r = obj.plot_cv_curve ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::ElasticNet::plot_cv_curve() const
      r = __datamunge__(2399, obj);
    end
    function r = plot_predicted_vs_actual (obj)
      % plot_predicted_vs_actual  C++ datamunge::ElasticNet::plot_predicted_vs_actual.
      %
      %   r = obj.plot_predicted_vs_actual ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::ElasticNet::plot_predicted_vs_actual() const
      r = __datamunge__(2400, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::ElasticNet::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::ElasticNet::plot_residuals_vs_fitted() const
      r = __datamunge__(2401, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::ElasticNet. Do not edit.
