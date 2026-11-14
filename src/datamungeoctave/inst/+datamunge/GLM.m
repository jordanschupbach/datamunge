classdef GLM < datamunge.NativeObject
  % GLM  SWIG-friendly facade for datamunge::stats::GLM — a generalized linear model (gaussian, binomial, poisson, or Gamma family with its canonical link, matching R's glm() defaults) fit by iteratively reweighted least squares.
  %
  %   obj = datamunge.GLM (data, formula)
  %   obj = datamunge.GLM (data, formula, family)
  %   obj = datamunge.GLM (data, formula, family, weights_column)
  %   obj = datamunge.GLM (data, formula, family, weights_column, max_iter)
  %   obj = datamunge.GLM (data, formula, family, weights_column, max_iter, tol)
  %
  %   Methods:
  %     formula_text  C++ datamunge::GLM::formula_text.
  %     family  C++ datamunge::GLM::family.
  %     has_intercept  C++ datamunge::GLM::has_intercept.
  %     observations  C++ datamunge::GLM::observations.
  %     rank  C++ datamunge::GLM::rank.
  %     degrees_of_freedom  C++ datamunge::GLM::degrees_of_freedom.
  %     coefficients  C++ datamunge::GLM::coefficients.
  %     coefficient_names  C++ datamunge::GLM::coefficient_names.
  %     fitted_values  C++ datamunge::GLM::fitted_values.
  %     linear_predictors  C++ datamunge::GLM::linear_predictors.
  %     residuals  C++ datamunge::GLM::residuals.
  %     pearson_residuals  C++ datamunge::GLM::pearson_residuals.
  %     standardized_residuals  C++ datamunge::GLM::standardized_residuals.
  %     leverage  C++ datamunge::GLM::leverage.
  %     standard_errors  C++ datamunge::GLM::standard_errors.
  %     test_statistics  C++ datamunge::GLM::test_statistics.
  %     p_values  C++ datamunge::GLM::p_values.
  %     deviance  C++ datamunge::GLM::deviance.
  %     null_deviance  C++ datamunge::GLM::null_deviance.
  %     dispersion  C++ datamunge::GLM::dispersion.
  %     aic  C++ datamunge::GLM::aic.
  %     confidence_interval_lower  C++ datamunge::GLM::confidence_interval_lower.
  %     confidence_interval_upper  C++ datamunge::GLM::confidence_interval_upper.
  %     summary  C++ datamunge::GLM::summary.
  %     print_summary  C++ datamunge::GLM::print_summary.
  %     predict  Response-scale predictions (back-transformed through the inverse link).
  %     predict_frame  C++ datamunge::GLM::predict_frame.
  %     plot_residuals_vs_fitted  C++ datamunge::GLM::plot_residuals_vs_fitted.
  %     plot_normal_qq  C++ datamunge::GLM::plot_normal_qq.
  %     plot_scale_location  C++ datamunge::GLM::plot_scale_location.
  %     plot_residuals_vs_leverage  C++ datamunge::GLM::plot_residuals_vs_leverage.
  %     save_diagnostic_plots  C++ datamunge::GLM::save_diagnostic_plots.
  %
  %   C++: class datamunge::GLM
  methods
    function obj = GLM (varargin)
      % GLM  Construct a C++ datamunge::GLM.
      %
      %   obj = datamunge.GLM (data, formula)
      %   obj = datamunge.GLM (data, formula, family)
      %   obj = datamunge.GLM (data, formula, family, weights_column)
      %   obj = datamunge.GLM (data, formula, family, weights_column, max_iter)
      %   obj = datamunge.GLM (data, formula, family, weights_column, max_iter, tol)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     family  char row, default 'gaussian': One of "gaussian" (identity link), "binomial" (logit link), "poisson" (log link), or "Gamma" (inverse link).
      %     weights_column  char row, default '': Optional column name enabling prior weights; pass "" (the default) for none.
      %     max_iter  integer, default 25
      %     tol  double, default 1e-08
      %
      %   C++: datamunge::GLM::GLM(const DataFrame & data, const std::string & formula, const std::string & family = "gaussian", const std::string & weights_column = "", std::size_t max_iter = 25, double tol = 1.0E-8)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2779, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::GLM::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLM::formula_text() const
      r = __datamunge__(2747, obj);
    end
    function r = family (obj)
      % family  C++ datamunge::GLM::family.
      %
      %   r = obj.family ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLM::family() const
      r = __datamunge__(2748, obj);
    end
    function r = has_intercept (obj)
      % has_intercept  C++ datamunge::GLM::has_intercept.
      %
      %   r = obj.has_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::GLM::has_intercept() const
      r = __datamunge__(2749, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::GLM::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLM::observations() const
      r = __datamunge__(2750, obj);
    end
    function r = rank (obj)
      % rank  C++ datamunge::GLM::rank.
      %
      %   r = obj.rank ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLM::rank() const
      r = __datamunge__(2751, obj);
    end
    function r = degrees_of_freedom (obj)
      % degrees_of_freedom  C++ datamunge::GLM::degrees_of_freedom.
      %
      %   r = obj.degrees_of_freedom ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::GLM::degrees_of_freedom() const
      r = __datamunge__(2752, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::GLM::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::coefficients() const
      r = __datamunge__(2753, obj);
    end
    function r = coefficient_names (obj)
      % coefficient_names  C++ datamunge::GLM::coefficient_names.
      %
      %   r = obj.coefficient_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::GLM::coefficient_names() const
      r = __datamunge__(2754, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::GLM::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::fitted_values() const
      r = __datamunge__(2755, obj);
    end
    function r = linear_predictors (obj)
      % linear_predictors  C++ datamunge::GLM::linear_predictors.
      %
      %   r = obj.linear_predictors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::linear_predictors() const
      r = __datamunge__(2756, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::GLM::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::residuals() const
      r = __datamunge__(2757, obj);
    end
    function r = pearson_residuals (obj)
      % pearson_residuals  C++ datamunge::GLM::pearson_residuals.
      %
      %   r = obj.pearson_residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::pearson_residuals() const
      r = __datamunge__(2758, obj);
    end
    function r = standardized_residuals (obj)
      % standardized_residuals  C++ datamunge::GLM::standardized_residuals.
      %
      %   r = obj.standardized_residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::standardized_residuals() const
      r = __datamunge__(2759, obj);
    end
    function r = leverage (obj)
      % leverage  C++ datamunge::GLM::leverage.
      %
      %   r = obj.leverage ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::leverage() const
      r = __datamunge__(2760, obj);
    end
    function r = standard_errors (obj)
      % standard_errors  C++ datamunge::GLM::standard_errors.
      %
      %   r = obj.standard_errors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::standard_errors() const
      r = __datamunge__(2761, obj);
    end
    function r = test_statistics (obj)
      % test_statistics  C++ datamunge::GLM::test_statistics.
      %
      %   r = obj.test_statistics ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::test_statistics() const
      r = __datamunge__(2762, obj);
    end
    function r = p_values (obj)
      % p_values  C++ datamunge::GLM::p_values.
      %
      %   r = obj.p_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::p_values() const
      r = __datamunge__(2763, obj);
    end
    function r = deviance (obj)
      % deviance  C++ datamunge::GLM::deviance.
      %
      %   r = obj.deviance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLM::deviance() const
      r = __datamunge__(2764, obj);
    end
    function r = null_deviance (obj)
      % null_deviance  C++ datamunge::GLM::null_deviance.
      %
      %   r = obj.null_deviance ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLM::null_deviance() const
      r = __datamunge__(2765, obj);
    end
    function r = dispersion (obj)
      % dispersion  C++ datamunge::GLM::dispersion.
      %
      %   r = obj.dispersion ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLM::dispersion() const
      r = __datamunge__(2766, obj);
    end
    function r = aic (obj)
      % aic  C++ datamunge::GLM::aic.
      %
      %   r = obj.aic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::GLM::aic() const
      r = __datamunge__(2767, obj);
    end
    function r = confidence_interval_lower (obj, varargin)
      % confidence_interval_lower  C++ datamunge::GLM::confidence_interval_lower.
      %
      %   r = obj.confidence_interval_lower ()
      %   r = obj.confidence_interval_lower (level)
      %
      %   Inputs:
      %     level  double, default 0.95
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::confidence_interval_lower(double level = 0.94999999999999996) const
      r = __datamunge__(2768, obj, varargin{:});
    end
    function r = confidence_interval_upper (obj, varargin)
      % confidence_interval_upper  C++ datamunge::GLM::confidence_interval_upper.
      %
      %   r = obj.confidence_interval_upper ()
      %   r = obj.confidence_interval_upper (level)
      %
      %   Inputs:
      %     level  double, default 0.95
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::confidence_interval_upper(double level = 0.94999999999999996) const
      r = __datamunge__(2769, obj, varargin{:});
    end
    function r = summary (obj)
      % summary  C++ datamunge::GLM::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::GLM::summary() const
      r = __datamunge__(2770, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::GLM::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::GLM::print_summary() const
      __datamunge__(2771, obj);
    end
    function r = predict (obj, newdata)
      % predict  Response-scale predictions (back-transformed through the inverse link).
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::GLM::predict(const DataFrame & newdata) const
      r = __datamunge__(2772, obj, newdata);
    end
    function r = predict_frame (obj, varargin)
      % predict_frame  C++ datamunge::GLM::predict_frame.
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
      %   C++: DataFrame * datamunge::GLM::predict_frame(const DataFrame & newdata, const std::string & interval_kind = "none", double level = 0.94999999999999996) const
      r = __datamunge__(2773, obj, varargin{:});
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::GLM::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GLM::plot_residuals_vs_fitted() const
      r = __datamunge__(2774, obj);
    end
    function r = plot_normal_qq (obj)
      % plot_normal_qq  C++ datamunge::GLM::plot_normal_qq.
      %
      %   r = obj.plot_normal_qq ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GLM::plot_normal_qq() const
      r = __datamunge__(2775, obj);
    end
    function r = plot_scale_location (obj)
      % plot_scale_location  C++ datamunge::GLM::plot_scale_location.
      %
      %   r = obj.plot_scale_location ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GLM::plot_scale_location() const
      r = __datamunge__(2776, obj);
    end
    function r = plot_residuals_vs_leverage (obj)
      % plot_residuals_vs_leverage  C++ datamunge::GLM::plot_residuals_vs_leverage.
      %
      %   r = obj.plot_residuals_vs_leverage ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::GLM::plot_residuals_vs_leverage() const
      r = __datamunge__(2777, obj);
    end
    function save_diagnostic_plots (obj, path_prefix)
      % save_diagnostic_plots  C++ datamunge::GLM::save_diagnostic_plots.
      %
      %   obj.save_diagnostic_plots (path_prefix)
      %
      %   Inputs:
      %     path_prefix  char row
      %
      %   C++: void datamunge::GLM::save_diagnostic_plots(const std::string & path_prefix) const
      __datamunge__(2778, obj, path_prefix);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::GLM. Do not edit.
