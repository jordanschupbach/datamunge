classdef LM < datamunge.NativeObject
  % LM  SWIG-friendly facade for datamunge::stats::LM — R-`lm()`-style linear models fit from a DataFrame.
  %
  %   obj = datamunge.LM (data, formula)
  %   obj = datamunge.LM (data, formula, weights_column)
  %
  %   Methods:
  %     formula_text  C++ datamunge::LM::formula_text.
  %     has_intercept  C++ datamunge::LM::has_intercept.
  %     observations  C++ datamunge::LM::observations.
  %     rank  C++ datamunge::LM::rank.
  %     degrees_of_freedom  C++ datamunge::LM::degrees_of_freedom.
  %     coefficients  C++ datamunge::LM::coefficients.
  %     coefficient_names  C++ datamunge::LM::coefficient_names.
  %     fitted_values  C++ datamunge::LM::fitted_values.
  %     residuals  C++ datamunge::LM::residuals.
  %     standard_errors  C++ datamunge::LM::standard_errors.
  %     t_values  C++ datamunge::LM::t_values.
  %     p_values  C++ datamunge::LM::p_values.
  %     r_squared  C++ datamunge::LM::r_squared.
  %     adjusted_r_squared  C++ datamunge::LM::adjusted_r_squared.
  %     sigma  C++ datamunge::LM::sigma.
  %     f_statistic  C++ datamunge::LM::f_statistic.
  %     f_p_value  C++ datamunge::LM::f_p_value.
  %     confidence_interval_lower  C++ datamunge::LM::confidence_interval_lower.
  %     confidence_interval_upper  C++ datamunge::LM::confidence_interval_upper.
  %     leverage  C++ datamunge::LM::leverage.
  %     standardized_residuals  C++ datamunge::LM::standardized_residuals.
  %     studentized_residuals  C++ datamunge::LM::studentized_residuals.
  %     cooks_distance  C++ datamunge::LM::cooks_distance.
  %     summary  C++ datamunge::LM::summary.
  %     print_summary  C++ datamunge::LM::print_summary.
  %     predict  C++ datamunge::LM::predict.
  %     predict_frame  C++ datamunge::LM::predict_frame.
  %     anova  Sequential (Type I) analysis-of-variance table as a DataFrame with columns term/df/sum_sq/mean_sq/f_value/p_value.
  %     plot_residuals_vs_fitted  C++ datamunge::LM::plot_residuals_vs_fitted.
  %     plot_normal_qq  C++ datamunge::LM::plot_normal_qq.
  %     plot_scale_location  C++ datamunge::LM::plot_scale_location.
  %     plot_residuals_vs_leverage  C++ datamunge::LM::plot_residuals_vs_leverage.
  %     save_diagnostic_plots  Saves all four diagnostic plots as "<path_prefix>_<name>.svg".
  %
  %   C++: class datamunge::LM
  methods
    function obj = LM (varargin)
      % LM  Construct a C++ datamunge::LM.
      %
      %   obj = datamunge.LM (data, formula)
      %   obj = datamunge.LM (data, formula, weights_column)
      %
      %   Inputs:
      %     data  datamunge.DataFrame
      %     formula  char row
      %     weights_column  char row, default '': Optional column name enabling weighted least squares; pass "" (the default) for OLS.
      %
      %   C++: datamunge::LM::LM(const DataFrame & data, const std::string & formula, const std::string & weights_column = "")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2211, false, varargin{:});
    end
    function r = formula_text (obj)
      % formula_text  C++ datamunge::LM::formula_text.
      %
      %   r = obj.formula_text ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LM::formula_text() const
      r = __datamunge__(2178, obj);
    end
    function r = has_intercept (obj)
      % has_intercept  C++ datamunge::LM::has_intercept.
      %
      %   r = obj.has_intercept ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::LM::has_intercept() const
      r = __datamunge__(2179, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::LM::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LM::observations() const
      r = __datamunge__(2180, obj);
    end
    function r = rank (obj)
      % rank  C++ datamunge::LM::rank.
      %
      %   r = obj.rank ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LM::rank() const
      r = __datamunge__(2181, obj);
    end
    function r = degrees_of_freedom (obj)
      % degrees_of_freedom  C++ datamunge::LM::degrees_of_freedom.
      %
      %   r = obj.degrees_of_freedom ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::LM::degrees_of_freedom() const
      r = __datamunge__(2182, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::LM::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::coefficients() const
      r = __datamunge__(2183, obj);
    end
    function r = coefficient_names (obj)
      % coefficient_names  C++ datamunge::LM::coefficient_names.
      %
      %   r = obj.coefficient_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::LM::coefficient_names() const
      r = __datamunge__(2184, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  C++ datamunge::LM::fitted_values.
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::fitted_values() const
      r = __datamunge__(2185, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::LM::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::residuals() const
      r = __datamunge__(2186, obj);
    end
    function r = standard_errors (obj)
      % standard_errors  C++ datamunge::LM::standard_errors.
      %
      %   r = obj.standard_errors ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::standard_errors() const
      r = __datamunge__(2187, obj);
    end
    function r = t_values (obj)
      % t_values  C++ datamunge::LM::t_values.
      %
      %   r = obj.t_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::t_values() const
      r = __datamunge__(2188, obj);
    end
    function r = p_values (obj)
      % p_values  C++ datamunge::LM::p_values.
      %
      %   r = obj.p_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::p_values() const
      r = __datamunge__(2189, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::LM::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LM::r_squared() const
      r = __datamunge__(2190, obj);
    end
    function r = adjusted_r_squared (obj)
      % adjusted_r_squared  C++ datamunge::LM::adjusted_r_squared.
      %
      %   r = obj.adjusted_r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LM::adjusted_r_squared() const
      r = __datamunge__(2191, obj);
    end
    function r = sigma (obj)
      % sigma  C++ datamunge::LM::sigma.
      %
      %   r = obj.sigma ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LM::sigma() const
      r = __datamunge__(2192, obj);
    end
    function r = f_statistic (obj)
      % f_statistic  C++ datamunge::LM::f_statistic.
      %
      %   r = obj.f_statistic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LM::f_statistic() const
      r = __datamunge__(2193, obj);
    end
    function r = f_p_value (obj)
      % f_p_value  C++ datamunge::LM::f_p_value.
      %
      %   r = obj.f_p_value ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::LM::f_p_value() const
      r = __datamunge__(2194, obj);
    end
    function r = confidence_interval_lower (obj, varargin)
      % confidence_interval_lower  C++ datamunge::LM::confidence_interval_lower.
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
      %   C++: std::vector<double> datamunge::LM::confidence_interval_lower(double level = 0.94999999999999996) const
      r = __datamunge__(2195, obj, varargin{:});
    end
    function r = confidence_interval_upper (obj, varargin)
      % confidence_interval_upper  C++ datamunge::LM::confidence_interval_upper.
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
      %   C++: std::vector<double> datamunge::LM::confidence_interval_upper(double level = 0.94999999999999996) const
      r = __datamunge__(2196, obj, varargin{:});
    end
    function r = leverage (obj)
      % leverage  C++ datamunge::LM::leverage.
      %
      %   r = obj.leverage ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::leverage() const
      r = __datamunge__(2197, obj);
    end
    function r = standardized_residuals (obj)
      % standardized_residuals  C++ datamunge::LM::standardized_residuals.
      %
      %   r = obj.standardized_residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::standardized_residuals() const
      r = __datamunge__(2198, obj);
    end
    function r = studentized_residuals (obj)
      % studentized_residuals  C++ datamunge::LM::studentized_residuals.
      %
      %   r = obj.studentized_residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::studentized_residuals() const
      r = __datamunge__(2199, obj);
    end
    function r = cooks_distance (obj)
      % cooks_distance  C++ datamunge::LM::cooks_distance.
      %
      %   r = obj.cooks_distance ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::cooks_distance() const
      r = __datamunge__(2200, obj);
    end
    function r = summary (obj)
      % summary  C++ datamunge::LM::summary.
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::LM::summary() const
      r = __datamunge__(2201, obj);
    end
    function print_summary (obj)
      % print_summary  C++ datamunge::LM::print_summary.
      %
      %   obj.print_summary ()
      %
      %   C++: void datamunge::LM::print_summary() const
      __datamunge__(2202, obj);
    end
    function r = predict (obj, newdata)
      % predict  C++ datamunge::LM::predict.
      %
      %   r = obj.predict (newdata)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::LM::predict(const DataFrame & newdata) const
      r = __datamunge__(2203, obj, newdata);
    end
    function r = predict_frame (obj, varargin)
      % predict_frame  C++ datamunge::LM::predict_frame.
      %
      %   r = obj.predict_frame (newdata)
      %   r = obj.predict_frame (newdata, interval_kind)
      %   r = obj.predict_frame (newdata, interval_kind, level)
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame
      %     interval_kind  char row, default 'none': One of "none", "confidence", "prediction". Returned DataFrame has a "fit" column, plus "se_fit"/"lwr"/"upr" when an interval is requested.
      %     level  double, default 0.95
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LM::predict_frame(const DataFrame & newdata, const std::string & interval_kind = "none", double level = 0.94999999999999996) const
      r = __datamunge__(2204, obj, varargin{:});
    end
    function r = anova (obj)
      % anova  Sequential (Type I) analysis-of-variance table as a DataFrame with columns term/df/sum_sq/mean_sq/f_value/p_value.
      %
      %   r = obj.anova ()
      %
      %   Output:
      %     r  datamunge.DataFrame or []
      %
      %   C++: DataFrame * datamunge::LM::anova() const
      r = __datamunge__(2205, obj);
    end
    function r = plot_residuals_vs_fitted (obj)
      % plot_residuals_vs_fitted  C++ datamunge::LM::plot_residuals_vs_fitted.
      %
      %   r = obj.plot_residuals_vs_fitted ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::LM::plot_residuals_vs_fitted() const
      r = __datamunge__(2206, obj);
    end
    function r = plot_normal_qq (obj)
      % plot_normal_qq  C++ datamunge::LM::plot_normal_qq.
      %
      %   r = obj.plot_normal_qq ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::LM::plot_normal_qq() const
      r = __datamunge__(2207, obj);
    end
    function r = plot_scale_location (obj)
      % plot_scale_location  C++ datamunge::LM::plot_scale_location.
      %
      %   r = obj.plot_scale_location ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::LM::plot_scale_location() const
      r = __datamunge__(2208, obj);
    end
    function r = plot_residuals_vs_leverage (obj)
      % plot_residuals_vs_leverage  C++ datamunge::LM::plot_residuals_vs_leverage.
      %
      %   r = obj.plot_residuals_vs_leverage ()
      %
      %   Output:
      %     r  datamunge.RPlot
      %
      %   C++: datamunge::graphics::RPlot datamunge::LM::plot_residuals_vs_leverage() const
      r = __datamunge__(2209, obj);
    end
    function save_diagnostic_plots (obj, path_prefix)
      % save_diagnostic_plots  Saves all four diagnostic plots as "<path_prefix>_<name>.svg".
      %
      %   obj.save_diagnostic_plots (path_prefix)
      %
      %   Inputs:
      %     path_prefix  char row
      %
      %   C++: void datamunge::LM::save_diagnostic_plots(const std::string & path_prefix) const
      __datamunge__(2210, obj, path_prefix);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::LM. Do not edit.
