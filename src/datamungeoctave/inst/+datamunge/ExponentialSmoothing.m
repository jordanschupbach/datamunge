classdef ExponentialSmoothing < datamunge.NativeObject
  % ExponentialSmoothing  Exponential smoothing (Holt-Winters family): simple exponential smoothing (no trend, no seasonality), Holt linear trend (with optional damping), and seasonal Holt-Winters (additive or multiplicative), unified under one class the way R's HoltWinters() and statsmodels' ExponentialSmoothing() are -- which components are active is controlled entirely by ExponentialSmoothingOptions::trend/seasonal.
  %
  %   obj = datamunge.ExponentialSmoothing (y)
  %   obj = datamunge.ExponentialSmoothing (y, options)
  %
  %   Initial level/trend/seasonal-index states are set once from a simple classical
  %   decomposition of the first one or two seasonal cycles and then held fixed; only
  %   the smoothing parameters (alpha, beta, gamma, phi) are fit, by minimizing
  %   one-step-ahead sum of squared errors with DifferentialEvolution.
  %
  %   Methods:
  %     options  C++ datamunge::stats::ExponentialSmoothing::options.
  %     alpha  C++ datamunge::stats::ExponentialSmoothing::alpha.
  %     beta  C++ datamunge::stats::ExponentialSmoothing::beta.
  %     gamma  C++ datamunge::stats::ExponentialSmoothing::gamma.
  %     phi  C++ datamunge::stats::ExponentialSmoothing::phi.
  %     observations  C++ datamunge::stats::ExponentialSmoothing::observations.
  %     sse  C++ datamunge::stats::ExponentialSmoothing::sse.
  %     sigma2  C++ datamunge::stats::ExponentialSmoothing::sigma2.
  %     log_likelihood  C++ datamunge::stats::ExponentialSmoothing::log_likelihood.
  %     aic  C++ datamunge::stats::ExponentialSmoothing::aic.
  %     bic  C++ datamunge::stats::ExponentialSmoothing::bic.
  %     fitted_values  In-sample one-step-ahead fitted values, aligned to the input series. The warm-up prefix used to set initial states (see class docs) is copied from the observed values (residual exactly zero there).
  %     residuals  C++ datamunge::stats::ExponentialSmoothing::residuals.
  %     forecast  Point forecasts for the next `horizon` periods.
  %
  %   C++: class datamunge::stats::ExponentialSmoothing
  methods
    function obj = ExponentialSmoothing (varargin)
      % ExponentialSmoothing  Construct a C++ datamunge::stats::ExponentialSmoothing.
      %
      %   obj = datamunge.ExponentialSmoothing (y)
      %   obj = datamunge.ExponentialSmoothing (y, options)
      %
      %   Inputs:
      %     y  double vector
      %     options  datamunge.ExponentialSmoothingOptions, default the C++ default
      %
      %   C++: datamunge::stats::ExponentialSmoothing::ExponentialSmoothing(const std::vector<double> & y, ExponentialSmoothingOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2013, false, varargin{:});
    end
    function r = options (obj)
      % options  C++ datamunge::stats::ExponentialSmoothing::options.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.ExponentialSmoothingOptions
      %
      %   C++: const ExponentialSmoothingOptions & datamunge::stats::ExponentialSmoothing::options() const
      r = __datamunge__(1999, obj);
    end
    function r = alpha (obj)
      % alpha  C++ datamunge::stats::ExponentialSmoothing::alpha.
      %
      %   r = obj.alpha ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::alpha() const
      r = __datamunge__(2000, obj);
    end
    function r = beta (obj)
      % beta  C++ datamunge::stats::ExponentialSmoothing::beta.
      %
      %   r = obj.beta ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::beta() const
      r = __datamunge__(2001, obj);
    end
    function r = gamma (obj)
      % gamma  C++ datamunge::stats::ExponentialSmoothing::gamma.
      %
      %   r = obj.gamma ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::gamma() const
      r = __datamunge__(2002, obj);
    end
    function r = phi (obj)
      % phi  C++ datamunge::stats::ExponentialSmoothing::phi.
      %
      %   r = obj.phi ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::phi() const
      r = __datamunge__(2003, obj);
    end
    function r = observations (obj)
      % observations  C++ datamunge::stats::ExponentialSmoothing::observations.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::stats::ExponentialSmoothing::observations() const
      r = __datamunge__(2004, obj);
    end
    function r = sse (obj)
      % sse  C++ datamunge::stats::ExponentialSmoothing::sse.
      %
      %   r = obj.sse ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::sse() const
      r = __datamunge__(2005, obj);
    end
    function r = sigma2 (obj)
      % sigma2  C++ datamunge::stats::ExponentialSmoothing::sigma2.
      %
      %   r = obj.sigma2 ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::sigma2() const
      r = __datamunge__(2006, obj);
    end
    function r = log_likelihood (obj)
      % log_likelihood  C++ datamunge::stats::ExponentialSmoothing::log_likelihood.
      %
      %   r = obj.log_likelihood ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::log_likelihood() const
      r = __datamunge__(2007, obj);
    end
    function r = aic (obj)
      % aic  C++ datamunge::stats::ExponentialSmoothing::aic.
      %
      %   r = obj.aic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::aic() const
      r = __datamunge__(2008, obj);
    end
    function r = bic (obj)
      % bic  C++ datamunge::stats::ExponentialSmoothing::bic.
      %
      %   r = obj.bic ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ExponentialSmoothing::bic() const
      r = __datamunge__(2009, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  In-sample one-step-ahead fitted values, aligned to the input series. The warm-up prefix used to set initial states (see class docs) is copied from the observed values (residual exactly zero there).
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: const std::vector<double> & datamunge::stats::ExponentialSmoothing::fitted_values() const
      r = __datamunge__(2010, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::stats::ExponentialSmoothing::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: const std::vector<double> & datamunge::stats::ExponentialSmoothing::residuals() const
      r = __datamunge__(2011, obj);
    end
    function r = forecast (obj, horizon)
      % forecast  Point forecasts for the next `horizon` periods.
      %
      %   r = obj.forecast (horizon)
      %
      %   Inputs:
      %     horizon  integer
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::stats::ExponentialSmoothing::forecast(std::size_t horizon) const
      r = __datamunge__(2012, obj, horizon);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::stats::ExponentialSmoothing. Do not edit.
