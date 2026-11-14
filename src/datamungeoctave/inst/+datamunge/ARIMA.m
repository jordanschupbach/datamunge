classdef ARIMA < datamunge.NativeObject
  % ARIMA  A Box-Jenkins seasonal ARIMA(p, d, q)(P, D, Q)_s model, fit to a single numeric series by conditional sum of squares (CSS): the series is differenced (regular d times, then seasonal D times with period s) to stationarity, the combined AR/MA polynomials (regular composed with seasonal, via polynomial multiplication) are estimated by minimizing the conditional residual sum of squares with DifferentialEvolution, and forecasts are produced by recursing the fitted difference equation forward and re-integrating through the differencing steps.
  %
  %   obj = datamunge.ARIMA (y)
  %   obj = datamunge.ARIMA (y, options)
  %
  %   Setting seasonal_period == 0 (the default) reduces this to a plain, non-seasonal
  %   ARIMA(p, d, q) model -- i.e. SARIMA is the general case, ARIMA the special case,
  %   exactly as in R's arima()/statsmodels' SARIMAX().
  %
  %   Note: CSS estimation approximates the exact Gaussian likelihood by conditioning
  %   on the first max(p + P*s, q + Q*s) observations rather than filtering the full
  %   series (e.g. via a Kalman filter); it is a standard, textbook ARIMA fitting
  %   method (R's arima(method = "CSS")) but will differ slightly from exact-ML fits.
  %
  %   Methods:
  %     options  The options the model was fit with.
  %     ar_coefficients  Fitted non-seasonal AR coefficients
  %     ma_coefficients  Fitted non-seasonal MA coefficients
  %     seasonal_ar_coefficients  Fitted seasonal AR coefficients
  %     seasonal_ma_coefficients  Fitted seasonal MA coefficients
  %     mean  Fitted mean of the differenced series.
  %     observations  Length of the input series.
  %     n_used  Number of residuals that enter the sum of squares.
  %     sigma2  Innovation variance estimate.
  %     log_likelihood  Gaussian log-likelihood at the CSS estimate, over n_used() residuals.
  %     aic  Akaike information criterion.
  %     bic  Bayesian information criterion.
  %     fitted_values  One-step-ahead fitted values on the original scale, aligned to the input series (the first `observations() - n_used()` entries -- the presample points conditioned on by CSS -- are copies of the observed value, matching a residual of exactly zero there rather than an undefined value).
  %     residuals  One-step-ahead residuals on the original scale (zero for the presample points).
  %     forecast  Point forecasts for the next `horizon` periods, on the original scale.
  %     expanded_ar_coefficients  AR coefficients of the stationary (differenced) model after multiplying the regular and seasonal polynomials,
  %     expanded_ma_coefficients  MA coefficients
  %     psi_weights  The first `count`
  %     forecast_with_intervals  Point forecasts and their standard errors (from the psi-weight / MA(infinity) expansion of the full, non-stationary AR/MA representation of y), on the original scale.
  %
  %   C++: class datamunge::stats::ARIMA
  methods
    function obj = ARIMA (varargin)
      % ARIMA  Construct a C++ datamunge::stats::ARIMA.
      %
      %   obj = datamunge.ARIMA (y)
      %   obj = datamunge.ARIMA (y, options)
      %
      %   Fit the model to a series.
      %
      %   Inputs:
      %     y  double vector: The observed series, in time order.
      %     options  datamunge.ARIMAOptions, default the C++ default: Model orders and optimizer settings.
      %
      %   Errors:
      %     datamunge:invalid_argument  If a seasonal order is set while seasonal_period is zero, or if the series is too short for the requested differencing or AR order. (C++ std::invalid_argument)
      %
      %   C++: datamunge::stats::ARIMA::ARIMA(const std::vector<double> & y, ARIMAOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1962, false, varargin{:});
    end
    function r = options (obj)
      % options  The options the model was fit with.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.ARIMAOptions: The stored ARIMAOptions.
      %
      %   C++: const ARIMAOptions & datamunge::stats::ARIMA::options() const
      r = __datamunge__(1943, obj);
    end
    function r = ar_coefficients (obj)
      % ar_coefficients  Fitted non-seasonal AR coefficients
      %
      %   r = obj.ar_coefficients ()
      %
      %   `\phi_1,\dots,\phi_p`.
      %
      %   `1-\sum_i\phi_i B^i`.
      %
      %   Output:
      %     r  double vector: A vector of length p; the AR polynomial is
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::ar_coefficients() const
      r = __datamunge__(1944, obj);
    end
    function r = ma_coefficients (obj)
      % ma_coefficients  Fitted non-seasonal MA coefficients
      %
      %   r = obj.ma_coefficients ()
      %
      %   `\theta_1,\dots,\theta_q`.
      %
      %   `1+\sum_j\theta_j B^j`.
      %
      %   Output:
      %     r  double vector: A vector of length q; the MA polynomial is
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::ma_coefficients() const
      r = __datamunge__(1945, obj);
    end
    function r = seasonal_ar_coefficients (obj)
      % seasonal_ar_coefficients  Fitted seasonal AR coefficients
      %
      %   r = obj.seasonal_ar_coefficients ()
      %
      %   `\Phi_1,\dots,\Phi_P`.
      %
      %   `1-\sum_i\Phi_i B^{is}`.
      %
      %   Output:
      %     r  double vector: A vector of length P; the polynomial is
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::seasonal_ar_coefficients() const
      r = __datamunge__(1946, obj);
    end
    function r = seasonal_ma_coefficients (obj)
      % seasonal_ma_coefficients  Fitted seasonal MA coefficients
      %
      %   r = obj.seasonal_ma_coefficients ()
      %
      %   `\Theta_1,\dots,\Theta_Q`.
      %
      %   `1+\sum_j\Theta_j B^{js}`.
      %
      %   Output:
      %     r  double vector: A vector of length Q; the polynomial is
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::seasonal_ma_coefficients() const
      r = __datamunge__(1947, obj);
    end
    function r = mean (obj)
      % mean  Fitted mean of the differenced series.
      %
      %   r = obj.mean ()
      %
      %   Output:
      %     r  double: The mean, or 0 when no mean term was fit (include_mean is false, or d or D is nonzero).
      %
      %   C++: double datamunge::stats::ARIMA::mean() const
      r = __datamunge__(1948, obj);
    end
    function r = observations (obj)
      % observations  Length of the input series.
      %
      %   r = obj.observations ()
      %
      %   Output:
      %     r  integer: The number of observations passed to the constructor.
      %
      %   C++: std::size_t datamunge::stats::ARIMA::observations() const
      r = __datamunge__(1949, obj);
    end
    function r = n_used (obj)
      % n_used  Number of residuals that enter the sum of squares.
      %
      %   r = obj.n_used ()
      %
      %   `p+sP` (the conditioned-on presample).
      %
      %   Output:
      %     r  integer: The differenced series length minus
      %
      %   C++: std::size_t datamunge::stats::ARIMA::n_used() const
      r = __datamunge__(1950, obj);
    end
    function r = sigma2 (obj)
      % sigma2  Innovation variance estimate.
      %
      %   r = obj.sigma2 ()
      %
      %   Output:
      %     r  double: The CSS residual sum of squares divided by n_used().
      %
      %   C++: double datamunge::stats::ARIMA::sigma2() const
      r = __datamunge__(1951, obj);
    end
    function r = log_likelihood (obj)
      % log_likelihood  Gaussian log-likelihood at the CSS estimate, over n_used() residuals.
      %
      %   r = obj.log_likelihood ()
      %
      %   `-\tfrac{n}{2}(\log 2\pi + \log\sigma^2 + 1)` with `n` = n_used().
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ARIMA::log_likelihood() const
      r = __datamunge__(1952, obj);
    end
    function r = aic (obj)
      % aic  Akaike information criterion.
      %
      %   r = obj.aic ()
      %
      %   `-2\ell + 2k`, where `k` counts the coefficients, the mean (if fit) and `\sigma^2`.
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ARIMA::aic() const
      r = __datamunge__(1953, obj);
    end
    function r = bic (obj)
      % bic  Bayesian information criterion.
      %
      %   r = obj.bic ()
      %
      %   `-2\ell + k\log n`, with `k` as in aic() and `n` = n_used().
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::stats::ARIMA::bic() const
      r = __datamunge__(1954, obj);
    end
    function r = fitted_values (obj)
      % fitted_values  One-step-ahead fitted values on the original scale, aligned to the input series (the first `observations() - n_used()` entries -- the presample points conditioned on by CSS -- are copies of the observed value, matching a residual of exactly zero there rather than an undefined value).
      %
      %   r = obj.fitted_values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::fitted_values() const
      r = __datamunge__(1955, obj);
    end
    function r = residuals (obj)
      % residuals  One-step-ahead residuals on the original scale (zero for the presample points).
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::residuals() const
      r = __datamunge__(1956, obj);
    end
    function r = forecast (obj, horizon)
      % forecast  Point forecasts for the next `horizon` periods, on the original scale.
      %
      %   r = obj.forecast (horizon)
      %
      %   Inputs:
      %     horizon  integer: Number of steps ahead to forecast.
      %
      %   Output:
      %     r  double vector: A vector of length `horizon` (empty when `horizon` is 0).
      %
      %   C++: std::vector<double> datamunge::stats::ARIMA::forecast(std::size_t horizon) const
      r = __datamunge__(1957, obj, horizon);
    end
    function r = expanded_ar_coefficients (obj)
      % expanded_ar_coefficients  AR coefficients of the stationary (differenced) model after multiplying the regular and seasonal polynomials,
      %
      %   r = obj.expanded_ar_coefficients ()
      %
      %   `w_t=\sum_i a_i w_{t-i}+\varepsilon_t+\sum_j b_j\varepsilon_{t-j}`.
      %
      %   `a_i`, `i=1,\dots,p+sP` (zeros where no lag enters).
      %
      %   Output:
      %     r  double vector: The
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::expanded_ar_coefficients() const
      r = __datamunge__(1958, obj);
    end
    function r = expanded_ma_coefficients (obj)
      % expanded_ma_coefficients  MA coefficients
      %
      %   r = obj.expanded_ma_coefficients ()
      %
      %   `b_j` of the stationary model, after multiplying the regular and
      %   seasonal MA polynomials; for the airline model `(\theta,0,\dots,0,\Theta,\theta\Theta)`.
      %
      %   `b_j`, `j=1,\dots,q+sQ`.
      %
      %   Output:
      %     r  double vector: The
      %
      %   C++: const std::vector<double> & datamunge::stats::ARIMA::expanded_ma_coefficients() const
      r = __datamunge__(1959, obj);
    end
    function r = psi_weights (obj, count)
      % psi_weights  The first `count`
      %
      %   r = obj.psi_weights (count)
      %
      %   `\psi`-weights of the full model, differencing included:
      %   `y_t=\sum_{j\ge0}\psi_j\varepsilon_{t-j}` formally, with `\psi_0=1`.
      %
      %   The h-step forecast error is `\sum_{j<h}\psi_j\varepsilon_{t+h-j}`, so its variance is
      %   `\sigma^2\sum_{j<h}\psi_j^2`; forecast_with_intervals() uses exactly these weights.
      %
      %   `\psi_0,\dots,\psi_{count-1}`.
      %
      %   Inputs:
      %     count  integer: Number of weights.
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::stats::ARIMA::psi_weights(std::size_t count) const
      r = __datamunge__(1960, obj, count);
    end
    function r = forecast_with_intervals (obj, horizon)
      % forecast_with_intervals  Point forecasts and their standard errors (from the psi-weight / MA(infinity) expansion of the full, non-stationary AR/MA representation of y), on the original scale.
      %
      %   r = obj.forecast_with_intervals (horizon)
      %
      %   Inputs:
      %     horizon  integer: Number of steps ahead to forecast.
      %
      %   Output:
      %     r  cell {double vector, double vector}: A pair (point forecasts, standard errors), each of length `horizon`.
      %
      %   C++: std::pair<std::vector<double>, std::vector<double>> datamunge::stats::ARIMA::forecast_with_intervals(std::size_t horizon) const
      r = __datamunge__(1961, obj, horizon);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::stats::ARIMA. Do not edit.
