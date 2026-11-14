classdef KalmanFilter < datamunge.NativeObject
  % KalmanFilter  The standard discrete-time linear Kalman filter: x_k = F x_{k-1} + w_k, z_k = H x_k + v_k, w_k ~ N(0, Q), v_k ~ N(0, R) -- the minimum-mean-squared-error linear   estimator for exactly this model. Stateful: holds the current (x, P) estimate,   mutated in place by predict()/update(); F, H, Q, R are fixed for the filter's   lifetime (construct a new filter if they change over time).
  %
  %   obj = datamunge.KalmanFilter (F, H, Q, R, x0, P0)
  %
  %   Methods:
  %     predict  The time-update (prediction) step: x = F x, P = F P F^T + Q.
  %     update  The measurement-update (correction) step given observation `z`: the standard innovation-covariance / Kalman-gain form (S = H P H^T + R, K = P H^T S^-1, x += K(z - Hx), P = (I - KH) P) -- not the numerically-hardened Joseph form, matching this codebase's general preference for the textbook formula over a more robust but more complex alternative.
  %     state  C++ datamunge::filter::KalmanFilter::state.
  %     covariance  C++ datamunge::filter::KalmanFilter::covariance.
  %     filter  Runs predict() then update(z) for every z in `measurements`, in order, mutating this filter's running state -- the ordinary causal (online) filtered estimate at every timestep.
  %     smooth  filter() followed by an RTS (Rauch-Tung-Striebel) backward smoothing pass, which uses the ENTIRE sequence (not just data up to each point) to refine every estimate -- always at least as accurate as the filtered estimate, at the cost of needing the full sequence in advance (not usable online). Leaves this filter's running (x, P) at its post-filter state, same as filter() would.
  %
  %   C++: class datamunge::filter::KalmanFilter
  methods
    function obj = KalmanFilter (varargin)
      % KalmanFilter  Construct a C++ datamunge::filter::KalmanFilter.
      %
      %   obj = datamunge.KalmanFilter (F, H, Q, R, x0, P0)
      %
      %   Inputs:
      %     F  double matrix (rows)
      %     H  double matrix (rows)
      %     Q  double matrix (rows)
      %     R  double matrix (rows)
      %     x0  double vector
      %     P0  double matrix (rows)
      %
      %   C++: datamunge::filter::KalmanFilter::KalmanFilter(std::vector<std::vector<double>> F, std::vector<std::vector<double>> H, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R, std::vector<double> x0, std::vector<std::vector<double>> P0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 804, false, varargin{:});
    end
    function predict (obj)
      % predict  The time-update (prediction) step: x = F x, P = F P F^T + Q.
      %
      %   obj.predict ()
      %
      %   C++: void datamunge::filter::KalmanFilter::predict()
      __datamunge__(798, obj);
    end
    function update (obj, z)
      % update  The measurement-update (correction) step given observation `z`: the standard innovation-covariance / Kalman-gain form (S = H P H^T + R, K = P H^T S^-1, x += K(z - Hx), P = (I - KH) P) -- not the numerically-hardened Joseph form, matching this codebase's general preference for the textbook formula over a more robust but more complex alternative.
      %
      %   obj.update (z)
      %
      %   Inputs:
      %     z  double vector
      %
      %   C++: void datamunge::filter::KalmanFilter::update(const std::vector<double> & z)
      __datamunge__(799, obj, z);
    end
    function r = state (obj)
      % state  C++ datamunge::filter::KalmanFilter::state.
      %
      %   r = obj.state ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::filter::KalmanFilter::state() const
      r = __datamunge__(800, obj);
    end
    function r = covariance (obj)
      % covariance  C++ datamunge::filter::KalmanFilter::covariance.
      %
      %   r = obj.covariance ()
      %
      %   Output:
      %     r  double matrix (rows)
      %
      %   C++: std::vector<std::vector<double>> datamunge::filter::KalmanFilter::covariance() const
      r = __datamunge__(801, obj);
    end
    function r = filter (obj, measurements)
      % filter  Runs predict() then update(z) for every z in `measurements`, in order, mutating this filter's running state -- the ordinary causal (online) filtered estimate at every timestep.
      %
      %   r = obj.filter (measurements)
      %
      %   Inputs:
      %     measurements  double matrix (rows)
      %
      %   Output:
      %     r  cell array of datamunge.KalmanState
      %
      %   C++: std::vector<KalmanState> datamunge::filter::KalmanFilter::filter(const std::vector<std::vector<double>> & measurements)
      r = __datamunge__(802, obj, measurements);
    end
    function r = smooth (obj, measurements)
      % smooth  filter() followed by an RTS (Rauch-Tung-Striebel) backward smoothing pass, which uses the ENTIRE sequence (not just data up to each point) to refine every estimate -- always at least as accurate as the filtered estimate, at the cost of needing the full sequence in advance (not usable online). Leaves this filter's running (x, P) at its post-filter state, same as filter() would.
      %
      %   r = obj.smooth (measurements)
      %
      %   Inputs:
      %     measurements  double matrix (rows)
      %
      %   Output:
      %     r  datamunge.SmoothResult
      %
      %   C++: SmoothResult datamunge::filter::KalmanFilter::smooth(const std::vector<std::vector<double>> & measurements)
      r = __datamunge__(803, obj, measurements);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::filter::KalmanFilter. Do not edit.
