classdef EnsembleKalmanFilter < datamunge.NativeObject
  % EnsembleKalmanFilter  The (stochastic, "perturbed observations") Ensemble Kalman Filter: represents the state distribution as a finite ensemble of sample state vectors rather than an explicit mean/covariance, propagating each member through the true nonlinear f (plus sampled process noise) and estimating the covariances the Kalman update needs from the ensemble's own sample statistics. Like UnscentedKalmanFilter, needs no Jacobian; unlike it, scales to very high-dimensional states (the classical use case this filter was designed for -- e.g. geophysical data assimilation with millions of state variables) since its cost scales with ensemble size, not state dimension squared/cubed. Statistical, not exact: converges toward the optimal (linear-Gaussian case: KalmanFilter's) estimate as `ensemble_size` grows, with Monte Carlo sampling noise at any finite size.
  %
  %   obj = datamunge.EnsembleKalmanFilter (f, h, Q, R, x0, P0)
  %   obj = datamunge.EnsembleKalmanFilter (f, h, Q, R, x0, P0, ensemble_size)
  %   obj = datamunge.EnsembleKalmanFilter (f, h, Q, R, x0, P0, ensemble_size, seed)
  %
  %   Methods:
  %     predict  Propagates every ensemble member through the nonlinear f, each perturbed by an independently-sampled draw from N(0, Q).
  %     update  The stochastic EnKF update (Evensen/Burgers): estimates the state-observation cross-covariance and innovation covariance from the ensemble's own sample statistics, then updates each member with an INDEPENDENTLY perturbed copy of `z` (adding a fresh N(0, R) draw to each member's correction) -- perturbing the observation per member, rather than applying one shared correction to all of them, is what keeps the ensemble's spread statistically consistent after the update instead of collapsing it.
  %     state  C++ datamunge::filter::EnsembleKalmanFilter::state.
  %     covariance  C++ datamunge::filter::EnsembleKalmanFilter::covariance.
  %     filter  C++ datamunge::filter::EnsembleKalmanFilter::filter.
  %
  %   C++: class datamunge::filter::EnsembleKalmanFilter
  methods
    function obj = EnsembleKalmanFilter (varargin)
      % EnsembleKalmanFilter  Construct a C++ datamunge::filter::EnsembleKalmanFilter.
      %
      %   obj = datamunge.EnsembleKalmanFilter (f, h, Q, R, x0, P0)
      %   obj = datamunge.EnsembleKalmanFilter (f, h, Q, R, x0, P0, ensemble_size)
      %   obj = datamunge.EnsembleKalmanFilter (f, h, Q, R, x0, P0, ensemble_size, seed)
      %
      %   Inputs:
      %     f  datamunge.VectorFunction
      %     h  datamunge.VectorFunction
      %     Q  double matrix (rows)
      %     R  double matrix (rows)
      %     x0  double vector
      %     P0  double matrix (rows)
      %     ensemble_size  integer, default 100
      %     seed  integer, default 42
      %
      %   C++: datamunge::filter::EnsembleKalmanFilter::EnsembleKalmanFilter(VectorFunction & f, VectorFunction & h, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R, std::vector<double> x0, std::vector<std::vector<double>> P0, int ensemble_size = 100, unsigned int seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 812, false, varargin{:});
    end
    function predict (obj)
      % predict  Propagates every ensemble member through the nonlinear f, each perturbed by an independently-sampled draw from N(0, Q).
      %
      %   obj.predict ()
      %
      %   C++: void datamunge::filter::EnsembleKalmanFilter::predict()
      __datamunge__(807, obj);
    end
    function update (obj, z)
      % update  The stochastic EnKF update (Evensen/Burgers): estimates the state-observation cross-covariance and innovation covariance from the ensemble's own sample statistics, then updates each member with an INDEPENDENTLY perturbed copy of `z` (adding a fresh N(0, R) draw to each member's correction) -- perturbing the observation per member, rather than applying one shared correction to all of them, is what keeps the ensemble's spread statistically consistent after the update instead of collapsing it.
      %
      %   obj.update (z)
      %
      %   Inputs:
      %     z  double vector
      %
      %   C++: void datamunge::filter::EnsembleKalmanFilter::update(const std::vector<double> & z)
      __datamunge__(808, obj, z);
    end
    function r = state (obj)
      % state  C++ datamunge::filter::EnsembleKalmanFilter::state.
      %
      %   r = obj.state ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::filter::EnsembleKalmanFilter::state() const
      r = __datamunge__(809, obj);
    end
    function r = covariance (obj)
      % covariance  C++ datamunge::filter::EnsembleKalmanFilter::covariance.
      %
      %   r = obj.covariance ()
      %
      %   Output:
      %     r  double matrix (rows)
      %
      %   C++: std::vector<std::vector<double>> datamunge::filter::EnsembleKalmanFilter::covariance() const
      r = __datamunge__(810, obj);
    end
    function r = filter (obj, measurements)
      % filter  C++ datamunge::filter::EnsembleKalmanFilter::filter.
      %
      %   r = obj.filter (measurements)
      %
      %   Inputs:
      %     measurements  double matrix (rows)
      %
      %   Output:
      %     r  cell array of datamunge.KalmanState
      %
      %   C++: std::vector<KalmanState> datamunge::filter::EnsembleKalmanFilter::filter(const std::vector<std::vector<double>> & measurements)
      r = __datamunge__(811, obj, measurements);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::filter::EnsembleKalmanFilter. Do not edit.
