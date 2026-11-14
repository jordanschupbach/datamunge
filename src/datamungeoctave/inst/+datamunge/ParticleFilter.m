classdef ParticleFilter < datamunge.NativeObject
  % ParticleFilter  A bootstrap particle filter (Sequential Importance Resampling): the fully general, non-Gaussian generalization of the Kalman family -- represents the state's entire posterior distribution as a weighted swarm of sample points rather than assuming any particular family of distribution (Gaussian, etc.), so it remains correct even for strongly non-Gaussian or multimodal posteriors where every other filter in this module (which all assume approximately-Gaussian beliefs) can fail. Costs scale with particle count and can need many particles in high dimensions ("the curse of dimensionality" for particle filters) -- EnsembleKalmanFilter is usually preferred for high-dimensional problems where the Gaussian assumption is acceptable.
  %
  %   obj = datamunge.ParticleFilter (f, h, Q, R, x0, P0)
  %   obj = datamunge.ParticleFilter (f, h, Q, R, x0, P0, num_particles)
  %   obj = datamunge.ParticleFilter (f, h, Q, R, x0, P0, num_particles, seed)
  %
  %   Methods:
  %     predict  Propagates every particle through the nonlinear f, each perturbed by an independently-sampled draw from N(0, Q). Weights are untouched (the transition density cancels out of the importance weight for this proposal, the standard bootstrap-filter simplification).
  %     update  Reweights every particle by its Gaussian (N(h(x), R)) observation likelihood given `z`, normalizes the weights, then resamples the whole swarm from the resulting weighted distribution (multinomial resampling, done unconditionally every step -- simpler than the adaptive "resample only when the effective sample size drops" scheme real-time systems often use, at the cost of extra Monte Carlo noise from resampling more often than strictly necessary).
  %     state  C++ datamunge::filter::ParticleFilter::state.
  %     covariance  C++ datamunge::filter::ParticleFilter::covariance.
  %     filter  C++ datamunge::filter::ParticleFilter::filter.
  %
  %   C++: class datamunge::filter::ParticleFilter
  methods
    function obj = ParticleFilter (varargin)
      % ParticleFilter  Construct a C++ datamunge::filter::ParticleFilter.
      %
      %   obj = datamunge.ParticleFilter (f, h, Q, R, x0, P0)
      %   obj = datamunge.ParticleFilter (f, h, Q, R, x0, P0, num_particles)
      %   obj = datamunge.ParticleFilter (f, h, Q, R, x0, P0, num_particles, seed)
      %
      %   Inputs:
      %     f  datamunge.VectorFunction
      %     h  datamunge.VectorFunction
      %     Q  double matrix (rows)
      %     R  double matrix (rows)
      %     x0  double vector
      %     P0  double matrix (rows)
      %     num_particles  integer, default 200
      %     seed  integer, default 42
      %
      %   C++: datamunge::filter::ParticleFilter::ParticleFilter(VectorFunction & f, VectorFunction & h, std::vector<std::vector<double>> Q, std::vector<std::vector<double>> R, std::vector<double> x0, std::vector<std::vector<double>> P0, int num_particles = 200, unsigned int seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 830, false, varargin{:});
    end
    function predict (obj)
      % predict  Propagates every particle through the nonlinear f, each perturbed by an independently-sampled draw from N(0, Q). Weights are untouched (the transition density cancels out of the importance weight for this proposal, the standard bootstrap-filter simplification).
      %
      %   obj.predict ()
      %
      %   C++: void datamunge::filter::ParticleFilter::predict()
      __datamunge__(825, obj);
    end
    function update (obj, z)
      % update  Reweights every particle by its Gaussian (N(h(x), R)) observation likelihood given `z`, normalizes the weights, then resamples the whole swarm from the resulting weighted distribution (multinomial resampling, done unconditionally every step -- simpler than the adaptive "resample only when the effective sample size drops" scheme real-time systems often use, at the cost of extra Monte Carlo noise from resampling more often than strictly necessary).
      %
      %   obj.update (z)
      %
      %   Inputs:
      %     z  double vector
      %
      %   C++: void datamunge::filter::ParticleFilter::update(const std::vector<double> & z)
      __datamunge__(826, obj, z);
    end
    function r = state (obj)
      % state  C++ datamunge::filter::ParticleFilter::state.
      %
      %   r = obj.state ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::filter::ParticleFilter::state() const
      r = __datamunge__(827, obj);
    end
    function r = covariance (obj)
      % covariance  C++ datamunge::filter::ParticleFilter::covariance.
      %
      %   r = obj.covariance ()
      %
      %   Output:
      %     r  double matrix (rows)
      %
      %   C++: std::vector<std::vector<double>> datamunge::filter::ParticleFilter::covariance() const
      r = __datamunge__(828, obj);
    end
    function r = filter (obj, measurements)
      % filter  C++ datamunge::filter::ParticleFilter::filter.
      %
      %   r = obj.filter (measurements)
      %
      %   Inputs:
      %     measurements  double matrix (rows)
      %
      %   Output:
      %     r  cell array of datamunge.KalmanState
      %
      %   C++: std::vector<KalmanState> datamunge::filter::ParticleFilter::filter(const std::vector<std::vector<double>> & measurements)
      r = __datamunge__(829, obj, measurements);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::filter::ParticleFilter. Do not edit.
