classdef RBFGaussianProcessSurrogate < datamunge.BayesianSurrogate
  % RBFGaussianProcessSurrogate  Exact Gaussian-process regression surrogate with the squared-exponential kernel
  %
  %   obj = datamunge.RBFGaussianProcessSurrogate ()
  %   obj = datamunge.RBFGaussianProcessSurrogate (length_scale)
  %   obj = datamunge.RBFGaussianProcessSurrogate (length_scale, noise)
  %   obj = datamunge.RBFGaussianProcessSurrogate (options)
  %
  %   `s\exp(-\lVert x-x'\rVert^2/2\ell^2)`.
  %
  %   fit() Cholesky-factors `K+\sigma_n^2I=LL^\top` once and solves for
  %   `\alpha=(K+\sigma_n^2I)^{-1}(y-m)`; predict() then needs one forward substitution per point:
  %   mean `m+k_*^\top\alpha` and variance `s-\lVert L^{-1}k_*\rVert^2`.
  %
  %   The two-argument constructor keeps the original fixed setup: zero prior mean, unit signal
  %   variance, no normalisation or fitting, expected improvement. The options constructor adds
  %   value normalisation, marginal-likelihood fitting of the hyperparameters and other acquisitions.
  %   The kernel is isotropic, so the inputs should be on comparable scales.
  %
  %   Example:
  %     dm::optim::RBFGaussianProcessSurrogate gp(1.0, 0.0);
  %     gp.fit({{0.0}, {2.0}}, {1.0, -1.0});
  %     const auto p = gp.predict({1.0});
  %     std::print("{:.4f} {:.4f}\n", p.mean, p.sd);  // 0.0000 0.5933
  %
  %   References:
  %   - C. E. Rasmussen and C. K. I. Williams (2006). *Gaussian Processes for Machine Learning*,
  %     chapters 2 and 5. MIT Press.
  %
  %   Inherits from datamunge.BayesianSurrogate.
  %
  %   Octave subclasses (classdef MyClass < datamunge.RBFGaussianProcessSurrogate) override its virtual
  %   methods, also when C++ calls them: fit, acquisition.
  %
  %   Methods:
  %     options  The options this surrogate was built with.
  %     fit  Fits the GP to the observations (refitting hyperparameters when configured).
  %     acquisition  The configured acquisition at `point`.
  %     predict  Posterior mean and standard deviation at `point`.
  %     log_marginal_likelihood  Log marginal likelihood of the values at the hyperparameters in use.
  %     length_scale  The length scale in use (fitted, after fit(), when fitting is on).
  %     noise  The noise variance in use, in normalised units when normalising.
  %     signal_variance  The signal variance in use, in normalised units when normalising.
  %     prior_mean  The prior mean: the mean of the values when normalising, else 0.
  %     value_scale  The value scale: the standard deviation of the values when normalising, else 1.
  %     points  The training points from the last fit() (empty before fitting).
  %     values  The training values from the last fit() (empty before fitting).
  %
  %   C++: class datamunge::optim::RBFGaussianProcessSurrogate
  %
  %   See also: datamunge.BayesianOptimization, datamunge.expected_improvement()
  methods
    function obj = RBFGaussianProcessSurrogate (varargin)
      % RBFGaussianProcessSurrogate  Construct a C++ datamunge::optim::RBFGaussianProcessSurrogate.
      %
      %   obj = datamunge.RBFGaussianProcessSurrogate ()
      %   obj = datamunge.RBFGaussianProcessSurrogate (length_scale)
      %   obj = datamunge.RBFGaussianProcessSurrogate (length_scale, noise)
      %   obj = datamunge.RBFGaussianProcessSurrogate (options)
      %
      %   `\ell>0`.
      %
      %   `\ge0`).
      %
      %   Fixed-hyperparameter surrogate with expected improvement.
      %
      %   Errors:
      %     datamunge:invalid_argument  If `length_scale` is not positive or `noise` is negative. (C++ std::invalid_argument)
      %
      %   C++: datamunge::optim::RBFGaussianProcessSurrogate::RBFGaussianProcessSurrogate(double length_scale = 1.0, double noise = 9.9999999999999995E-7)
      %   C++: datamunge::optim::RBFGaussianProcessSurrogate::RBFGaussianProcessSurrogate(const GaussianProcessSurrogateOptions & options)
      obj@datamunge.BayesianSurrogate(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1794, ~strcmp (class (obj), 'datamunge.RBFGaussianProcessSurrogate'), varargin{:});
    end
    function r = options (obj)
      % options  The options this surrogate was built with.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.GaussianProcessSurrogateOptions: The stored options.
      %
      %   C++: const GaussianProcessSurrogateOptions & datamunge::optim::RBFGaussianProcessSurrogate::options() const
      r = __datamunge__(1782, obj);
    end
    function fit (obj, points, values)
      % fit  Fits the GP to the observations (refitting hyperparameters when configured).
      %
      %   obj.fit (points, values)
      %
      %   Inputs:
      %     points  double matrix (rows): Training points, all of one dimension.
      %     values  double vector: Objective values.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the data are empty, mismatched or of mixed dimension. (C++ std::invalid_argument)
      %
      %   C++: void datamunge::optim::RBFGaussianProcessSurrogate::fit(const std::vector<std::vector<double>> & points, const std::vector<double> & values)
      __datamunge__(1783, obj, points, values);
    end
    function r = acquisition (obj, point, incumbent)
      % acquisition  The configured acquisition at `point`.
      %
      %   r = obj.acquisition (point, incumbent)
      %
      %   Inputs:
      %     point  double vector: Candidate point.
      %     incumbent  double: Smallest observed value.
      %
      %   Output:
      %     r  double: The acquisition value (0 before fitting).
      %
      %   Errors:
      %     datamunge:invalid_argument  If `point` has the wrong dimension. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::acquisition(const std::vector<double> & point, double incumbent)
      r = __datamunge__(1784, obj, point, incumbent);
    end
    function r = predict (obj, point)
      % predict  Posterior mean and standard deviation at `point`.
      %
      %   r = obj.predict (point)
      %
      %   Inputs:
      %     point  double vector: Query point.
      %
      %   Output:
      %     r  datamunge.SurrogatePrediction: The prediction, in the objective's units.
      %
      %   Errors:
      %     datamunge:native  Before fit(). (C++ std::logic_error)
      %     datamunge:invalid_argument  If `point` has the wrong dimension. (C++ std::invalid_argument)
      %
      %   C++: SurrogatePrediction datamunge::optim::RBFGaussianProcessSurrogate::predict(const std::vector<double> & point) const
      r = __datamunge__(1785, obj, point);
    end
    function r = log_marginal_likelihood (obj)
      % log_marginal_likelihood  Log marginal likelihood of the values at the hyperparameters in use.
      %
      %   r = obj.log_marginal_likelihood ()
      %
      %   The density of the raw values under the GP with the prior mean (and scale, when
      %   normalising) treated as known, so it includes the `-n\log(\text{scale})` Jacobian.
      %
      %   `\log p(y)`; 0 before fit().
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::log_marginal_likelihood() const
      r = __datamunge__(1786, obj);
    end
    function r = length_scale (obj)
      % length_scale  The length scale in use (fitted, after fit(), when fitting is on).
      %
      %   r = obj.length_scale ()
      %
      %   `\ell`.
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::length_scale() const
      r = __datamunge__(1787, obj);
    end
    function r = noise (obj)
      % noise  The noise variance in use, in normalised units when normalising.
      %
      %   r = obj.noise ()
      %
      %   `\sigma_n^2`.
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::noise() const
      r = __datamunge__(1788, obj);
    end
    function r = signal_variance (obj)
      % signal_variance  The signal variance in use, in normalised units when normalising.
      %
      %   r = obj.signal_variance ()
      %
      %   `s`.
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::signal_variance() const
      r = __datamunge__(1789, obj);
    end
    function r = prior_mean (obj)
      % prior_mean  The prior mean: the mean of the values when normalising, else 0.
      %
      %   r = obj.prior_mean ()
      %
      %   `m`.
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::prior_mean() const
      r = __datamunge__(1790, obj);
    end
    function r = value_scale (obj)
      % value_scale  The value scale: the standard deviation of the values when normalising, else 1.
      %
      %   r = obj.value_scale ()
      %
      %   Output:
      %     r  double: The scale.
      %
      %   C++: double datamunge::optim::RBFGaussianProcessSurrogate::value_scale() const
      r = __datamunge__(1791, obj);
    end
    function r = points (obj)
      % points  The training points from the last fit() (empty before fitting).
      %
      %   r = obj.points ()
      %
      %   Output:
      %     r  double matrix (rows): The stored training points.
      %
      %   C++: const std::vector<std::vector<double>> & datamunge::optim::RBFGaussianProcessSurrogate::points() const
      r = __datamunge__(1792, obj);
    end
    function r = values (obj)
      % values  The training values from the last fit() (empty before fitting).
      %
      %   r = obj.values ()
      %
      %   Output:
      %     r  double vector: The stored training values.
      %
      %   C++: const std::vector<double> & datamunge::optim::RBFGaussianProcessSurrogate::values() const
      r = __datamunge__(1793, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::RBFGaussianProcessSurrogate. Do not edit.
