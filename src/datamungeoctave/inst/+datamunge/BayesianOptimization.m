classdef BayesianOptimization < datamunge.NativeObject
  % BayesianOptimization  Minimises an expensive black-box function on a box with few evaluations.
  %
  %   obj = datamunge.BayesianOptimization ()
  %   obj = datamunge.BayesianOptimization (options)
  %
  %   The loop is:
  %   1. evaluate `initial_samples` points drawn uniformly in `[lower, upper]`;
  %   2. repeat `max_iterations` times: fit the surrogate to all observations, score `candidates`
  %     uniform random points with BayesianSurrogate::acquisition() (the incumbent being the
  %     smallest value so far), polish the best one by compass search on the acquisition (steps
  %     of 10% of each side, halved `refine_iterations` times, clipped to the box), and evaluate
  %     the objective there.
  %
  %   Only point evaluations of the objective are used, so it suits expensive simulations,
  %   experiments and hyperparameter tuning where each evaluation costs far more than the
  %   surrogate fit.
  %
  %   Example:
  %     struct Bowl : dm::optim::ArbitraryFunction {
  %         double evaluate(const std::vector<double>& x) override { return (x[0] - 1.0) * (x[0] - 1.0); }
  %     } bowl;
  %     dm::optim::RBFGaussianProcessSurrogate gp({.fit_hyperparameters = true});
  %     const auto result = dm::optim::BayesianOptimization({.initial_samples = 4, .max_iterations = 10})
  %                             .run(bowl, {-3.0}, {3.0}, gp);
  %     std::print("{:.3f} after {} evaluations\n", result.best_point[0], result.values.size());  // 1.002 after 14 evaluations
  %
  %   References:
  %   - D. R. Jones, M. Schonlau and W. J. Welch (1998). "Efficient global optimization of expensive
  %     black-box functions". Journal of Global Optimization 13, 455–492.
  %   - B. Shahriari et al. (2016). "Taking the human out of the loop: a review of Bayesian
  %     optimization". Proceedings of the IEEE 104(1), 148–175.
  %
  %   Methods:
  %     run  Runs the loop and records every evaluation.
  %     optimize  Runs the loop and returns only the best value.
  %     options  The options this BayesianOptimization was configured with.
  %
  %   C++: class datamunge::optim::BayesianOptimization
  %
  %   See also: datamunge.RBFGaussianProcessSurrogate, datamunge.expected_improvement()
  methods
    function obj = BayesianOptimization (varargin)
      % BayesianOptimization  Construct a C++ datamunge::optim::BayesianOptimization.
      %
      %   obj = datamunge.BayesianOptimization ()
      %   obj = datamunge.BayesianOptimization (options)
      %
      %   Creates the optimizer.
      %
      %   Inputs:
      %     options  datamunge.BayesianOptimizationOptions, default the C++ default: Design size, iteration count, seed and acquisition search settings.
      %
      %   C++: datamunge::optim::BayesianOptimization::BayesianOptimization(BayesianOptimizationOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1331, false, varargin{:});
    end
    function r = run (obj, function_, lower, upper, surrogate)
      % run  Runs the loop and records every evaluation.
      %
      %   r = obj.run (function_, lower, upper, surrogate)
      %
      %   Inputs:
      %     function_  datamunge.ArbitraryFunction: Objective to minimise.
      %     lower  double vector: Lower corner of the box.
      %     upper  double vector: Upper corner of the box (each entry larger than in `lower`).
      %     surrogate  datamunge.BayesianSurrogate: Surrogate refitted before every proposal.
      %
      %   Output:
      %     r  datamunge.BayesianOptimizationResult: The evaluations and the best point.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the bounds are empty, differ in length or are not increasing. (C++ std::invalid_argument)
      %
      %   C++: BayesianOptimizationResult datamunge::optim::BayesianOptimization::run(ArbitraryFunction & function, const std::vector<double> & lower, const std::vector<double> & upper, BayesianSurrogate & surrogate) const
      r = __datamunge__(1328, obj, function_, lower, upper, surrogate);
    end
    function varargout = optimize (obj, varargin)
      % optimize  Runs the loop and returns only the best value.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates, lower, upper, surrogate)
      %
      %   Inputs:
      %     function_  datamunge.ArbitraryFunction: Objective to minimise.
      %     coordinates  double vector (updated: also an output): Output: the best point (its length must equal the bounds').
      %     lower  double vector: Lower corner of the box.
      %     upper  double vector: Upper corner of the box.
      %     surrogate  datamunge.BayesianSurrogate: Surrogate refitted before every proposal.
      %
      %   Output:
      %     r  double: The smallest observed value.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the bounds do not match `coordinates` or are not increasing. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::optim::BayesianOptimization::optimize(ArbitraryFunction & function, std::vector<double> & coordinates, const std::vector<double> & lower, const std::vector<double> & upper, BayesianSurrogate & surrogate) const
      [varargout{1:max(nargout, 1)}] = __datamunge__(1329, obj, varargin{:});
    end
    function r = options (obj)
      % options  The options this BayesianOptimization was configured with.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.BayesianOptimizationOptions: The stored options.
      %
      %   C++: const BayesianOptimizationOptions & datamunge::optim::BayesianOptimization::options() const
      r = __datamunge__(1330, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::BayesianOptimization. Do not edit.
