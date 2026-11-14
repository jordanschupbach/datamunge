classdef DifferentialEvolution < datamunge.NativeObject
  % DifferentialEvolution  DifferentialEvolution on the ArbitraryFunction callback interface. Box-constrained differential mutation/crossover with greedy trial replacement. Supports the declared rand1/best1/current_to_best1/rand2 and binomial/exponential strategies. Coordinates seed one clamped member; more than twenty small-best-improvement generations can stop the run. O(population_size*p) storage; no global certificate. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.DifferentialEvolution ()
  %   obj = datamunge.DifferentialEvolution (options)
  %
  %   Example:
  %     const dm::optim::DifferentialEvolution optimizer;
  %     std::print("{}\n",optimizer.options().max_generations); // 500
  %
  %   References:
  %   - R. Storn and K. Price (1997), Differential Evolution - A Simple and Efficient Heuristic for global Optimization over Continuous Spaces, https://doi.org/10.1023/A:1008202821328
  %
  %   Methods:
  %     optimize  Run DifferentialEvolution from the supplied point using the configured finite budget. Box-constrained differential mutation/crossover with greedy trial replacement. Supports the declared rand1/best1/current_to_best1/rand2 and binomial/exponential strategies. Coordinates seed one clamped member; more than twenty small-best-improvement generations can stop the run. O(population_size*p) storage; no global certificate.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::DifferentialEvolution
  %
  %   See also: datamunge.DEOptions, datamunge.trace_optimize()
  methods
    function obj = DifferentialEvolution (varargin)
      % DifferentialEvolution  Construct a C++ datamunge::optim::DifferentialEvolution.
      %
      %   obj = datamunge.DifferentialEvolution ()
      %   obj = datamunge.DifferentialEvolution (options)
      %
      %   Example:
      %     const dm::optim::DifferentialEvolution optimizer;
      %     std::print("{}\n",optimizer.options().max_generations); // 500
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.DEOptions, default the C++ default: Method-specific controls; defaults are listed by DEOptions.
      %
      %   C++: datamunge::optim::DifferentialEvolution::DifferentialEvolution(DEOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1431, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run DifferentialEvolution from the supplied point using the configured finite budget. Box-constrained differential mutation/crossover with greedy trial replacement. Supports the declared rand1/best1/current_to_best1/rand2 and binomial/exponential strategies. Coordinates seed one clamped member; more than twenty small-best-improvement generations can stop the run. O(population_size*p) storage; no global certificate.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates, lower_bound, upper_bound)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::DEOptions controls;controls.max_generations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::DifferentialEvolution(controls).optimize(f,x,std::vector<double>{-3,-3},std::vector<double>{3,3});
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.ArbitraryFunction: Borrowed ArbitraryFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %     lower_bound  double vector: Matching finite inclusive lower box limits.
      %     upper_bound  double vector: Matching finite inclusive upper limits, greater than lower limits.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  DifferentialEvolution::optimize: crossover_strategy must be 'binomial' or 'exponential'. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::DifferentialEvolution::optimize(ArbitraryFunction & function, std::vector<double> & coordinates, const std::vector<double> & lower_bound, const std::vector<double> & upper_bound) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.DEOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1429, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.DEOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const DEOptions & datamunge::optim::DifferentialEvolution::options() const
      r = __datamunge__(1430, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::DifferentialEvolution. Do not edit.
