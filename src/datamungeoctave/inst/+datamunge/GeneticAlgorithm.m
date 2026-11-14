classdef GeneticAlgorithm < datamunge.NativeObject
  % GeneticAlgorithm  GeneticAlgorithm on the ArbitraryFunction callback interface. Real-valued box-constrained selection, crossover, Gaussian mutation and optional elitism. Initial coordinates seed a clamped population member. Returns the best visited point and can stop after more than twenty small-best-improvement generations. O(population_size*p) storage; strategy/rate choices affect results, with no global certificate. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.GeneticAlgorithm ()
  %   obj = datamunge.GeneticAlgorithm (options)
  %
  %   Example:
  %     const dm::optim::GeneticAlgorithm optimizer;
  %     std::print("{}\n",optimizer.options().max_generations); // 500
  %
  %   References:
  %   - J. H. Holland (1992 reprint), Adaptation in Natural and Artificial Systems, MIT Press, https://mitpress.mit.edu/9780262581110/adaptation-in-natural-and-artificial-systems/. This implementation uses real-valued operators, as specified above, rather than binary chromosomes.
  %
  %   Methods:
  %     optimize  Run GeneticAlgorithm from the supplied point using the configured finite budget. Real-valued box-constrained selection, crossover, Gaussian mutation and optional elitism. Initial coordinates seed a clamped population member. Returns the best visited point and can stop after more than twenty small-best-improvement generations. O(population_size*p) storage; strategy/rate choices affect results, with no global certificate.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::GeneticAlgorithm
  %
  %   See also: datamunge.GAOptions, datamunge.trace_optimize()
  methods
    function obj = GeneticAlgorithm (varargin)
      % GeneticAlgorithm  Construct a C++ datamunge::optim::GeneticAlgorithm.
      %
      %   obj = datamunge.GeneticAlgorithm ()
      %   obj = datamunge.GeneticAlgorithm (options)
      %
      %   Example:
      %     const dm::optim::GeneticAlgorithm optimizer;
      %     std::print("{}\n",optimizer.options().max_generations); // 500
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.GAOptions, default the C++ default: Method-specific controls; defaults are listed by GAOptions.
      %
      %   C++: datamunge::optim::GeneticAlgorithm::GeneticAlgorithm(GAOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1527, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run GeneticAlgorithm from the supplied point using the configured finite budget. Real-valued box-constrained selection, crossover, Gaussian mutation and optional elitism. Initial coordinates seed a clamped population member. Returns the best visited point and can stop after more than twenty small-best-improvement generations. O(population_size*p) storage; strategy/rate choices affect results, with no global certificate.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates, lower_bound, upper_bound)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::GAOptions controls;controls.max_generations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::GeneticAlgorithm(controls).optimize(f,x,std::vector<double>{-3,-3},std::vector<double>{3,3});
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
      %     datamunge:invalid_argument  GeneticAlgorithm::optimize: population_size must be at least 2. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::GeneticAlgorithm::optimize(ArbitraryFunction & function, std::vector<double> & coordinates, const std::vector<double> & lower_bound, const std::vector<double> & upper_bound) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.GAOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1525, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.GAOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const GAOptions & datamunge::optim::GeneticAlgorithm::options() const
      r = __datamunge__(1526, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::GeneticAlgorithm. Do not edit.
