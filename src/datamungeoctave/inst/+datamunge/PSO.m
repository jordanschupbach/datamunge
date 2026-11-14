classdef PSO < datamunge.NativeObject
  % PSO  PSO on the ArbitraryFunction callback interface. Box-constrained particle swarm with global or ring attraction and constant or linearly decayed inertia. Coordinates seed one clamped particle; returns the best visited feasible point. Stops after more than twenty small-best-improvement iterations or its budget. O(population_size*p) storage; no global-optimality certificate. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.PSO ()
  %   obj = datamunge.PSO (options)
  %
  %   Example:
  %     const dm::optim::PSO optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 1000
  %
  %   References:
  %   - J. Kennedy and R. Eberhart (1995), Particle Swarm Optimization, https://doi.org/10.1109/ICNN.1995.488968. This implementation adds the documented inertia, topology and clamping controls.
  %
  %   Methods:
  %     optimize  Run PSO from the supplied point using the configured finite budget. Box-constrained particle swarm with global or ring attraction and constant or linearly decayed inertia. Coordinates seed one clamped particle; returns the best visited feasible point. Stops after more than twenty small-best-improvement iterations or its budget. O(population_size*p) storage; no global-optimality certificate.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::PSO
  %
  %   See also: datamunge.PSOOptions, datamunge.trace_optimize()
  methods
    function obj = PSO (varargin)
      % PSO  Construct a C++ datamunge::optim::PSO.
      %
      %   obj = datamunge.PSO ()
      %   obj = datamunge.PSO (options)
      %
      %   Example:
      %     const dm::optim::PSO optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 1000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.PSOOptions, default the C++ default: Method-specific controls; defaults are listed by PSOOptions.
      %
      %   C++: datamunge::optim::PSO::PSO(PSOOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1709, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run PSO from the supplied point using the configured finite budget. Box-constrained particle swarm with global or ring attraction and constant or linearly decayed inertia. Coordinates seed one clamped particle; returns the best visited feasible point. Stops after more than twenty small-best-improvement iterations or its budget. O(population_size*p) storage; no global-optimality certificate.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates, lower_bound, upper_bound)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::PSOOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::PSO(controls).optimize(f,x,std::vector<double>{-3,-3},std::vector<double>{3,3});
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
      %     datamunge:invalid_argument  PSO::optimize: population_size must be at least 2; PSO::optimize: topology must be 'global' or 'ring'; PSO::optimize: inertia_strategy must be 'constant' or 'linear_decay'. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::PSO::optimize(ArbitraryFunction & function, std::vector<double> & coordinates, const std::vector<double> & lower_bound, const std::vector<double> & upper_bound) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.PSOOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1707, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.PSOOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const PSOOptions & datamunge::optim::PSO::options() const
      r = __datamunge__(1708, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::PSO. Do not edit.
