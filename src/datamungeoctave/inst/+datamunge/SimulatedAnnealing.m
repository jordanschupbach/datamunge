classdef SimulatedAnnealing < datamunge.NativeObject
  % SimulatedAnnealing  SimulatedAnnealing on the ArbitraryFunction callback interface. Gaussian random-walk proposals with geometrically decaying temperature and Metropolis acceptance of worse values. Returns the best visited point, not the last accepted state. A finite cooling schedule does not certify a global optimum; O(p) storage. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.SimulatedAnnealing ()
  %   obj = datamunge.SimulatedAnnealing (options)
  %
  %   Example:
  %     const dm::optim::SimulatedAnnealing optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 10000
  %
  %   References:
  %   - S. Kirkpatrick, C. D. Gelatt and M. P. Vecchi (1983), Optimization by Simulated Annealing, https://doi.org/10.1126/science.220.4598.671. Finite geometric cooling here supplies no global guarantee.
  %
  %   Methods:
  %     optimize  Run SimulatedAnnealing from the supplied point using the configured finite budget. Gaussian random-walk proposals with geometrically decaying temperature and Metropolis acceptance of worse values. Returns the best visited point, not the last accepted state. A finite cooling schedule does not certify a global optimum; O(p) storage.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::SimulatedAnnealing
  %
  %   See also: datamunge.SimulatedAnnealingOptions, datamunge.trace_optimize()
  methods
    function obj = SimulatedAnnealing (varargin)
      % SimulatedAnnealing  Construct a C++ datamunge::optim::SimulatedAnnealing.
      %
      %   obj = datamunge.SimulatedAnnealing ()
      %   obj = datamunge.SimulatedAnnealing (options)
      %
      %   Example:
      %     const dm::optim::SimulatedAnnealing optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 10000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.SimulatedAnnealingOptions, default the C++ default: Method-specific controls; defaults are listed by SimulatedAnnealingOptions.
      %
      %   C++: datamunge::optim::SimulatedAnnealing::SimulatedAnnealing(SimulatedAnnealingOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1824, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run SimulatedAnnealing from the supplied point using the configured finite budget. Gaussian random-walk proposals with geometrically decaying temperature and Metropolis acceptance of worse values. Returns the best visited point, not the last accepted state. A finite cooling schedule does not certify a global optimum; O(p) storage.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::SimulatedAnnealingOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::SimulatedAnnealing(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.ArbitraryFunction: Borrowed ArbitraryFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::SimulatedAnnealing::optimize(ArbitraryFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.SimulatedAnnealingOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1822, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.SimulatedAnnealingOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const SimulatedAnnealingOptions & datamunge::optim::SimulatedAnnealing::options() const
      r = __datamunge__(1823, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::SimulatedAnnealing. Do not edit.
