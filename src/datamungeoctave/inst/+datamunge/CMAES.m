classdef CMAES < datamunge.NativeObject
  % CMAES  CMAES on the ArbitraryFunction callback interface. Unbounded covariance-adaptation evolution strategy with weighted elites, evolution paths and global step-size adaptation. Default offspring count is 4+floor(3*log(p)). Covariance eigensolve is dense with numerical eigenvalue floors. Stops on search-scale/axis tolerance or generation budget; stochastic results depend on seed and population, not a global certificate. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.CMAES ()
  %   obj = datamunge.CMAES (options)
  %
  %   Example:
  %     const dm::optim::CMAES optimizer;
  %     std::print("{}\n",optimizer.options().max_generations); // 500
  %
  %   References:
  %   - N. Hansen and A. Ostermeier (2001), Completely Derandomized Self-Adaptation in Evolution Strategies, https://doi.org/10.1162/106365601750190398
  %
  %   Methods:
  %     optimize  Run CMAES from the supplied point using the configured finite budget. Unbounded covariance-adaptation evolution strategy with weighted elites, evolution paths and global step-size adaptation. Default offspring count is 4+floor(3*log(p)). Covariance eigensolve is dense with numerical eigenvalue floors. Stops on search-scale/axis tolerance or generation budget; stochastic results depend on seed and population, not a global certificate.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::CMAES
  %
  %   See also: datamunge.CMAESOptions, datamunge.trace_optimize()
  methods
    function obj = CMAES (varargin)
      % CMAES  Construct a C++ datamunge::optim::CMAES.
      %
      %   obj = datamunge.CMAES ()
      %   obj = datamunge.CMAES (options)
      %
      %   Example:
      %     const dm::optim::CMAES optimizer;
      %     std::print("{}\n",optimizer.options().max_generations); // 500
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.CMAESOptions, default the C++ default: Method-specific controls; defaults are listed by CMAESOptions.
      %
      %   C++: datamunge::optim::CMAES::CMAES(CMAESOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1359, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run CMAES from the supplied point using the configured finite budget. Unbounded covariance-adaptation evolution strategy with weighted elites, evolution paths and global step-size adaptation. Default offspring count is 4+floor(3*log(p)). Covariance eigensolve is dense with numerical eigenvalue floors. Stops on search-scale/axis tolerance or generation budget; stochastic results depend on seed and population, not a global certificate.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::CMAESOptions controls;controls.max_generations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::CMAES(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  CMAES: initial_step_size must be positive; CMAES: tolerance must be non-negative; CMAES: population_size must be at least 2. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::CMAES::optimize(ArbitraryFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.CMAESOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1357, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.CMAESOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const CMAESOptions & datamunge::optim::CMAES::options() const
      r = __datamunge__(1358, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::CMAES. Do not edit.
