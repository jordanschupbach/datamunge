classdef CoordinateDescent < datamunge.NativeObject
  % CoordinateDescent  CoordinateDescent on the DifferentiableFunction callback interface. Each epoch visits coordinates in index order. Every partial update requests a full gradient and an Armijo-backtracked value sequence; this is not exact one-dimensional minimisation. An exhausted search restores that coordinate. Stops on the full end-of-epoch gradient norm. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.CoordinateDescent ()
  %   obj = datamunge.CoordinateDescent (options)
  %
  %   Example:
  %     const dm::optim::CoordinateDescent optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 1000
  %
  %   References:
  %   - S. J. Wright (2015), Coordinate descent algorithms, https://doi.org/10.1007/s10107-015-0892-3. This implementation uses the stated full-gradient coordinate/block steps.
  %
  %   Methods:
  %     optimize  Run CoordinateDescent from the supplied point using the configured finite budget. Each epoch visits coordinates in index order. Every partial update requests a full gradient and an Armijo-backtracked value sequence; this is not exact one-dimensional minimisation. An exhausted search restores that coordinate. Stops on the full end-of-epoch gradient norm.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::CoordinateDescent
  %
  %   See also: datamunge.CoordinateDescentOptions, datamunge.trace_optimize()
  methods
    function obj = CoordinateDescent (varargin)
      % CoordinateDescent  Construct a C++ datamunge::optim::CoordinateDescent.
      %
      %   obj = datamunge.CoordinateDescent ()
      %   obj = datamunge.CoordinateDescent (options)
      %
      %   Example:
      %     const dm::optim::CoordinateDescent optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 1000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.CoordinateDescentOptions, default the C++ default: Method-specific controls; defaults are listed by CoordinateDescentOptions.
      %
      %   C++: datamunge::optim::CoordinateDescent::CoordinateDescent(CoordinateDescentOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1375, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run CoordinateDescent from the supplied point using the configured finite budget. Each epoch visits coordinates in index order. Every partial update requests a full gradient and an Armijo-backtracked value sequence; this is not exact one-dimensional minimisation. An exhausted search restores that coordinate. Stops on the full end-of-epoch gradient norm.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::CoordinateDescentOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::CoordinateDescent(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.DifferentiableFunction: Borrowed DifferentiableFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  CoordinateDescent: step_size must be positive; CoordinateDescent: tolerance must be non-negative; CoordinateDescent: armijo_c1 must be in (0, 1); CoordinateDescent: backtracking_factor must be in (0, 1). (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::CoordinateDescent::optimize(DifferentiableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.CoordinateDescentOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1373, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.CoordinateDescentOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const CoordinateDescentOptions & datamunge::optim::CoordinateDescent::options() const
      r = __datamunge__(1374, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::CoordinateDescent. Do not edit.
