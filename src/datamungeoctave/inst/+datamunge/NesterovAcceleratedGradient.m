classdef NesterovAcceleratedGradient < datamunge.NativeObject
  % NesterovAcceleratedGradient  NesterovAcceleratedGradient on the DifferentiableFunction callback interface. Evaluates the gradient at coordinates+momentum*velocity, then updates velocity and coordinates. Tests that look-ahead gradient after the update. Fixed momentum and rate do not implement every strongly-convex acceleration schedule. O(p) reset local state. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.NesterovAcceleratedGradient ()
  %   obj = datamunge.NesterovAcceleratedGradient (options)
  %
  %   Example:
  %     const dm::optim::NesterovAcceleratedGradient optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 10000
  %
  %   References:
  %   - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
  %
  %   Methods:
  %     optimize  Run NesterovAcceleratedGradient from the supplied point using the configured finite budget. Evaluates the gradient at coordinates+momentum*velocity, then updates velocity and coordinates. Tests that look-ahead gradient after the update. Fixed momentum and rate do not implement every strongly-convex acceleration schedule. O(p) reset local state.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::NesterovAcceleratedGradient
  %
  %   See also: datamunge.NesterovAcceleratedGradientOptions, datamunge.trace_optimize()
  methods
    function obj = NesterovAcceleratedGradient (varargin)
      % NesterovAcceleratedGradient  Construct a C++ datamunge::optim::NesterovAcceleratedGradient.
      %
      %   obj = datamunge.NesterovAcceleratedGradient ()
      %   obj = datamunge.NesterovAcceleratedGradient (options)
      %
      %   Example:
      %     const dm::optim::NesterovAcceleratedGradient optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 10000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.NesterovAcceleratedGradientOptions, default the C++ default: Method-specific controls; defaults are listed by NesterovAcceleratedGradientOptions.
      %
      %   C++: datamunge::optim::NesterovAcceleratedGradient::NesterovAcceleratedGradient(NesterovAcceleratedGradientOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1639, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run NesterovAcceleratedGradient from the supplied point using the configured finite budget. Evaluates the gradient at coordinates+momentum*velocity, then updates velocity and coordinates. Tests that look-ahead gradient after the update. Fixed momentum and rate do not implement every strongly-convex acceleration schedule. O(p) reset local state.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::NesterovAcceleratedGradientOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::NesterovAcceleratedGradient(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  NesterovAcceleratedGradient: step_size must be positive; NesterovAcceleratedGradient: momentum must be in [0, 1); NesterovAcceleratedGradient: tolerance must be non-negative. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::NesterovAcceleratedGradient::optimize(DifferentiableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.NesterovAcceleratedGradientOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1637, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.NesterovAcceleratedGradientOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const NesterovAcceleratedGradientOptions & datamunge::optim::NesterovAcceleratedGradient::options() const
      r = __datamunge__(1638, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::NesterovAcceleratedGradient. Do not edit.
