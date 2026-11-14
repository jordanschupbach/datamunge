classdef Newton < datamunge.NativeObject
  % Newton  Newton on the HessianFunction callback interface. Adds fixed damping to the supplied Hessian diagonal, solves with partial-pivot Gaussian elimination, and takes a full step. This is not a globalised line-search method; indefinite or nearly singular Hessians can increase the loss or produce huge steps. Stops on gradient norm. Dense O(p^3) solve/O(p^2) storage. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.Newton ()
  %   obj = datamunge.Newton (options)
  %
  %   Example:
  %     const dm::optim::Newton optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 100
  %
  %   References:
  %   - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
  %
  %   Methods:
  %     optimize  Run Newton from the supplied point using the configured finite budget. Adds fixed damping to the supplied Hessian diagonal, solves with partial-pivot Gaussian elimination, and takes a full step. This is not a globalised line-search method; indefinite or nearly singular Hessians can increase the loss or produce huge steps. Stops on gradient norm. Dense O(p^3) solve/O(p^2) storage.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::Newton
  %
  %   See also: datamunge.NewtonOptions, datamunge.trace_optimize()
  methods
    function obj = Newton (varargin)
      % Newton  Construct a C++ datamunge::optim::Newton.
      %
      %   obj = datamunge.Newton ()
      %   obj = datamunge.Newton (options)
      %
      %   Example:
      %     const dm::optim::Newton optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 100
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.NewtonOptions, default the C++ default: Method-specific controls; defaults are listed by NewtonOptions.
      %
      %   C++: datamunge::optim::Newton::Newton(NewtonOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1665, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run Newton from the supplied point using the configured finite budget. Adds fixed damping to the supplied Hessian diagonal, solves with partial-pivot Gaussian elimination, and takes a full step. This is not a globalised line-search method; indefinite or nearly singular Hessians can increase the loss or produce huge steps. Stops on gradient norm. Dense O(p^3) solve/O(p^2) storage.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::NewtonOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::Newton(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.HessianFunction: Borrowed HessianFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  Newton: Hessian has wrong dimension; Newton: singular Hessian. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::Newton::optimize(HessianFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.NewtonOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1663, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.NewtonOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const NewtonOptions & datamunge::optim::Newton::options() const
      r = __datamunge__(1664, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::Newton. Do not edit.
