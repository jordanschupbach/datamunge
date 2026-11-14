classdef AugmentedLagrangian < datamunge.NativeObject
  % AugmentedLagrangian  AugmentedLagrangian on the EqualityConstrainedFunction callback interface. Uses fixed inner gradient steps on f+lambda*c+rho*||c||^2/2, capped so each displacement norm is at most one. Updates multipliers after each inner budget and doubles rho until feasibility tolerance or outer budget. No dual-stationarity test is exposed. Objective return alone is not a KKT certificate. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.AugmentedLagrangian ()
  %   obj = datamunge.AugmentedLagrangian (options)
  %
  %   Example:
  %     const dm::optim::AugmentedLagrangian optimizer;
  %     std::print("{}\n",optimizer.options().max_outer_iterations); // 50
  %
  %   References:
  %   - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
  %
  %   Methods:
  %     optimize  Run AugmentedLagrangian from the supplied point using the configured finite budget. Uses fixed inner gradient steps on f+lambda*c+rho*||c||^2/2, capped so each displacement norm is at most one. Updates multipliers after each inner budget and doubles rho until feasibility tolerance or outer budget. No dual-stationarity test is exposed. Objective return alone is not a KKT certificate.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::AugmentedLagrangian
  %
  %   See also: datamunge.AugmentedLagrangianOptions, datamunge.trace_optimize()
  methods
    function obj = AugmentedLagrangian (varargin)
      % AugmentedLagrangian  Construct a C++ datamunge::optim::AugmentedLagrangian.
      %
      %   obj = datamunge.AugmentedLagrangian ()
      %   obj = datamunge.AugmentedLagrangian (options)
      %
      %   Example:
      %     const dm::optim::AugmentedLagrangian optimizer;
      %     std::print("{}\n",optimizer.options().max_outer_iterations); // 50
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.AugmentedLagrangianOptions, default the C++ default: Method-specific controls; defaults are listed by AugmentedLagrangianOptions.
      %
      %   C++: datamunge::optim::AugmentedLagrangian::AugmentedLagrangian(AugmentedLagrangianOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1300, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run AugmentedLagrangian from the supplied point using the configured finite budget. Uses fixed inner gradient steps on f+lambda*c+rho*||c||^2/2, capped so each displacement norm is at most one. Updates multipliers after each inner budget and doubles rho until feasibility tolerance or outer budget. No dual-stationarity test is exposed. Objective return alone is not a KKT certificate.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::EqualityQuadratic f({2,0},1);
      %     dm::optim::AugmentedLagrangianOptions controls;controls.max_outer_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::AugmentedLagrangian(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.EqualityConstrainedFunction: Borrowed EqualityConstrainedFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  AugmentedLagrangian: positive step_size and penalty required. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::AugmentedLagrangian::optimize(EqualityConstrainedFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.AugmentedLagrangianOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1298, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.AugmentedLagrangianOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const AugmentedLagrangianOptions & datamunge::optim::AugmentedLagrangian::options() const
      r = __datamunge__(1299, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::AugmentedLagrangian. Do not edit.
