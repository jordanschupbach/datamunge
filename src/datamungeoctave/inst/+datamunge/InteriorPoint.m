classdef InteriorPoint < datamunge.NativeObject
  % InteriorPoint  InteriorPoint on the InequalityConstrainedFunction callback interface. Gradient steps on f-mu*sum(log(-c)) with c<0. Backtracks only for strict feasibility, not barrier-objective decrease, then decays mu after each fixed inner budget. Requires a strictly feasible start and can fail later when feasible steps underflow. No KKT or barrier-stationarity stop test; no phase-I feasibility search. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.InteriorPoint ()
  %   obj = datamunge.InteriorPoint (options)
  %
  %   Example:
  %     const dm::optim::InteriorPoint optimizer;
  %     std::print("{}\n",optimizer.options().max_outer_iterations); // 20
  %
  %   References:
  %   - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
  %
  %   Methods:
  %     optimize  Run InteriorPoint from the supplied point using the configured finite budget. Gradient steps on f-mu*sum(log(-c)) with c<0. Backtracks only for strict feasibility, not barrier-objective decrease, then decays mu after each fixed inner budget. Requires a strictly feasible start and can fail later when feasible steps underflow. No KKT or barrier-stationarity stop test; no phase-I feasibility search.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::InteriorPoint
  %
  %   See also: datamunge.InteriorPointOptions, datamunge.trace_optimize()
  methods
    function obj = InteriorPoint (varargin)
      % InteriorPoint  Construct a C++ datamunge::optim::InteriorPoint.
      %
      %   obj = datamunge.InteriorPoint ()
      %   obj = datamunge.InteriorPoint (options)
      %
      %   Example:
      %     const dm::optim::InteriorPoint optimizer;
      %     std::print("{}\n",optimizer.options().max_outer_iterations); // 20
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.InteriorPointOptions, default the C++ default: Method-specific controls; defaults are listed by InteriorPointOptions.
      %
      %   C++: datamunge::optim::InteriorPoint::InteriorPoint(InteriorPointOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1583, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run InteriorPoint from the supplied point using the configured finite budget. Gradient steps on f-mu*sum(log(-c)) with c<0. Backtracks only for strict feasibility, not barrier-objective decrease, then decays mu after each fixed inner budget. Requires a strictly feasible start and can fail later when feasible steps underflow. No KKT or barrier-stationarity stop test; no phase-I feasibility search.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::NonnegativeQuadratic f({-1,2});
      %     dm::optim::InteriorPointOptions controls;controls.max_outer_iterations=1;
      %     std::vector<double> x{.5,.5};
      %     const auto value=dm::optim::InteriorPoint(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.InequalityConstrainedFunction: Borrowed InequalityConstrainedFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  InteriorPoint: starting point must be strictly feasible. (C++ std::invalid_argument)
      %     datamunge:native  InteriorPoint: unable to find feasible step. (C++ std::runtime_error)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::InteriorPoint::optimize(InequalityConstrainedFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.InteriorPointOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1581, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.InteriorPointOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const InteriorPointOptions & datamunge::optim::InteriorPoint::options() const
      r = __datamunge__(1582, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::InteriorPoint. Do not edit.
