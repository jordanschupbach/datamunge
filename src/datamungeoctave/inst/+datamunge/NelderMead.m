classdef NelderMead < datamunge.NativeObject
  % NelderMead  NelderMead on the ArbitraryFunction callback interface. Local simplex reflection, expansion, contraction and shrink search. Starts with per-coordinate displacements initial_simplex_scale and stops when simplex diameter and value spread are small. Those conditions do not prove stationarity or a global minimum. O(p^2) simplex storage. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.NelderMead ()
  %   obj = datamunge.NelderMead (options)
  %
  %   Example:
  %     const dm::optim::NelderMead optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 2000
  %
  %   References:
  %   - J. A. Nelder and R. Mead (1965), A Simplex Method for Function Minimization, https://doi.org/10.1093/comjnl/7.4.308
  %
  %   Methods:
  %     optimize  Run NelderMead from the supplied point using the configured finite budget. Local simplex reflection, expansion, contraction and shrink search. Starts with per-coordinate displacements initial_simplex_scale and stops when simplex diameter and value spread are small. Those conditions do not prove stationarity or a global minimum. O(p^2) simplex storage.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::NelderMead
  %
  %   See also: datamunge.NelderMeadOptions, datamunge.trace_optimize()
  methods
    function obj = NelderMead (varargin)
      % NelderMead  Construct a C++ datamunge::optim::NelderMead.
      %
      %   obj = datamunge.NelderMead ()
      %   obj = datamunge.NelderMead (options)
      %
      %   Example:
      %     const dm::optim::NelderMead optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 2000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.NelderMeadOptions, default the C++ default: Method-specific controls; defaults are listed by NelderMeadOptions.
      %
      %   C++: datamunge::optim::NelderMead::NelderMead(NelderMeadOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1627, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run NelderMead from the supplied point using the configured finite budget. Local simplex reflection, expansion, contraction and shrink search. Starts with per-coordinate displacements initial_simplex_scale and stops when simplex diameter and value spread are small. Those conditions do not prove stationarity or a global minimum. O(p^2) simplex storage.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::NelderMeadOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::NelderMead(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  NelderMead: initial_simplex_scale must be positive; NelderMead: reflection must be positive; NelderMead: expansion must be greater than 1; NelderMead: contraction must be in (0, 1); NelderMead: shrink must be in (0, 1); NelderMead: tolerance must be non-negative. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::NelderMead::optimize(ArbitraryFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.NelderMeadOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1625, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.NelderMeadOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const NelderMeadOptions & datamunge::optim::NelderMead::options() const
      r = __datamunge__(1626, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::NelderMead. Do not edit.
