classdef ConjugateGradient < datamunge.NativeObject
  % ConjugateGradient  ConjugateGradient on the DifferentiableFunction callback interface. Nonlinear Polak-Ribiere-plus directions with negative beta truncated to zero, periodic/descent restarts and an Armijo line search. It is not linear CG with exact line minimisation and does not imply termination in p steps. Returns when gradient/change/search conditions stop progress. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.ConjugateGradient ()
  %   obj = datamunge.ConjugateGradient (options)
  %
  %   Example:
  %     const dm::optim::ConjugateGradient optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 1000
  %
  %   References:
  %   - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
  %
  %   Methods:
  %     optimize  Run ConjugateGradient from the supplied point using the configured finite budget. Nonlinear Polak-Ribiere-plus directions with negative beta truncated to zero, periodic/descent restarts and an Armijo line search. It is not linear CG with exact line minimisation and does not imply termination in p steps. Returns when gradient/change/search conditions stop progress.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::ConjugateGradient
  %
  %   See also: datamunge.ConjugateGradientOptions, datamunge.trace_optimize()
  methods
    function obj = ConjugateGradient (varargin)
      % ConjugateGradient  Construct a C++ datamunge::optim::ConjugateGradient.
      %
      %   obj = datamunge.ConjugateGradient ()
      %   obj = datamunge.ConjugateGradient (options)
      %
      %   Example:
      %     const dm::optim::ConjugateGradient optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 1000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.ConjugateGradientOptions, default the C++ default: Method-specific controls; defaults are listed by ConjugateGradientOptions.
      %
      %   C++: datamunge::optim::ConjugateGradient::ConjugateGradient(ConjugateGradientOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1345, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run ConjugateGradient from the supplied point using the configured finite budget. Nonlinear Polak-Ribiere-plus directions with negative beta truncated to zero, periodic/descent restarts and an Armijo line search. It is not linear CG with exact line minimisation and does not imply termination in p steps. Returns when gradient/change/search conditions stop progress.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::ConjugateGradientOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::ConjugateGradient(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  ConjugateGradient: tolerance must be non-negative; ConjugateGradient: armijo_c1 must be in (0, 1); ConjugateGradient: backtracking_factor must be in (0, 1). (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::ConjugateGradient::optimize(DifferentiableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.ConjugateGradientOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1343, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.ConjugateGradientOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const ConjugateGradientOptions & datamunge::optim::ConjugateGradient::options() const
      r = __datamunge__(1344, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::ConjugateGradient. Do not edit.
