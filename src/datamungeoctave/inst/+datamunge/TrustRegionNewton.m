classdef TrustRegionNewton < datamunge.NativeObject
  % TrustRegionNewton  TrustRegionNewton on the HessianFunction callback interface. Cauchy-point variant: clips a steepest-descent ray using gHg and the current radius. Accepts any strict objective decrease and doubles the radius, otherwise halves it. It does not compute a model-agreement ratio, dogleg step or exact trust-region subproblem. Stops on gradient norm or tiny radius; O(p^2) Hessian work. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.TrustRegionNewton ()
  %   obj = datamunge.TrustRegionNewton (options)
  %
  %   Example:
  %     const dm::optim::TrustRegionNewton optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 200
  %
  %   References:
  %   - J. Nocedal and S. J. Wright (2006), Numerical Optimization, second edition, https://doi.org/10.1007/978-0-387-40065-5. The implemented variant and stopping rules are specified above.
  %
  %   Methods:
  %     optimize  Run TrustRegionNewton from the supplied point using the configured finite budget. Cauchy-point variant: clips a steepest-descent ray using gHg and the current radius. Accepts any strict objective decrease and doubles the radius, otherwise halves it. It does not compute a model-agreement ratio, dogleg step or exact trust-region subproblem. Stops on gradient norm or tiny radius; O(p^2) Hessian work.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::TrustRegionNewton
  %
  %   See also: datamunge.TrustRegionNewtonOptions, datamunge.trace_optimize()
  methods
    function obj = TrustRegionNewton (varargin)
      % TrustRegionNewton  Construct a C++ datamunge::optim::TrustRegionNewton.
      %
      %   obj = datamunge.TrustRegionNewton ()
      %   obj = datamunge.TrustRegionNewton (options)
      %
      %   Example:
      %     const dm::optim::TrustRegionNewton optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 200
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.TrustRegionNewtonOptions, default the C++ default: Method-specific controls; defaults are listed by TrustRegionNewtonOptions.
      %
      %   C++: datamunge::optim::TrustRegionNewton::TrustRegionNewton(TrustRegionNewtonOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1874, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run TrustRegionNewton from the supplied point using the configured finite budget. Cauchy-point variant: clips a steepest-descent ray using gHg and the current radius. Accepts any strict objective decrease and doubles the radius, otherwise halves it. It does not compute a model-agreement ratio, dogleg step or exact trust-region subproblem. Stops on gradient norm or tiny radius; O(p^2) Hessian work.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::TrustRegionNewtonOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::TrustRegionNewton(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  TrustRegionNewton: initial_radius must be positive. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::TrustRegionNewton::optimize(HessianFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.TrustRegionNewtonOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1872, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.TrustRegionNewtonOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const TrustRegionNewtonOptions & datamunge::optim::TrustRegionNewton::options() const
      r = __datamunge__(1873, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::TrustRegionNewton. Do not edit.
