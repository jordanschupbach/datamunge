classdef LevenbergMarquardt < datamunge.NativeObject
  % LevenbergMarquardt  LevenbergMarquardt on the ResidualFunction callback interface. Forms J transpose J plus scalar damping*I and solves by unpivoted elimination. Accepts only strict reductions of half residual SSE, multiplying damping by .3 on acceptance and 10 on rejection. Tests proposed step norm even after rejection. A small rejected step is not a gradient certificate; no parameter constraints. O(np^2+p^3) work. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.LevenbergMarquardt ()
  %   obj = datamunge.LevenbergMarquardt (options)
  %
  %   Example:
  %     const dm::optim::LevenbergMarquardt optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 200
  %
  %   References:
  %   - K. Levenberg (1944), A Method for the Solution of Certain Non-Linear Problems in Least Squares, https://doi.org/10.1090/qam/10666; D. W. Marquardt (1963), An Algorithm for Least-Squares Estimation of Nonlinear Parameters, https://doi.org/10.1137/0111030
  %
  %   Methods:
  %     optimize  Run LevenbergMarquardt from the supplied point using the configured finite budget. Forms J transpose J plus scalar damping*I and solves by unpivoted elimination. Accepts only strict reductions of half residual SSE, multiplying damping by .3 on acceptance and 10 on rejection. Tests proposed step norm even after rejection. A small rejected step is not a gradient certificate; no parameter constraints. O(np^2+p^3) work.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::LevenbergMarquardt
  %
  %   See also: datamunge.LevenbergMarquardtOptions, datamunge.trace_optimize()
  methods
    function obj = LevenbergMarquardt (varargin)
      % LevenbergMarquardt  Construct a C++ datamunge::optim::LevenbergMarquardt.
      %
      %   obj = datamunge.LevenbergMarquardt ()
      %   obj = datamunge.LevenbergMarquardt (options)
      %
      %   Example:
      %     const dm::optim::LevenbergMarquardt optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 200
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.LevenbergMarquardtOptions, default the C++ default: Method-specific controls; defaults are listed by LevenbergMarquardtOptions.
      %
      %   C++: datamunge::optim::LevenbergMarquardt::LevenbergMarquardt(LevenbergMarquardtOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1609, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run LevenbergMarquardt from the supplied point using the configured finite budget. Forms J transpose J plus scalar damping*I and solves by unpivoted elimination. Accepts only strict reductions of half residual SSE, multiplying damping by .3 on acceptance and 10 on rejection. Tests proposed step norm even after rejection. A small rejected step is not a gradient certificate; no parameter constraints. O(np^2+p^3) work.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     const auto data=dm::optim::optimization_curve_data();dm::optim::CurveResidual f(data);
      %     dm::optim::LevenbergMarquardtOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{2,.5};
      %     const auto value=dm::optim::LevenbergMarquardt(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.ResidualFunction: Borrowed ResidualFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates (half residual SSE); not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  LevenbergMarquardt: initial_damping must be positive. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::LevenbergMarquardt::optimize(ResidualFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.LevenbergMarquardtOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1607, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.LevenbergMarquardtOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const LevenbergMarquardtOptions & datamunge::optim::LevenbergMarquardt::options() const
      r = __datamunge__(1608, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::LevenbergMarquardt. Do not edit.
