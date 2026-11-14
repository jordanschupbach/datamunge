classdef AMSGrad < datamunge.NativeObject
  % AMSGrad  AMSGrad on the DifferentiableFunction callback interface. Bias-corrected AMSGrad variant: maximises uncorrected second moments, then divides that maximum by 1-beta2^t. The corrected denominator need not be monotone, so the original uncorrected theorem is not automatically inherited. Tests the gradient used after updating; O(p) reset state. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.AMSGrad ()
  %   obj = datamunge.AMSGrad (options)
  %
  %   Example:
  %     const dm::optim::AMSGrad optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 10000
  %
  %   References:
  %   - S. Reddi, S. Kale and S. Kumar (2018), On the Convergence of Adam and Beyond, https://openreview.net/forum?id=ryQu7f-RZ
  %
  %   Methods:
  %     optimize  Run AMSGrad from the supplied point using the configured finite budget. Bias-corrected AMSGrad variant: maximises uncorrected second moments, then divides that maximum by 1-beta2^t. The corrected denominator need not be monotone, so the original uncorrected theorem is not automatically inherited. Tests the gradient used after updating; O(p) reset state.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::AMSGrad
  %
  %   See also: datamunge.AMSGradOptions, datamunge.trace_optimize()
  methods
    function obj = AMSGrad (varargin)
      % AMSGrad  Construct a C++ datamunge::optim::AMSGrad.
      %
      %   obj = datamunge.AMSGrad ()
      %   obj = datamunge.AMSGrad (options)
      %
      %   Example:
      %     const dm::optim::AMSGrad optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 10000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.AMSGradOptions, default the C++ default: Method-specific controls; defaults are listed by AMSGradOptions.
      %
      %   C++: datamunge::optim::AMSGrad::AMSGrad(AMSGradOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1272, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run AMSGrad from the supplied point using the configured finite budget. Bias-corrected AMSGrad variant: maximises uncorrected second moments, then divides that maximum by 1-beta2^t. The corrected denominator need not be monotone, so the original uncorrected theorem is not automatically inherited. Tests the gradient used after updating; O(p) reset state.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::AMSGradOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::AMSGrad(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  AMSGrad: step_size must be positive; AMSGrad: beta1 must be in [0, 1); AMSGrad: beta2 must be in [0, 1); AMSGrad: epsilon must be positive; AMSGrad: tolerance must be non-negative. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::AMSGrad::optimize(DifferentiableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.AMSGradOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1270, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.AMSGradOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const AMSGradOptions & datamunge::optim::AMSGrad::options() const
      r = __datamunge__(1271, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::AMSGrad. Do not edit.
