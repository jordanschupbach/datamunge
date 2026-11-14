classdef ProximalGradient < datamunge.NativeObject
  % ProximalGradient  ProximalGradient on the ProximalFunction callback interface. ISTA update prox(x-step_size*gradient(x),step_size), with gradient of the smooth part only. Stops on coordinate displacement norm. There is no Lipschitz estimation or backtracking; caller chooses a stable rate. The returned objective includes both smooth and nonsmooth parts. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.ProximalGradient ()
  %   obj = datamunge.ProximalGradient (options)
  %
  %   Example:
  %     const dm::optim::ProximalGradient optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 10000
  %
  %   References:
  %   - A. Beck and M. Teboulle (2009), A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems, https://doi.org/10.1137/080716542
  %
  %   Methods:
  %     optimize  Run ProximalGradient from the supplied point using the configured finite budget. ISTA update prox(x-step_size*gradient(x),step_size), with gradient of the smooth part only. Stops on coordinate displacement norm. There is no Lipschitz estimation or backtracking; caller chooses a stable rate. The returned objective includes both smooth and nonsmooth parts.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::ProximalGradient
  %
  %   See also: datamunge.ProximalGradientOptions, datamunge.trace_optimize()
  methods
    function obj = ProximalGradient (varargin)
      % ProximalGradient  Construct a C++ datamunge::optim::ProximalGradient.
      %
      %   obj = datamunge.ProximalGradient ()
      %   obj = datamunge.ProximalGradient (options)
      %
      %   Example:
      %     const dm::optim::ProximalGradient optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 10000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.ProximalGradientOptions, default the C++ default: Method-specific controls; defaults are listed by ProximalGradientOptions.
      %
      %   C++: datamunge::optim::ProximalGradient::ProximalGradient(ProximalGradientOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1719, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run ProximalGradient from the supplied point using the configured finite budget. ISTA update prox(x-step_size*gradient(x),step_size), with gradient of the smooth part only. Stops on coordinate displacement norm. There is no Lipschitz estimation or backtracking; caller chooses a stable rate. The returned objective includes both smooth and nonsmooth parts.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::L1LeastSquaresObjective f({{1}},{3},1);
      %     dm::optim::ProximalGradientOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0};
      %     const auto value=dm::optim::ProximalGradient(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.ProximalFunction: Borrowed ProximalFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  ProximalGradient: step_size must be positive; ProximalGradient: proximal returned wrong dimension. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::ProximalGradient::optimize(ProximalFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.ProximalGradientOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1717, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.ProximalGradientOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const ProximalGradientOptions & datamunge::optim::ProximalGradient::options() const
      r = __datamunge__(1718, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::ProximalGradient. Do not edit.
