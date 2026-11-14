classdef SVRG < datamunge.NativeObject
  % SVRG  SVRG on the DifferentiableSeparableFunction callback interface. Uses mean component gradients: current_i-snapshot_i+full_snapshot_gradient/n. The division by n is essential for an unbiased mean-loss estimator and matches SGD/SAGA scale. Earlier code omitted it. Stops on absolute full-sum objective change per epoch. O(p) snapshot state, two component-gradient calls per inner update plus one full gradient per epoch. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.SVRG ()
  %   obj = datamunge.SVRG (options)
  %
  %   Example:
  %     const dm::optim::SVRG optimizer;
  %     std::print("{}\n",optimizer.options().max_epochs); // 100
  %
  %   References:
  %   - R. Johnson and T. Zhang (2013), Accelerating Stochastic Gradient Descent using Predictive Variance Reduction, https://proceedings.neurips.cc/paper/2013/hash/ac1dd209cbcc5e5d1c6e28598e8cbbe8-Abstract.html
  %
  %   Methods:
  %     optimize  Run SVRG from the supplied point using the configured finite budget. Uses mean component gradients: current_i-snapshot_i+full_snapshot_gradient/n. The division by n is essential for an unbiased mean-loss estimator and matches SGD/SAGA scale. Earlier code omitted it. Stops on absolute full-sum objective change per epoch. O(p) snapshot state, two component-gradient calls per inner update plus one full gradient per epoch.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::SVRG
  %
  %   See also: datamunge.SVRGOptions, datamunge.trace_optimize()
  methods
    function obj = SVRG (varargin)
      % SVRG  Construct a C++ datamunge::optim::SVRG.
      %
      %   obj = datamunge.SVRG ()
      %   obj = datamunge.SVRG (options)
      %
      %   Example:
      %     const dm::optim::SVRG optimizer;
      %     std::print("{}\n",optimizer.options().max_epochs); // 100
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.SVRGOptions, default the C++ default: Method-specific controls; defaults are listed by SVRGOptions.
      %
      %   C++: datamunge::optim::SVRG::SVRG(SVRGOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1850, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run SVRG from the supplied point using the configured finite budget. Uses mean component gradients: current_i-snapshot_i+full_snapshot_gradient/n. The division by n is essential for an unbiased mean-loss estimator and matches SGD/SAGA scale. Earlier code omitted it. Stops on absolute full-sum objective change per epoch. O(p) snapshot state, two component-gradient calls per inner update plus one full gradient per epoch.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
      %     dm::optim::SVRGOptions controls;controls.max_epochs=5;
      %     std::vector<double> x{0};
      %     const auto value=dm::optim::SVRG(controls).optimize(f,x);
      %     std::print("{}\n",std::isfinite(value)); // true
      %
      %   Inputs:
      %     function_  datamunge.DifferentiableSeparableFunction: Borrowed DifferentiableSeparableFunction with matching finite output shapes.
      %     coordinates  double vector (updated: also an output): Owned-by-caller finite start; overwritten with the native returned point.
      %
      %   Output:
      %     r  double: Objective at the native returned coordinates; not a stop-status code.
      %
      %   Errors:
      %     datamunge:invalid_argument  SVRG: step_size must be positive; SVRG: tolerance must be non-negative; SVRG: function has zero terms. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::SVRG::optimize(DifferentiableSeparableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.SVRGOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1848, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.SVRGOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const SVRGOptions & datamunge::optim::SVRG::options() const
      r = __datamunge__(1849, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::SVRG. Do not edit.
