classdef SGD < datamunge.NativeObject
  % SGD  SGD on the DifferentiableSeparableFunction callback interface. Uses the mean gradient of each mini-batch, reshuffling component indices each epoch when enabled. Full objective remains the sum of components. Stops on absolute full-objective change per epoch. Nonempty components and positive batch_size are required; seeds reproduce the same build, not a cross-library shuffle transcript. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.SGD ()
  %   obj = datamunge.SGD (options)
  %
  %   Example:
  %     const dm::optim::SGD optimizer;
  %     std::print("{}\n",optimizer.options().max_epochs); // 100
  %
  %   References:
  %   - H. Robbins and S. Monro (1951), A Stochastic Approximation Method, https://doi.org/10.1214/aoms/1177729586. The fixed-rate reshuffling variant here differs from their decreasing-step stochastic approximation.
  %
  %   Methods:
  %     optimize  Run SGD from the supplied point using the configured finite budget. Uses the mean gradient of each mini-batch, reshuffling component indices each epoch when enabled. Full objective remains the sum of components. Stops on absolute full-objective change per epoch. Nonempty components and positive batch_size are required; seeds reproduce the same build, not a cross-library shuffle transcript.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::SGD
  %
  %   See also: datamunge.SGDOptions, datamunge.trace_optimize()
  methods
    function obj = SGD (varargin)
      % SGD  Construct a C++ datamunge::optim::SGD.
      %
      %   obj = datamunge.SGD ()
      %   obj = datamunge.SGD (options)
      %
      %   Example:
      %     const dm::optim::SGD optimizer;
      %     std::print("{}\n",optimizer.options().max_epochs); // 100
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.SGDOptions, default the C++ default: Method-specific controls; defaults are listed by SGDOptions.
      %
      %   C++: datamunge::optim::SGD::SGD(SGDOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1810, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run SGD from the supplied point using the configured finite budget. Uses the mean gradient of each mini-batch, reshuffling component indices each epoch when enabled. Full objective remains the sum of components. Stops on absolute full-objective change per epoch. Nonempty components and positive batch_size are required; seeds reproduce the same build, not a cross-library shuffle transcript.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
      %     dm::optim::SGDOptions controls;controls.max_epochs=5;
      %     std::vector<double> x{0};
      %     const auto value=dm::optim::SGD(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  SGD: batch_size must be positive; SGD::optimize: function has zero terms. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::SGD::optimize(DifferentiableSeparableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.SGDOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1808, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.SGDOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const SGDOptions & datamunge::optim::SGD::options() const
      r = __datamunge__(1809, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::SGD. Do not edit.
