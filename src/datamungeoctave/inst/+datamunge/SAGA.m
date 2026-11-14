classdef SAGA < datamunge.NativeObject
  % SAGA  SAGA on the DifferentiableSeparableFunction callback interface. Uses current_i-table_i+mean(table), then refreshes one stored component gradient and its running mean. The estimator targets mean component loss while evaluate() returns their sum. Stops on epoch-boundary objective change or the last estimator norm; the latter is not the full gradient norm. O(n*p) gradient table. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.SAGA ()
  %   obj = datamunge.SAGA (options)
  %
  %   Example:
  %     const dm::optim::SAGA optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 10000
  %
  %   References:
  %   - A. Defazio, F. Bach and S. Lacoste-Julien (2014), SAGA: A Fast Incremental Gradient Method With Support for Non-Strongly Convex Composite Objectives, https://arxiv.org/abs/1407.0202
  %
  %   Methods:
  %     optimize  Run SAGA from the supplied point using the configured finite budget. Uses current_i-table_i+mean(table), then refreshes one stored component gradient and its running mean. The estimator targets mean component loss while evaluate() returns their sum. Stops on epoch-boundary objective change or the last estimator norm; the latter is not the full gradient norm. O(n*p) gradient table.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::SAGA
  %
  %   See also: datamunge.SAGAOptions, datamunge.trace_optimize()
  methods
    function obj = SAGA (varargin)
      % SAGA  Construct a C++ datamunge::optim::SAGA.
      %
      %   obj = datamunge.SAGA ()
      %   obj = datamunge.SAGA (options)
      %
      %   Example:
      %     const dm::optim::SAGA optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 10000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.SAGAOptions, default the C++ default: Method-specific controls; defaults are listed by SAGAOptions.
      %
      %   C++: datamunge::optim::SAGA::SAGA(SAGAOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1836, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run SAGA from the supplied point using the configured finite budget. Uses current_i-table_i+mean(table), then refreshes one stored component gradient and its running mean. The estimator targets mean component loss while evaluate() returns their sum. Stops on epoch-boundary objective change or the last estimator norm; the latter is not the full gradient norm. O(n*p) gradient table.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::LeastSquaresObjective f({{1},{1}},{1,3});
      %     dm::optim::SAGAOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0};
      %     const auto value=dm::optim::SAGA(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  SAGA: step_size must be positive; SAGA: tolerance must be non-negative; SAGA: function has zero terms. (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::SAGA::optimize(DifferentiableSeparableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.SAGAOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1834, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.SAGAOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const SAGAOptions & datamunge::optim::SAGA::options() const
      r = __datamunge__(1835, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::SAGA. Do not edit.
