classdef RandomizedBlockCoordinateDescent < datamunge.NativeObject
  % RandomizedBlockCoordinateDescent  RandomizedBlockCoordinateDescent on the DifferentiableFunction callback interface. Randomly permutes coordinates each epoch and visits consecutive blocks once. Full-gradient block directions use Armijo backtracking and restore a failed block. This is random reshuffling, not independent coordinate draws. Stops on the full epoch-end gradient norm; O(p) local storage. The caller supplies finite, dimensionally consistent callbacks. Inputs are borrowed; coordinates mutate in place and can retain a partial state if a callback throws. Calls use local optimizer state and propagate callback exceptions; sharing a mutable objective across threads requires external synchronisation. Normal return alone is not a uniform convergence certificate.
  %
  %   obj = datamunge.RandomizedBlockCoordinateDescent ()
  %   obj = datamunge.RandomizedBlockCoordinateDescent (options)
  %
  %   Example:
  %     const dm::optim::RandomizedBlockCoordinateDescent optimizer;
  %     std::print("{}\n",optimizer.options().max_iterations); // 1000
  %
  %   References:
  %   - S. J. Wright (2015), Coordinate descent algorithms, https://doi.org/10.1007/s10107-015-0892-3. This implementation uses the stated full-gradient coordinate/block steps.
  %
  %   Methods:
  %     optimize  Run RandomizedBlockCoordinateDescent from the supplied point using the configured finite budget. Randomly permutes coordinates each epoch and visits consecutive blocks once. Full-gradient block directions use Armijo backtracking and restore a failed block. This is random reshuffling, not independent coordinate draws. Stops on the full epoch-end gradient norm; O(p) local storage.
  %     options  Borrow this optimizer's stored configuration.
  %
  %   C++: class datamunge::optim::RandomizedBlockCoordinateDescent
  %
  %   See also: datamunge.RandomizedBlockCoordinateDescentOptions, datamunge.trace_optimize()
  methods
    function obj = RandomizedBlockCoordinateDescent (varargin)
      % RandomizedBlockCoordinateDescent  Construct a C++ datamunge::optim::RandomizedBlockCoordinateDescent.
      %
      %   obj = datamunge.RandomizedBlockCoordinateDescent ()
      %   obj = datamunge.RandomizedBlockCoordinateDescent (options)
      %
      %   Example:
      %     const dm::optim::RandomizedBlockCoordinateDescent optimizer;
      %     std::print("{}\n",optimizer.options().max_iterations); // 1000
      %
      %   Store a copy of the controls without running the objective.
      %
      %   Inputs:
      %     options  datamunge.RandomizedBlockCoordinateDescentOptions, default the C++ default: Method-specific controls; defaults are listed by RandomizedBlockCoordinateDescentOptions.
      %
      %   C++: datamunge::optim::RandomizedBlockCoordinateDescent::RandomizedBlockCoordinateDescent(RandomizedBlockCoordinateDescentOptions options = {})
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 1739, false, varargin{:});
    end
    function varargout = optimize (obj, varargin)
      % optimize  Run RandomizedBlockCoordinateDescent from the supplied point using the configured finite budget. Randomly permutes coordinates each epoch and visits consecutive blocks once. Full-gradient block directions use Armijo backtracking and restore a failed block. This is random reshuffling, not independent coordinate draws. Stops on the full epoch-end gradient norm; O(p) local storage.
      %
      %   [r, coordinates] = obj.optimize (function_, coordinates)
      %
      %   Example:
      %     dm::optim::QuadraticObjective f({2,8},{1,-2});
      %     dm::optim::RandomizedBlockCoordinateDescentOptions controls;controls.max_iterations=5;
      %     std::vector<double> x{0,0};
      %     const auto value=dm::optim::RandomizedBlockCoordinateDescent(controls).optimize(f,x);
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
      %     datamunge:invalid_argument  RandomizedBlockCoordinateDescent: block_size must be positive; RandomizedBlockCoordinateDescent: step_size must be positive; RandomizedBlockCoordinateDescent: tolerance must be non-negative; RandomizedBlockCoordinateDescent: armijo_c1 must be in (0, 1); RandomizedBlockCoordinateDescent: backtracking_factor must be in (0, 1). (C++ std::invalid_argument)
      %     datamunge:native  Exceptions from supplied callbacks propagate. Raw solvers do not uniformly reject nonfinite inputs. (C++ std::exception)
      %
      %   C++: double datamunge::optim::RandomizedBlockCoordinateDescent::optimize(DifferentiableFunction & function, std::vector<double> & coordinates) const
      %
      %   See also: datamunge.trace_optimize(), datamunge.RandomizedBlockCoordinateDescentOptions
      [varargout{1:max(nargout, 1)}] = __datamunge__(1737, obj, varargin{:});
    end
    function r = options (obj)
      % options  Borrow this optimizer's stored configuration.
      %
      %   r = obj.options ()
      %
      %   Output:
      %     r  datamunge.RandomizedBlockCoordinateDescentOptions: Const option reference, valid during this object's lifetime; no mutation.
      %
      %   C++: const RandomizedBlockCoordinateDescentOptions & datamunge::optim::RandomizedBlockCoordinateDescent::options() const
      r = __datamunge__(1738, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::optim::RandomizedBlockCoordinateDescent. Do not edit.
