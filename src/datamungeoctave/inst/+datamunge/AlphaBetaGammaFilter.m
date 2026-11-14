classdef AlphaBetaGammaFilter < datamunge.NativeObject
  % AlphaBetaGammaFilter  The constant-gain alpha-beta-gamma filter: AlphaBetaFilter's constant-ACCELERATION generalization (a 3-state g-h-k filter), adding a third fixed gain gamma for the acceleration correction. Same fixed-gain-vs-adaptive-Kalman-gain tradeoff as AlphaBetaFilter, one derivative order higher.
  %
  %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt)
  %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt, x0)
  %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt, x0, v0)
  %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt, x0, v0, a0)
  %
  %   Methods:
  %     update  C++ datamunge::filter::AlphaBetaGammaFilter::update.
  %     position  C++ datamunge::filter::AlphaBetaGammaFilter::position.
  %     velocity  C++ datamunge::filter::AlphaBetaGammaFilter::velocity.
  %     acceleration  C++ datamunge::filter::AlphaBetaGammaFilter::acceleration.
  %     alpha  The position-correction gain alpha.
  %     beta  The velocity-correction gain beta.
  %     gamma  The acceleration-correction gain gamma.
  %     dt  The time step between measurements.
  %     filter  C++ datamunge::filter::AlphaBetaGammaFilter::filter.
  %
  %   C++: class datamunge::filter::AlphaBetaGammaFilter
  methods
    function obj = AlphaBetaGammaFilter (varargin)
      % AlphaBetaGammaFilter  Construct a C++ datamunge::filter::AlphaBetaGammaFilter.
      %
      %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt)
      %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt, x0)
      %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt, x0, v0)
      %   obj = datamunge.AlphaBetaGammaFilter (alpha, beta, gamma, dt, x0, v0, a0)
      %
      %   Inputs:
      %     alpha  double
      %     beta  double
      %     gamma  double
      %     dt  double
      %     x0  double, default 0.0
      %     v0  double, default 0.0
      %     a0  double, default 0.0
      %
      %   C++: datamunge::filter::AlphaBetaGammaFilter::AlphaBetaGammaFilter(double alpha, double beta, double gamma, double dt, double x0 = 0.0, double v0 = 0.0, double a0 = 0.0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 787, false, varargin{:});
    end
    function update (obj, z)
      % update  C++ datamunge::filter::AlphaBetaGammaFilter::update.
      %
      %   obj.update (z)
      %
      %   Inputs:
      %     z  double
      %
      %   C++: void datamunge::filter::AlphaBetaGammaFilter::update(double z)
      __datamunge__(778, obj, z);
    end
    function r = position (obj)
      % position  C++ datamunge::filter::AlphaBetaGammaFilter::position.
      %
      %   r = obj.position ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::position() const
      r = __datamunge__(779, obj);
    end
    function r = velocity (obj)
      % velocity  C++ datamunge::filter::AlphaBetaGammaFilter::velocity.
      %
      %   r = obj.velocity ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::velocity() const
      r = __datamunge__(780, obj);
    end
    function r = acceleration (obj)
      % acceleration  C++ datamunge::filter::AlphaBetaGammaFilter::acceleration.
      %
      %   r = obj.acceleration ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::acceleration() const
      r = __datamunge__(781, obj);
    end
    function r = alpha (obj)
      % alpha  The position-correction gain alpha.
      %
      %   r = obj.alpha ()
      %
      %   Output:
      %     r  double: The value passed to the constructor.
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::alpha() const
      r = __datamunge__(782, obj);
    end
    function r = beta (obj)
      % beta  The velocity-correction gain beta.
      %
      %   r = obj.beta ()
      %
      %   Output:
      %     r  double: The value passed to the constructor.
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::beta() const
      r = __datamunge__(783, obj);
    end
    function r = gamma (obj)
      % gamma  The acceleration-correction gain gamma.
      %
      %   r = obj.gamma ()
      %
      %   Output:
      %     r  double: The value passed to the constructor.
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::gamma() const
      r = __datamunge__(784, obj);
    end
    function r = dt (obj)
      % dt  The time step between measurements.
      %
      %   r = obj.dt ()
      %
      %   Output:
      %     r  double: The value passed to the constructor.
      %
      %   C++: double datamunge::filter::AlphaBetaGammaFilter::dt() const
      r = __datamunge__(785, obj);
    end
    function r = filter (obj, measurements)
      % filter  C++ datamunge::filter::AlphaBetaGammaFilter::filter.
      %
      %   r = obj.filter (measurements)
      %
      %   Inputs:
      %     measurements  double vector
      %
      %   Output:
      %     r  cell array of datamunge.AlphaBetaGammaState
      %
      %   C++: std::vector<AlphaBetaGammaState> datamunge::filter::AlphaBetaGammaFilter::filter(const std::vector<double> & measurements)
      r = __datamunge__(786, obj, measurements);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::filter::AlphaBetaGammaFilter. Do not edit.
