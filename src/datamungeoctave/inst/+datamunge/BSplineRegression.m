classdef BSplineRegression < datamunge.NativeObject
  % BSplineRegression  A least-squares B-spline regression curve y ≈ Σ_j β_j B_j(x). `dm::BSR` for short.
  %
  %   obj = datamunge.BSplineRegression (x, y, n_basis)
  %   obj = datamunge.BSplineRegression (x, y, n_basis, degree)
  %   obj = datamunge.BSplineRegression (x, y, knots)
  %   obj = datamunge.BSplineRegression (x, y, knots, degree)
  %
  %   The fit is ordinary least squares (no roughness penalty). Throws std::invalid_argument for
  %   mismatched/empty inputs or too few basis functions for the degree, and std::runtime_error when
  %   the design is rank deficient (some basis function has too little data under it; use fewer
  %   basis functions / knots).
  %
  %   Methods:
  %     predict  Evaluates the fitted curve at `x`. Points outside the fitted domain take the value at the nearest boundary, i.e. constant extrapolation.
  %     coefficients  C++ datamunge::BSplineRegression::coefficients.
  %     fitted  C++ datamunge::BSplineRegression::fitted.
  %     residuals  C++ datamunge::BSplineRegression::residuals.
  %     r_squared  C++ datamunge::BSplineRegression::r_squared.
  %     basis_size  C++ datamunge::BSplineRegression::basis_size.
  %     degree  C++ datamunge::BSplineRegression::degree.
  %     knots  The full knot sequence (boundary knots repeated degree + 1 times).
  %     domain  The [min, max] interval the basis covers.
  %
  %   C++: class datamunge::BSplineRegression
  methods
    function obj = BSplineRegression (varargin)
      % BSplineRegression  Construct a C++ datamunge::BSplineRegression.
      %
      %   obj = datamunge.BSplineRegression (x, y, n_basis)
      %   obj = datamunge.BSplineRegression (x, y, n_basis, degree)
      %   obj = datamunge.BSplineRegression (x, y, knots)
      %   obj = datamunge.BSplineRegression (x, y, knots, degree)
      %
      %   Fixed-width basis: `n_basis` B-splines of the given `degree` (cubic by default) whose knots are equally spaced over the range of `x`.
      %
      %   C++: datamunge::BSplineRegression::BSplineRegression(const Vector<> & x, const Vector<> & y, std::size_t n_basis, std::size_t degree = 3)
      %   C++: datamunge::BSplineRegression::BSplineRegression(const Vector<> & x, const Vector<> & y, const Vector<> & knots, std::size_t degree = 3)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2024, false, varargin{:});
    end
    function r = predict (obj, x)
      % predict  Evaluates the fitted curve at `x`. Points outside the fitted domain take the value at the nearest boundary, i.e. constant extrapolation.
      %
      %   r = obj.predict (x)
      %
      %   Inputs:
      %     x  value convertible to std::vector<double>
      %
      %   Output:
      %     r  value convertible to std::vector<double>
      %
      %   C++: Vector<> datamunge::BSplineRegression::predict(const Vector<> & x) const
      r = __datamunge__(2015, obj, x);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::BSplineRegression::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>
      %
      %   C++: Vector<> datamunge::BSplineRegression::coefficients() const
      r = __datamunge__(2016, obj);
    end
    function r = fitted (obj)
      % fitted  C++ datamunge::BSplineRegression::fitted.
      %
      %   r = obj.fitted ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>
      %
      %   C++: Vector<> datamunge::BSplineRegression::fitted() const
      r = __datamunge__(2017, obj);
    end
    function r = residuals (obj)
      % residuals  C++ datamunge::BSplineRegression::residuals.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>
      %
      %   C++: Vector<> datamunge::BSplineRegression::residuals() const
      r = __datamunge__(2018, obj);
    end
    function r = r_squared (obj)
      % r_squared  C++ datamunge::BSplineRegression::r_squared.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::BSplineRegression::r_squared() const
      r = __datamunge__(2019, obj);
    end
    function r = basis_size (obj)
      % basis_size  C++ datamunge::BSplineRegression::basis_size.
      %
      %   r = obj.basis_size ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::BSplineRegression::basis_size() const
      r = __datamunge__(2020, obj);
    end
    function r = degree (obj)
      % degree  C++ datamunge::BSplineRegression::degree.
      %
      %   r = obj.degree ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::BSplineRegression::degree() const
      r = __datamunge__(2021, obj);
    end
    function r = knots (obj)
      % knots  The full knot sequence (boundary knots repeated degree + 1 times).
      %
      %   r = obj.knots ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>
      %
      %   C++: Vector<> datamunge::BSplineRegression::knots() const
      r = __datamunge__(2022, obj);
    end
    function r = domain (obj)
      % domain  The [min, max] interval the basis covers.
      %
      %   r = obj.domain ()
      %
      %   Output:
      %     r  cell {double, double}
      %
      %   C++: std::pair<double, double> datamunge::BSplineRegression::domain() const
      r = __datamunge__(2023, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::BSplineRegression. Do not edit.
