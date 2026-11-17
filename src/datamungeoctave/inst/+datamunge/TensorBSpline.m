classdef TensorBSpline < datamunge.NativeObject
  % TensorBSpline  A scalar tensor-product B-spline f(x) = sum_j c_j B_j(x) with owned coefficients.
  %
  %   obj = datamunge.TensorBSpline (basis, coefficients)
  %   obj = datamunge.TensorBSpline (bases, coefficients)
  %
  %   With one basis this is an ordinary spline curve; with d bases a surface or volume. The
  %   coefficient net is row-major over the per-axis basis sizes (last axis fastest), the same
  %   column order as fda::BSpline::evaluate(). Evaluation clamps each coordinate into the
  %   basis domain, so points outside the domain take the boundary value. Const methods are
  %   safe to call concurrently while the object is not mutated.
  %
  %   Example:
  %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);     // linear on [0, 1]
  %     dm::fda::TensorBSpline f({b, b}, {0, 1, 1, 2});         // f(x, y) = x + y
  %     std::print("{:.2f}\n", f.evaluate({0.25, 0.5}));        // 0.75
  %
  %   References:
  %   - C. de Boor (1972). On calculating with B-splines. Journal of Approximation Theory
  %     6(1), 50-62. https://doi.org/10.1016/0021-9045(72)90080-9
  %
  %   Methods:
  %     dimensions  Number of parameter dimensions d.
  %     bases  The per-axis bases.
  %     shape  Per-axis basis sizes, i.e. the shape of the coefficient net.
  %     coefficient_count  Total number of coefficients (product of shape()).
  %     coefficients  The row-major coefficient net.
  %     domain_min  Lower corner of the parameter domain (each basis's first clamped knot).
  %     domain_max  Upper corner of the parameter domain.
  %     coefficient  Reads one coefficient by per-axis basis index.
  %     set_coefficient  Overwrites one coefficient by per-axis basis index.
  %     evaluate  Evaluates a one-dimensional spline at `x` (clamped into the domain).
  %     evaluate_points  Evaluates at many points.
  %     evaluate_grid  Evaluates on the Cartesian product of per-axis sample values.
  %     dyadic_refine  Dyadically refines every axis and maps the coefficients, giving an identical function on the finer, nested tensor basis.
  %
  %   C++: class datamunge::fda::TensorBSpline
  %
  %   See also: datamunge.fit_tensor_bspline(), datamunge.fit_tensor_pspline(), datamunge.BSpline
  methods
    function obj = TensorBSpline (varargin)
      % TensorBSpline  Construct a C++ datamunge::fda::TensorBSpline.
      %
      %   obj = datamunge.TensorBSpline (basis, coefficients)
      %   obj = datamunge.TensorBSpline (bases, coefficients)
      %
      %   Example:
      %     dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {1.0, 3.0});
      %     std::print("{}\n", f.evaluate(0.5)); // 2
      %
      %   One-dimensional spline curve on `basis`.
      %
      %   Errors:
      %     datamunge:invalid_argument  If the coefficient count differs from basis.basis_size(). (C++ std::invalid_argument)
      %
      %   C++: datamunge::fda::TensorBSpline::TensorBSpline(BSplineBasis basis, std::vector<double> coefficients)
      %   C++: datamunge::fda::TensorBSpline::TensorBSpline(std::vector<BSplineBasis> bases, std::vector<double> coefficients)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3001, false, varargin{:});
    end
    function r = dimensions (obj)
      % dimensions  Number of parameter dimensions d.
      %
      %   r = obj.dimensions ()
      %
      %   Output:
      %     r  integer: bases().size().
      %
      %   C++: std::size_t datamunge::fda::TensorBSpline::dimensions() const
      r = __datamunge__(2988, obj);
    end
    function r = bases (obj)
      % bases  The per-axis bases.
      %
      %   r = obj.bases ()
      %
      %   Output:
      %     r  cell array of datamunge.BSplineBasis: Reference valid for the lifetime of this object.
      %
      %   C++: const std::vector<BSplineBasis> & datamunge::fda::TensorBSpline::bases() const
      r = __datamunge__(2989, obj);
    end
    function r = shape (obj)
      % shape  Per-axis basis sizes, i.e. the shape of the coefficient net.
      %
      %   r = obj.shape ()
      %
      %   Output:
      %     r  integer vector: d sizes in axis order.
      %
      %   C++: std::vector<int> datamunge::fda::TensorBSpline::shape() const
      r = __datamunge__(2990, obj);
    end
    function r = coefficient_count (obj)
      % coefficient_count  Total number of coefficients (product of shape()).
      %
      %   r = obj.coefficient_count ()
      %
      %   Output:
      %     r  integer: Coefficient count.
      %
      %   C++: std::size_t datamunge::fda::TensorBSpline::coefficient_count() const
      r = __datamunge__(2991, obj);
    end
    function r = coefficients (obj)
      % coefficients  The row-major coefficient net.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector: Reference valid until the next mutation of this object.
      %
      %   C++: const std::vector<double> & datamunge::fda::TensorBSpline::coefficients() const
      r = __datamunge__(2992, obj);
    end
    function r = domain_min (obj)
      % domain_min  Lower corner of the parameter domain (each basis's first clamped knot).
      %
      %   r = obj.domain_min ()
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   C++: std::vector<double> datamunge::fda::TensorBSpline::domain_min() const
      r = __datamunge__(2993, obj);
    end
    function r = domain_max (obj)
      % domain_max  Upper corner of the parameter domain.
      %
      %   r = obj.domain_max ()
      %
      %   Output:
      %     r  double vector: d coordinates.
      %
      %   C++: std::vector<double> datamunge::fda::TensorBSpline::domain_max() const
      r = __datamunge__(2994, obj);
    end
    function r = coefficient (obj, index)
      % coefficient  Reads one coefficient by per-axis basis index.
      %
      %   r = obj.coefficient (index)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TensorBSpline f({b, b}, {0, 1, 2, 3});
      %     std::print("{}\n", f.coefficient({1, 0})); // 2
      %
      %   Inputs:
      %     index  integer vector: d indices, each in [0, shape()[k]).
      %
      %   Output:
      %     r  double: The coefficient.
      %
      %   Errors:
      %     datamunge:invalid_argument  If index.size() != dimensions(). (C++ std::invalid_argument)
      %     datamunge:out_of_range  If a component is out of range. (C++ std::out_of_range)
      %
      %   C++: double datamunge::fda::TensorBSpline::coefficient(const std::vector<int> & index) const
      r = __datamunge__(2995, obj, index);
    end
    function set_coefficient (obj, index, value_)
      % set_coefficient  Overwrites one coefficient by per-axis basis index.
      %
      %   obj.set_coefficient (index, value_)
      %
      %   Example:
      %     dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {0, 0});
      %     f.set_coefficient({1}, 5.0);
      %     std::print("{}\n", f.evaluate(1.0)); // 5
      %
      %   Inputs:
      %     index  integer vector: d indices, each in [0, shape()[k]).
      %     value_  double: New coefficient.
      %
      %   Errors:
      %     datamunge:invalid_argument  If index.size() != dimensions(). (C++ std::invalid_argument)
      %     datamunge:out_of_range  If a component is out of range. (C++ std::out_of_range)
      %
      %   C++: void datamunge::fda::TensorBSpline::set_coefficient(const std::vector<int> & index, double value)
      __datamunge__(2996, obj, index, value_);
    end
    function r = evaluate (obj, varargin)
      % evaluate  Evaluates a one-dimensional spline at `x` (clamped into the domain).
      %
      %   r = obj.evaluate (x)
      %   r = obj.evaluate (point)
      %
      %   Example:
      %     dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {1.0, 3.0});
      %     std::print("{}\n", f.evaluate(2.0)); // 3  (clamped to x = 1)
      %
      %   Errors:
      %     datamunge:invalid_argument  If dimensions() != 1 or `x` is not finite. (C++ std::invalid_argument)
      %
      %   Overloads are chosen at run time from the argument types (C++):
      %   C++: double datamunge::fda::TensorBSpline::evaluate(double x) const
      %   C++: double datamunge::fda::TensorBSpline::evaluate(const std::vector<double> & point) const
      r = __datamunge__(2997, obj, varargin{:});
    end
    function r = evaluate_points (obj, points)
      % evaluate_points  Evaluates at many points.
      %
      %   r = obj.evaluate_points (points)
      %
      %   Example:
      %     dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(1, 2), {0.0, 1.0});
      %     std::print("{}\n", f.evaluate_points({{0.0}, {0.5}})); // [0, 0.5]
      %
      %   Inputs:
      %     points  double matrix (rows): Points, each of length d.
      %
      %   Output:
      %     r  double vector: One value per point, in input order.
      %
      %   Errors:
      %     datamunge:invalid_argument  As for evaluate(const std::vector<double>&). (C++ std::invalid_argument)
      %
      %   C++: std::vector<double> datamunge::fda::TensorBSpline::evaluate_points(const std::vector<std::vector<double>> & points) const
      r = __datamunge__(2998, obj, points);
    end
    function r = evaluate_grid (obj, axes)
      % evaluate_grid  Evaluates on the Cartesian product of per-axis sample values.
      %
      %   r = obj.evaluate_grid (axes)
      %
      %   Example:
      %     auto b = dm::fda::BSplineBasis::open_uniform(1, 2);
      %     dm::fda::TensorBSpline f({b, b}, {0, 1, 1, 2});
      %     std::print("{}\n", f.evaluate_grid({{0, 1}, {0, 1}})); // [0, 1, 1, 2]
      %
      %   Inputs:
      %     axes  double matrix (rows): d non-empty coordinate lists.
      %
      %   Output:
      %     r  double vector: Row-major grid of values (axis 0 slowest, last axis fastest).
      %
      %   Errors:
      %     datamunge:invalid_argument  If axes.size() != dimensions() or an axis is empty. (C++ std::invalid_argument)
      %
      %   C++: std::vector<double> datamunge::fda::TensorBSpline::evaluate_grid(const std::vector<std::vector<double>> & axes) const
      r = __datamunge__(2999, obj, axes);
    end
    function r = dyadic_refine (obj)
      % dyadic_refine  Dyadically refines every axis and maps the coefficients, giving an identical function on the finer, nested tensor basis.
      %
      %   r = obj.dyadic_refine ()
      %
      %   Example:
      %     dm::fda::TensorBSpline f(dm::fda::BSplineBasis::open_uniform(2, 4), {0, 1, 3, 2});
      %     auto g = f.dyadic_refine();
      %     std::print("{} {:.6f}\n", g.coefficient_count(), g.evaluate(0.3) - f.evaluate(0.3)); // 6 0.000000
      %
      %   Output:
      %     r  datamunge.TensorBSpline: The refined spline.
      %
      %   Errors:
      %     datamunge:invalid_argument  If some axis basis is not clamped. (C++ std::invalid_argument)
      %
      %   C++: TensorBSpline datamunge::fda::TensorBSpline::dyadic_refine() const
      %
      %   See also: datamunge.dyadic_refine(const BSplineBasis&)
      r = __datamunge__(3000, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::fda::TensorBSpline. Do not edit.
