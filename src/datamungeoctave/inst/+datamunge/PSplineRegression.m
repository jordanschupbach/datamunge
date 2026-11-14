classdef PSplineRegression < datamunge.NativeObject
  % PSplineRegression  A P-spline regression curve y ≈ Σ_j β_j B_j(x) (Eilers & Marx). `dm::PSR` for short.
  %
  %   obj = datamunge.PSplineRegression (x, y, n_basis)
  %   obj = datamunge.PSplineRegression (x, y, n_basis, lambda)
  %   obj = datamunge.PSplineRegression (x, y, n_basis, lambda, degree)
  %   obj = datamunge.PSplineRegression (x, y, n_basis, lambda, degree, penalty_order)
  %   obj = datamunge.PSplineRegression (x, y, knots)
  %   obj = datamunge.PSplineRegression (x, y, knots, lambda)
  %   obj = datamunge.PSplineRegression (x, y, knots, lambda, degree)
  %   obj = datamunge.PSplineRegression (x, y, knots, lambda, degree, penalty_order)
  %
  %   Minimizes ||y - Bβ||² + λ ||D_d β||², where D_d is the order-d difference matrix on the
  %   coefficients (d = 2 by default). Uniform knots continue past both ends of the data
  %   without repeated boundary knots. When d <= degree+1, polynomials of degree < d
  %   go unpenalized: λ → ∞ gives the least-squares line for d = 2. A rich basis can
  %   include functions with no data under them if the data identify the penalty nullspace.
  %   Without an explicit λ, it is chosen
  %   by minimizing generalized cross-validation, GCV(λ) = n·RSS / (n − edf)².
  %
  %   The polynomial-nullspace claim requires uniform spacing; it need not hold for irregular
  %   supplied knots. A positive lambda identifies a rich basis only when the design identifies
  %   the penalty nullspace. No values are dropped. The class owns its fit and permits concurrent
  %   const reads. Sparse assembly and dense Cholesky cost O(n(p+1)^2+K^3); GCV repeats
  %   this work. Very large lambda can make the equations numerically singular.
  %
  %   Example:
  %     const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
  %     std::print("{:.6f} {:.6f}\n",fit.predict({1})[0],fit.effective_df()); // 0.428571 2.142857
  %
  %   References:
  %   - P. H. C. Eilers and B. D. Marx (1996). Flexible smoothing with B-splines and penalties.
  %     https://doi.org/10.1214/ss/1038425655
  %
  %   Methods:
  %     predict  Evaluates the fitted curve at `x`. Points outside the fitted domain take the value at the nearest boundary, i.e. constant extrapolation.
  %     coefficients  Fitted coefficients in basis order.
  %     fitted  Predictions at the training coordinates.
  %     residuals  Observed minus fitted training responses.
  %     r_squared  In-sample 1-RSS/TSS, with the constant-response convention of PSplineFitResult.
  %     lambda  The smoothing parameter λ used for the fit (the GCV minimizer when not given).
  %     penalty_order  Order of adjacent coefficient differences.
  %     effective_df  Effective degrees of freedom, trace of the hat matrix.
  %     gcv  Generalized cross-validation score n·RSS / (n − edf)² of the fit.
  %     basis_size  Number of basis functions, including boundary functions.
  %     degree  Polynomial degree of each spline segment.
  %     knots  The full knot sequence, including the `degree` extra knots past each end of the data.
  %     domain  The [min, max] interval the basis covers.
  %     coefficient_differences  Fitted order-d coefficient differences, whose squared norm is roughness.
  %
  %   C++: class datamunge::PSplineRegression
  %
  %   See also: datamunge.fit_pspline(), datamunge.smoothing_path(), datamunge.summary()
  methods
    function obj = PSplineRegression (varargin)
      % PSplineRegression  Construct a C++ datamunge::PSplineRegression.
      %
      %   obj = datamunge.PSplineRegression (x, y, n_basis)
      %   obj = datamunge.PSplineRegression (x, y, n_basis, lambda)
      %   obj = datamunge.PSplineRegression (x, y, n_basis, lambda, degree)
      %   obj = datamunge.PSplineRegression (x, y, n_basis, lambda, degree, penalty_order)
      %   obj = datamunge.PSplineRegression (x, y, knots)
      %   obj = datamunge.PSplineRegression (x, y, knots, lambda)
      %   obj = datamunge.PSplineRegression (x, y, knots, lambda, degree)
      %   obj = datamunge.PSplineRegression (x, y, knots, lambda, degree, penalty_order)
      %
      %   Example:
      %     const dm::Vector<> x=dm::seql(0,1,30);
      %     const dm::PSR fit(x,dm::sin(6*x),10);
      %     std::print("{}\n",fit.lambda()>0); // true
      %
      %   References:
      %   - G. H. Golub, M. Heath and G. Wahba (1979). Generalized cross-validation as a
      %     method for choosing a good ridge parameter. https://doi.org/10.1080/00401706.1979.10489751
      %
      %   `n_basis` cubic B-splines with equally spaced knots over the range of `x`; λ chosen by GCV. Searches log10(lambda) on [-8,8] in steps of 0.5, then uses golden-section refinement around the best grid point. Singular candidates score infinity. This bounded refinement does not guarantee a global continuous minimum. Penalty order is two.
      %
      %   Errors:
      %     datamunge:invalid_argument  For invalid data or basis size. (C++ std::invalid_argument)
      %     datamunge:native  If all candidate systems fail numerically. (C++ std::runtime_error)
      %
      %   C++: datamunge::PSplineRegression::PSplineRegression(const Vector<> & x, const Vector<> & y, std::size_t n_basis)
      %   C++: datamunge::PSplineRegression::PSplineRegression(const Vector<> & x, const Vector<> & y, std::size_t n_basis, double lambda, std::size_t degree = 3, std::size_t penalty_order = 2)
      %   C++: datamunge::PSplineRegression::PSplineRegression(const Vector<> & x, const Vector<> & y, const Vector<> & knots)
      %   C++: datamunge::PSplineRegression::PSplineRegression(const Vector<> & x, const Vector<> & y, const Vector<> & knots, double lambda, std::size_t degree = 3, std::size_t penalty_order = 2)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2039, false, varargin{:});
    end
    function r = predict (obj, x)
      % predict  Evaluates the fitted curve at `x`. Points outside the fitted domain take the value at the nearest boundary, i.e. constant extrapolation.
      %
      %   r = obj.predict (x)
      %
      %   Example:
      %     const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
      %     std::print("{:.6f}\n",fit.predict({-1})[0]); // 0.285714
      %
      %   Inputs:
      %     x  value convertible to std::vector<double>: Finite evaluation coordinates; empty is accepted.
      %
      %   Output:
      %     r  value convertible to std::vector<double>: Owned predictions in input order, after clamping x to domain().
      %
      %   Errors:
      %     datamunge:invalid_argument  For non-finite coordinates. (C++ std::invalid_argument)
      %
      %   C++: Vector<> datamunge::PSplineRegression::predict(const Vector<> & x) const
      %
      %   See also: datamunge.basis_matrix(), datamunge.domain()
      r = __datamunge__(2025, obj, x);
    end
    function r = coefficients (obj)
      % coefficients  Fitted coefficients in basis order.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>: Owned K-vector.
      %
      %   C++: Vector<> datamunge::PSplineRegression::coefficients() const
      r = __datamunge__(2026, obj);
    end
    function r = fitted (obj)
      % fitted  Predictions at the training coordinates.
      %
      %   r = obj.fitted ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>: Owned n-vector in training order.
      %
      %   C++: Vector<> datamunge::PSplineRegression::fitted() const
      r = __datamunge__(2027, obj);
    end
    function r = residuals (obj)
      % residuals  Observed minus fitted training responses.
      %
      %   r = obj.residuals ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>: Owned n-vector in training order.
      %
      %   C++: Vector<> datamunge::PSplineRegression::residuals() const
      r = __datamunge__(2028, obj);
    end
    function r = r_squared (obj)
      % r_squared  In-sample 1-RSS/TSS, with the constant-response convention of PSplineFitResult.
      %
      %   r = obj.r_squared ()
      %
      %   Output:
      %     r  double: Scalar R squared.
      %
      %   C++: double datamunge::PSplineRegression::r_squared() const
      r = __datamunge__(2029, obj);
    end
    function r = lambda (obj)
      % lambda  The smoothing parameter λ used for the fit (the GCV minimizer when not given).
      %
      %   r = obj.lambda ()
      %
      %   Output:
      %     r  double: Finite nonnegative multiplier used in the fit.
      %
      %   C++: double datamunge::PSplineRegression::lambda() const
      r = __datamunge__(2030, obj);
    end
    function r = penalty_order (obj)
      % penalty_order  Order of adjacent coefficient differences.
      %
      %   r = obj.penalty_order ()
      %
      %   Output:
      %     r  integer: Positive difference order, independent of spline degree.
      %
      %   C++: std::size_t datamunge::PSplineRegression::penalty_order() const
      r = __datamunge__(2031, obj);
    end
    function r = effective_df (obj)
      % effective_df  Effective degrees of freedom, trace of the hat matrix.
      %
      %   r = obj.effective_df ()
      %
      %   Output:
      %     r  double: Fixed-lambda smoother trace, the sum of training leverages.
      %
      %   C++: double datamunge::PSplineRegression::effective_df() const
      r = __datamunge__(2032, obj);
    end
    function r = gcv (obj)
      % gcv  Generalized cross-validation score n·RSS / (n − edf)² of the fit.
      %
      %   r = obj.gcv ()
      %
      %   Output:
      %     r  double: Fixed-lambda score in response-squared units; infinity when n-edf is numerically zero.
      %
      %   C++: double datamunge::PSplineRegression::gcv() const
      %
      %   See also: datamunge.gcv(), datamunge.PSplineRegression
      r = __datamunge__(2033, obj);
    end
    function r = basis_size (obj)
      % basis_size  Number of basis functions, including boundary functions.
      %
      %   r = obj.basis_size ()
      %
      %   Output:
      %     r  integer: Basis dimension K.
      %
      %   C++: std::size_t datamunge::PSplineRegression::basis_size() const
      r = __datamunge__(2034, obj);
    end
    function r = degree (obj)
      % degree  Polynomial degree of each spline segment.
      %
      %   r = obj.degree ()
      %
      %   Output:
      %     r  integer: Degree p, independent of penalty order.
      %
      %   C++: std::size_t datamunge::PSplineRegression::degree() const
      r = __datamunge__(2035, obj);
    end
    function r = knots (obj)
      % knots  The full knot sequence, including the `degree` extra knots past each end of the data.
      %
      %   r = obj.knots ()
      %
      %   Output:
      %     r  value convertible to std::vector<double>: Owned sequence of K+degree+1 knots.
      %
      %   C++: Vector<> datamunge::PSplineRegression::knots() const
      r = __datamunge__(2036, obj);
    end
    function r = domain (obj)
      % domain  The [min, max] interval the basis covers.
      %
      %   r = obj.domain ()
      %
      %   Output:
      %     r  cell {double, double}: Closed interval endpoints (minimum,maximum).
      %
      %   C++: std::pair<double, double> datamunge::PSplineRegression::domain() const
      r = __datamunge__(2037, obj);
    end
    function r = coefficient_differences (obj)
      % coefficient_differences  Fitted order-d coefficient differences, whose squared norm is roughness.
      %
      %   r = obj.coefficient_differences ()
      %
      %   Example:
      %     const dm::PSR fit({0,1,2},{0,1,0},3,1.0,1,2);
      %     std::print("{:.6f}\n",fit.coefficient_differences()[0]); // -0.285714
      %
      %   Output:
      %     r  value convertible to std::vector<double>: Owned K-d vector in left-to-right difference order.
      %
      %   C++: Vector<> datamunge::PSplineRegression::coefficient_differences() const
      %
      %   See also: datamunge.summary(), datamunge.pspline_difference_matrix()
      r = __datamunge__(2038, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::PSplineRegression. Do not edit.
