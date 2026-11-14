classdef Polynomial < datamunge.NativeObject
  % Polynomial  Dense univariate polynomial over the reals, stored as coefficients in ascending degree order (coefficients()[i] is the coefficient of x^i). Always kept trimmed -- no trailing (highest-degree) zero coefficients, except for the zero polynomial itself, which is stored as the single coefficient {0.0}.
  %
  %   obj = datamunge.Polynomial ()
  %   obj = datamunge.Polynomial (constant)
  %   obj = datamunge.Polynomial (coeffs)
  %
  %   Methods:
  %     degree  Degree of the polynomial; the zero polynomial has degree 0 (there is no universally-agreed convention for its "true" degree of -infinity, and 0 keeps every coefficient()/loop bound in this module well-defined).
  %     is_zero  C++ datamunge::algebra::Polynomial::is_zero.
  %     coefficient  Coefficient of x^i; 0.0 for any i outside [0, degree()] (including negative i).
  %     coefficients  C++ datamunge::algebra::Polynomial::coefficients.
  %     evaluate  Evaluates the polynomial at x via Horner's method.
  %     derivative  C++ datamunge::algebra::Polynomial::derivative.
  %     antiderivative  The antiderivative with zero constant term (i.e. integral from 0 to x, as a polynomial in x).
  %     add  C++ datamunge::algebra::Polynomial::add.
  %     subtract  C++ datamunge::algebra::Polynomial::subtract.
  %     multiply  C++ datamunge::algebra::Polynomial::multiply.
  %     scale  C++ datamunge::algebra::Polynomial::scale.
  %     negate  C++ datamunge::algebra::Polynomial::negate.
  %     to_string  C++ datamunge::algebra::Polynomial::to_string.
  %     to_tex  Render descending powers using a raw TeX symbol and 1-17 significant digits; return raw TeX.
  %
  %   Operators: ==, ~=.
  %
  %   C++: class datamunge::algebra::Polynomial
  methods
    function obj = Polynomial (varargin)
      % Polynomial  Construct a C++ datamunge::algebra::Polynomial.
      %
      %   obj = datamunge.Polynomial ()
      %   obj = datamunge.Polynomial (constant)
      %   obj = datamunge.Polynomial (coeffs)
      %
      %   C++: datamunge::algebra::Polynomial::Polynomial()
      %   C++: datamunge::algebra::Polynomial::Polynomial(double constant)
      %   C++: datamunge::algebra::Polynomial::Polynomial(std::vector<double> coeffs)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 281, false, varargin{:});
    end
    function r = degree (obj)
      % degree  Degree of the polynomial; the zero polynomial has degree 0 (there is no universally-agreed convention for its "true" degree of -infinity, and 0 keeps every coefficient()/loop bound in this module well-defined).
      %
      %   r = obj.degree ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::algebra::Polynomial::degree() const
      r = __datamunge__(265, obj);
    end
    function r = is_zero (obj)
      % is_zero  C++ datamunge::algebra::Polynomial::is_zero.
      %
      %   r = obj.is_zero ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::algebra::Polynomial::is_zero() const
      r = __datamunge__(266, obj);
    end
    function r = coefficient (obj, i)
      % coefficient  Coefficient of x^i; 0.0 for any i outside [0, degree()] (including negative i).
      %
      %   r = obj.coefficient (i)
      %
      %   Inputs:
      %     i  integer
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::algebra::Polynomial::coefficient(int i) const
      r = __datamunge__(267, obj, i);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::algebra::Polynomial::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: const std::vector<double> & datamunge::algebra::Polynomial::coefficients() const
      r = __datamunge__(268, obj);
    end
    function r = evaluate (obj, x)
      % evaluate  Evaluates the polynomial at x via Horner's method.
      %
      %   r = obj.evaluate (x)
      %
      %   Inputs:
      %     x  double
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::algebra::Polynomial::evaluate(double x) const
      r = __datamunge__(269, obj, x);
    end
    function r = derivative (obj)
      % derivative  C++ datamunge::algebra::Polynomial::derivative.
      %
      %   r = obj.derivative ()
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::derivative() const
      r = __datamunge__(270, obj);
    end
    function r = antiderivative (obj)
      % antiderivative  The antiderivative with zero constant term (i.e. integral from 0 to x, as a polynomial in x).
      %
      %   r = obj.antiderivative ()
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::antiderivative() const
      r = __datamunge__(271, obj);
    end
    function r = add (obj, other)
      % add  C++ datamunge::algebra::Polynomial::add.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.Polynomial
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::add(const Polynomial & other) const
      r = __datamunge__(272, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  C++ datamunge::algebra::Polynomial::subtract.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.Polynomial
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::subtract(const Polynomial & other) const
      r = __datamunge__(273, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  C++ datamunge::algebra::Polynomial::multiply.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.Polynomial
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::multiply(const Polynomial & other) const
      r = __datamunge__(274, obj, other);
    end
    function r = scale (obj, factor)
      % scale  C++ datamunge::algebra::Polynomial::scale.
      %
      %   r = obj.scale (factor)
      %
      %   Inputs:
      %     factor  double
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::scale(double factor) const
      r = __datamunge__(275, obj, factor);
    end
    function r = negate (obj)
      % negate  C++ datamunge::algebra::Polynomial::negate.
      %
      %   r = obj.negate ()
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: Polynomial datamunge::algebra::Polynomial::negate() const
      r = __datamunge__(276, obj);
    end
    function r = to_string (obj)
      % to_string  C++ datamunge::algebra::Polynomial::to_string.
      %
      %   r = obj.to_string ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::algebra::Polynomial::to_string() const
      r = __datamunge__(277, obj);
    end
    function r = to_tex (obj, varargin)
      % to_tex  Render descending powers using a raw TeX symbol and 1-17 significant digits; return raw TeX.
      %
      %   r = obj.to_tex ()
      %   r = obj.to_tex (variable)
      %   r = obj.to_tex (variable, precision)
      %
      %   Inputs:
      %     variable  char row, default 'x'
      %     precision  integer, default 7
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::octave_bindings::polynomial_to_tex(const datamunge::algebra::Polynomial & self, const std::string & variable = "x", int precision = 7)
      r = __datamunge__(278, obj, varargin{:});
    end
    function r = eq (a, b)
      % eq  C++ operator == on datamunge.Polynomial.
      if isa (a, 'datamunge.Polynomial')
        r = __datamunge__(279, a, b);
      else
        error ('datamunge:binding', 'operator == is not defined for %s and %s', class (a), class (b));
      end
    end
    function r = ne (a, b)
      % ne  C++ operator ~= on datamunge.Polynomial.
      if isa (a, 'datamunge.Polynomial')
        r = __datamunge__(280, a, b);
      else
        error ('datamunge:binding', 'operator ~= is not defined for %s and %s', class (a), class (b));
      end
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::algebra::Polynomial. Do not edit.
