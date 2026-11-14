classdef MultivariatePolynomial < datamunge.NativeObject
  % MultivariatePolynomial  Sparse multivariate polynomial over the reals in a fixed number of variables (x0, x1, ...), stored internally as a map from exponent vector to coefficient. Exposed to bindings via parallel arrays (exponents()/coefficients()) rather than the internal map, matching this module's std::map-avoidance convention for SWIG-facing types.
  %
  %   obj = datamunge.MultivariatePolynomial ()
  %   obj = datamunge.MultivariatePolynomial (num_variables)
  %   obj = datamunge.MultivariatePolynomial (num_variables, exponents, coefficients)
  %
  %   Methods:
  %     num_variables  C++ datamunge::algebra::MultivariatePolynomial::num_variables.
  %     num_terms  C++ datamunge::algebra::MultivariatePolynomial::num_terms.
  %     is_zero  C++ datamunge::algebra::MultivariatePolynomial::is_zero.
  %     exponents  C++ datamunge::algebra::MultivariatePolynomial::exponents.
  %     coefficients  C++ datamunge::algebra::MultivariatePolynomial::coefficients.
  %     evaluate  C++ datamunge::algebra::MultivariatePolynomial::evaluate.
  %     add  C++ datamunge::algebra::MultivariatePolynomial::add.
  %     subtract  C++ datamunge::algebra::MultivariatePolynomial::subtract.
  %     multiply  C++ datamunge::algebra::MultivariatePolynomial::multiply.
  %     scale  C++ datamunge::algebra::MultivariatePolynomial::scale.
  %     leading_exponent  C++ datamunge::algebra::MultivariatePolynomial::leading_exponent.
  %     leading_coefficient  C++ datamunge::algebra::MultivariatePolynomial::leading_coefficient.
  %     to_string  C++ datamunge::algebra::MultivariatePolynomial::to_string.
  %
  %   C++: class datamunge::algebra::MultivariatePolynomial
  methods
    function obj = MultivariatePolynomial (varargin)
      % MultivariatePolynomial  Construct a C++ datamunge::algebra::MultivariatePolynomial.
      %
      %   obj = datamunge.MultivariatePolynomial ()
      %   obj = datamunge.MultivariatePolynomial (num_variables)
      %   obj = datamunge.MultivariatePolynomial (num_variables, exponents, coefficients)
      %
      %   C++: datamunge::algebra::MultivariatePolynomial::MultivariatePolynomial()
      %   C++: datamunge::algebra::MultivariatePolynomial::MultivariatePolynomial(int num_variables)
      %   C++: datamunge::algebra::MultivariatePolynomial::MultivariatePolynomial(int num_variables, const std::vector<std::vector<int>> & exponents, const std::vector<double> & coefficients)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 335, false, varargin{:});
    end
    function r = num_variables (obj)
      % num_variables  C++ datamunge::algebra::MultivariatePolynomial::num_variables.
      %
      %   r = obj.num_variables ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::algebra::MultivariatePolynomial::num_variables() const
      r = __datamunge__(322, obj);
    end
    function r = num_terms (obj)
      % num_terms  C++ datamunge::algebra::MultivariatePolynomial::num_terms.
      %
      %   r = obj.num_terms ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::algebra::MultivariatePolynomial::num_terms() const
      r = __datamunge__(323, obj);
    end
    function r = is_zero (obj)
      % is_zero  C++ datamunge::algebra::MultivariatePolynomial::is_zero.
      %
      %   r = obj.is_zero ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::algebra::MultivariatePolynomial::is_zero() const
      r = __datamunge__(324, obj);
    end
    function r = exponents (obj)
      % exponents  C++ datamunge::algebra::MultivariatePolynomial::exponents.
      %
      %   r = obj.exponents ()
      %
      %   Output:
      %     r  cell array of integer vector
      %
      %   C++: std::vector<std::vector<int>> datamunge::algebra::MultivariatePolynomial::exponents() const
      r = __datamunge__(325, obj);
    end
    function r = coefficients (obj)
      % coefficients  C++ datamunge::algebra::MultivariatePolynomial::coefficients.
      %
      %   r = obj.coefficients ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::algebra::MultivariatePolynomial::coefficients() const
      r = __datamunge__(326, obj);
    end
    function r = evaluate (obj, point)
      % evaluate  C++ datamunge::algebra::MultivariatePolynomial::evaluate.
      %
      %   r = obj.evaluate (point)
      %
      %   Inputs:
      %     point  double vector
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::algebra::MultivariatePolynomial::evaluate(const std::vector<double> & point) const
      r = __datamunge__(327, obj, point);
    end
    function r = add (obj, other)
      % add  C++ datamunge::algebra::MultivariatePolynomial::add.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.MultivariatePolynomial
      %
      %   Output:
      %     r  datamunge.MultivariatePolynomial
      %
      %   C++: MultivariatePolynomial datamunge::algebra::MultivariatePolynomial::add(const MultivariatePolynomial & other) const
      r = __datamunge__(328, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  C++ datamunge::algebra::MultivariatePolynomial::subtract.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.MultivariatePolynomial
      %
      %   Output:
      %     r  datamunge.MultivariatePolynomial
      %
      %   C++: MultivariatePolynomial datamunge::algebra::MultivariatePolynomial::subtract(const MultivariatePolynomial & other) const
      r = __datamunge__(329, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  C++ datamunge::algebra::MultivariatePolynomial::multiply.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.MultivariatePolynomial
      %
      %   Output:
      %     r  datamunge.MultivariatePolynomial
      %
      %   C++: MultivariatePolynomial datamunge::algebra::MultivariatePolynomial::multiply(const MultivariatePolynomial & other) const
      r = __datamunge__(330, obj, other);
    end
    function r = scale (obj, factor)
      % scale  C++ datamunge::algebra::MultivariatePolynomial::scale.
      %
      %   r = obj.scale (factor)
      %
      %   Inputs:
      %     factor  double
      %
      %   Output:
      %     r  datamunge.MultivariatePolynomial
      %
      %   C++: MultivariatePolynomial datamunge::algebra::MultivariatePolynomial::scale(double factor) const
      r = __datamunge__(331, obj, factor);
    end
    function r = leading_exponent (obj, order)
      % leading_exponent  C++ datamunge::algebra::MultivariatePolynomial::leading_exponent.
      %
      %   r = obj.leading_exponent (order)
      %
      %   Inputs:
      %     order  datamunge.MonomialOrder value
      %
      %   Output:
      %     r  integer vector
      %
      %   Errors:
      %     datamunge:invalid_argument  if this is the zero polynomial. (C++ std::invalid_argument)
      %
      %   C++: std::vector<int> datamunge::algebra::MultivariatePolynomial::leading_exponent(MonomialOrder order) const
      r = __datamunge__(332, obj, order);
    end
    function r = leading_coefficient (obj, order)
      % leading_coefficient  C++ datamunge::algebra::MultivariatePolynomial::leading_coefficient.
      %
      %   r = obj.leading_coefficient (order)
      %
      %   Inputs:
      %     order  datamunge.MonomialOrder value
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::algebra::MultivariatePolynomial::leading_coefficient(MonomialOrder order) const
      r = __datamunge__(333, obj, order);
    end
    function r = to_string (obj)
      % to_string  C++ datamunge::algebra::MultivariatePolynomial::to_string.
      %
      %   r = obj.to_string ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::algebra::MultivariatePolynomial::to_string() const
      r = __datamunge__(334, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::algebra::MultivariatePolynomial. Do not edit.
