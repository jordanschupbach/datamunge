classdef Expr < datamunge.NativeObject
  % Expr  A symbolic expression tree over +, -, *, /, ^ (power) and sin/cos/exp/log, supporting symbolic differentiation, basic algebraic simplification, and numeric evaluation. Value type wrapping a shared, immutable node tree -- copies are cheap and structural sharing across copies is always safe since nodes are never mutated after construction.
  %
  %   obj = datamunge.Expr ()
  %   obj = datamunge.Expr (value_)
  %
  %   Methods:
  %     add  C++ datamunge::algebra::Expr::add.
  %     subtract  C++ datamunge::algebra::Expr::subtract.
  %     multiply  C++ datamunge::algebra::Expr::multiply.
  %     divide  C++ datamunge::algebra::Expr::divide.
  %     power_  C++ datamunge::algebra::Expr::power.
  %     negate  C++ datamunge::algebra::Expr::negate.
  %     sin  C++ datamunge::algebra::Expr::sin.
  %     cos  C++ datamunge::algebra::Expr::cos.
  %     exp  C++ datamunge::algebra::Expr::exp.
  %     log  C++ datamunge::algebra::Expr::log.
  %     differentiate  The symbolic derivative with respect to `var` (any other variable name is treated as a constant).
  %     integrate  The symbolic antiderivative with respect to `var` (any other variable name is treated as a constant, matching differentiate()). Supports: the sum/difference and constant-multiple rules; the power rule x^c (any real c != -1) and its log special case (c == -1); sin/cos/exp/log of any expression that is affine (a*var + b) in var, via the standard substitution result for each; a constant divided by an affine expression (reduces to the log rule); and, for a product of two var-dependent factors, generic tabular integration by parts -- reusing differentiate() to peel one factor down to the constant 0 across at most 15 repeated derivatives (terminates for any genuine polynomial factor), paired with repeated antiderivatives of the other factor. Not a full computer-algebra integrator: general rational functions (a non-constant denominator that isn't affine), a variable exponent, and products/quotients outside the forms above (e.g. sin(x)*cos(x), which never reduces to zero under repeated differentiation) are not attempted.
  %     simplify  Basic algebraic simplification: constant folding, plus identities like x+0, x*1, x*0, x/1, x^0, x^1. Not a full computer-algebra simplifier (no factoring, collecting like terms, or trig identities).
  %     evaluate  Evaluates the expression, binding each entry of `names` to the corresponding entry of `values` (parallel arrays, must be the same length).
  %     to_string  C++ datamunge::algebra::Expr::to_string.
  %     to_tex  Render the expression using native grouping and precision; return raw TeX without math delimiters.
  %
  %   Static methods:
  %     constant  C++ datamunge::algebra::Expr::constant.
  %     variable  C++ datamunge::algebra::Expr::variable.
  %
  %   C++: class datamunge::algebra::Expr
  methods
    function obj = Expr (varargin)
      % Expr  Construct a C++ datamunge::algebra::Expr.
      %
      %   obj = datamunge.Expr ()
      %   obj = datamunge.Expr (value_)
      %
      %   Numeric arguments convert to double, matching constant(); integers larger than double's
      %   exact integer range may round. Infinity and NaN are accepted. The new immutable node
      %   owns its value. Enables literals on either side of expression operators and assignment
      %   such as Expr f = 1. Does not add a conversion from Expr back to a number; use evaluate().
      %
      %   Example:
      %     dm::algebra::Expr value = 2.5;
      %     std::print("{}\n", value.evaluate({}, {})); // 2.5
      %
      %   Construct a symbolic constant from a C++ numeric value, including implicit conversion.
      %
      %   Errors:
      %     datamunge:allocation  If the constant node cannot be allocated. (C++ std::bad_alloc)
      %
      %   C++: datamunge::algebra::Expr::Expr()
      %   C++: datamunge::algebra::Expr::Expr(double value)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 264, false, varargin{:});
    end
    function r = add (obj, other)
      % add  C++ datamunge::algebra::Expr::add.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.Expr
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::add(const Expr & other) const
      r = __datamunge__(248, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  C++ datamunge::algebra::Expr::subtract.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.Expr
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::subtract(const Expr & other) const
      r = __datamunge__(249, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  C++ datamunge::algebra::Expr::multiply.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.Expr
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::multiply(const Expr & other) const
      r = __datamunge__(250, obj, other);
    end
    function r = divide (obj, other)
      % divide  C++ datamunge::algebra::Expr::divide.
      %
      %   r = obj.divide (other)
      %
      %   Inputs:
      %     other  datamunge.Expr
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::divide(const Expr & other) const
      r = __datamunge__(251, obj, other);
    end
    function r = power_ (obj, exponent)
      % power_  C++ datamunge::algebra::Expr::power.
      %
      %   r = obj.power_ (exponent)
      %
      %   Inputs:
      %     exponent  datamunge.Expr
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::power(const Expr & exponent) const
      r = __datamunge__(252, obj, exponent);
    end
    function r = negate (obj)
      % negate  C++ datamunge::algebra::Expr::negate.
      %
      %   r = obj.negate ()
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::negate() const
      r = __datamunge__(253, obj);
    end
    function r = sin (obj)
      % sin  C++ datamunge::algebra::Expr::sin.
      %
      %   r = obj.sin ()
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::sin() const
      r = __datamunge__(254, obj);
    end
    function r = cos (obj)
      % cos  C++ datamunge::algebra::Expr::cos.
      %
      %   r = obj.cos ()
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::cos() const
      r = __datamunge__(255, obj);
    end
    function r = exp (obj)
      % exp  C++ datamunge::algebra::Expr::exp.
      %
      %   r = obj.exp ()
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::exp() const
      r = __datamunge__(256, obj);
    end
    function r = log (obj)
      % log  C++ datamunge::algebra::Expr::log.
      %
      %   r = obj.log ()
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::log() const
      r = __datamunge__(257, obj);
    end
    function r = differentiate (obj, var)
      % differentiate  The symbolic derivative with respect to `var` (any other variable name is treated as a constant).
      %
      %   r = obj.differentiate (var)
      %
      %   Inputs:
      %     var  char row
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   Errors:
      %     datamunge:invalid_argument  if the expression contains `base ^ exponent` where `exponent` is not a constant (only the constant-exponent power rule is supported). (C++ std::invalid_argument)
      %
      %   C++: Expr datamunge::algebra::Expr::differentiate(const std::string & var) const
      r = __datamunge__(258, obj, var);
    end
    function r = integrate (obj, var)
      % integrate  The symbolic antiderivative with respect to `var` (any other variable name is treated as a constant, matching differentiate()). Supports: the sum/difference and constant-multiple rules; the power rule x^c (any real c != -1) and its log special case (c == -1); sin/cos/exp/log of any expression that is affine (a*var + b) in var, via the standard substitution result for each; a constant divided by an affine expression (reduces to the log rule); and, for a product of two var-dependent factors, generic tabular integration by parts -- reusing differentiate() to peel one factor down to the constant 0 across at most 15 repeated derivatives (terminates for any genuine polynomial factor), paired with repeated antiderivatives of the other factor. Not a full computer-algebra integrator: general rational functions (a non-constant denominator that isn't affine), a variable exponent, and products/quotients outside the forms above (e.g. sin(x)*cos(x), which never reduces to zero under repeated differentiation) are not attempted.
      %
      %   r = obj.integrate (var)
      %
      %   Inputs:
      %     var  char row
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   Errors:
      %     datamunge:invalid_argument  when no rule above applies. (C++ std::invalid_argument)
      %
      %   C++: Expr datamunge::algebra::Expr::integrate(const std::string & var) const
      r = __datamunge__(259, obj, var);
    end
    function r = simplify (obj)
      % simplify  Basic algebraic simplification: constant folding, plus identities like x+0, x*1, x*0, x/1, x^0, x^1. Not a full computer-algebra simplifier (no factoring, collecting like terms, or trig identities).
      %
      %   r = obj.simplify ()
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::simplify() const
      r = __datamunge__(260, obj);
    end
    function r = evaluate (obj, names, values)
      % evaluate  Evaluates the expression, binding each entry of `names` to the corresponding entry of `values` (parallel arrays, must be the same length).
      %
      %   r = obj.evaluate (names, values)
      %
      %   Inputs:
      %     names  cell array of char
      %     values  double vector
      %
      %   Output:
      %     r  double
      %
      %   Errors:
      %     datamunge:invalid_argument  if names/values sizes disagree or a variable in the expression has no binding. (C++ std::invalid_argument)
      %
      %   C++: double datamunge::algebra::Expr::evaluate(const std::vector<std::string> & names, const std::vector<double> & values) const
      r = __datamunge__(261, obj, names, values);
    end
    function r = to_string (obj)
      % to_string  C++ datamunge::algebra::Expr::to_string.
      %
      %   r = obj.to_string ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::algebra::Expr::to_string() const
      r = __datamunge__(262, obj);
    end
    function r = to_tex (obj, varargin)
      % to_tex  Render the expression using native grouping and precision; return raw TeX without math delimiters.
      %
      %   r = obj.to_tex ()
      %   r = obj.to_tex (precision)
      %
      %   Inputs:
      %     precision  integer, default 7
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::octave_bindings::expr_to_tex(const datamunge::algebra::Expr & self, int precision = 7)
      r = __datamunge__(263, obj, varargin{:});
    end
  end
  methods (Static)
    function r = constant (value_)
      % constant  C++ datamunge::algebra::Expr::constant.
      %
      %   r = datamunge.Expr.constant (value_)
      %
      %   Inputs:
      %     value_  double
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::constant(double value)
      r = __datamunge__(246, value_);
    end
    function r = variable (name)
      % variable  C++ datamunge::algebra::Expr::variable.
      %
      %   r = datamunge.Expr.variable (name)
      %
      %   Inputs:
      %     name  char row
      %
      %   Output:
      %     r  datamunge.Expr
      %
      %   C++: Expr datamunge::algebra::Expr::variable(const std::string & name)
      r = __datamunge__(247, name);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::algebra::Expr. Do not edit.
