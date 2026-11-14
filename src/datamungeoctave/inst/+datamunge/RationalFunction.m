classdef RationalFunction < datamunge.NativeObject
  % RationalFunction  A ratio of two polynomials (numerator / denominator), kept automatically reduced by their GCD -- e.g. (x^2-1)/(x-1) simplifies to x+1 -- with the denominator normalized to monic.
  %
  %   obj = datamunge.RationalFunction ()
  %   obj = datamunge.RationalFunction (numerator, denominator)
  %
  %   Methods:
  %     numerator  C++ datamunge::algebra::RationalFunction::numerator.
  %     denominator  C++ datamunge::algebra::RationalFunction::denominator.
  %     evaluate  C++ datamunge::algebra::RationalFunction::evaluate.
  %     add  C++ datamunge::algebra::RationalFunction::add.
  %     subtract  C++ datamunge::algebra::RationalFunction::subtract.
  %     multiply  C++ datamunge::algebra::RationalFunction::multiply.
  %     divide  C++ datamunge::algebra::RationalFunction::divide.
  %     to_string  C++ datamunge::algebra::RationalFunction::to_string.
  %     to_tex  Render the native reduced rational function as raw TeX, omitting a unit denominator.
  %
  %   C++: class datamunge::algebra::RationalFunction
  methods
    function obj = RationalFunction (varargin)
      % RationalFunction  Construct a C++ datamunge::algebra::RationalFunction.
      %
      %   obj = datamunge.RationalFunction ()
      %   obj = datamunge.RationalFunction (numerator, denominator)
      %
      %   C++: datamunge::algebra::RationalFunction::RationalFunction()
      %   C++: datamunge::algebra::RationalFunction::RationalFunction(Polynomial numerator, Polynomial denominator)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 352, false, varargin{:});
    end
    function r = numerator (obj)
      % numerator  C++ datamunge::algebra::RationalFunction::numerator.
      %
      %   r = obj.numerator ()
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: const Polynomial & datamunge::algebra::RationalFunction::numerator() const
      r = __datamunge__(343, obj);
    end
    function r = denominator (obj)
      % denominator  C++ datamunge::algebra::RationalFunction::denominator.
      %
      %   r = obj.denominator ()
      %
      %   Output:
      %     r  datamunge.Polynomial
      %
      %   C++: const Polynomial & datamunge::algebra::RationalFunction::denominator() const
      r = __datamunge__(344, obj);
    end
    function r = evaluate (obj, x)
      % evaluate  C++ datamunge::algebra::RationalFunction::evaluate.
      %
      %   r = obj.evaluate (x)
      %
      %   Inputs:
      %     x  double
      %
      %   Output:
      %     r  double
      %
      %   Errors:
      %     datamunge:domain_error  if x is a pole (denominator evaluates to zero at x). (C++ std::domain_error)
      %
      %   C++: double datamunge::algebra::RationalFunction::evaluate(double x) const
      r = __datamunge__(345, obj, x);
    end
    function r = add (obj, other)
      % add  C++ datamunge::algebra::RationalFunction::add.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.RationalFunction
      %
      %   Output:
      %     r  datamunge.RationalFunction
      %
      %   C++: RationalFunction datamunge::algebra::RationalFunction::add(const RationalFunction & other) const
      r = __datamunge__(346, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  C++ datamunge::algebra::RationalFunction::subtract.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.RationalFunction
      %
      %   Output:
      %     r  datamunge.RationalFunction
      %
      %   C++: RationalFunction datamunge::algebra::RationalFunction::subtract(const RationalFunction & other) const
      r = __datamunge__(347, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  C++ datamunge::algebra::RationalFunction::multiply.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.RationalFunction
      %
      %   Output:
      %     r  datamunge.RationalFunction
      %
      %   C++: RationalFunction datamunge::algebra::RationalFunction::multiply(const RationalFunction & other) const
      r = __datamunge__(348, obj, other);
    end
    function r = divide (obj, other)
      % divide  C++ datamunge::algebra::RationalFunction::divide.
      %
      %   r = obj.divide (other)
      %
      %   Inputs:
      %     other  datamunge.RationalFunction
      %
      %   Output:
      %     r  datamunge.RationalFunction
      %
      %   Errors:
      %     datamunge:invalid_argument  if other is the zero rational function. (C++ std::invalid_argument)
      %
      %   C++: RationalFunction datamunge::algebra::RationalFunction::divide(const RationalFunction & other) const
      r = __datamunge__(349, obj, other);
    end
    function r = to_string (obj)
      % to_string  C++ datamunge::algebra::RationalFunction::to_string.
      %
      %   r = obj.to_string ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::algebra::RationalFunction::to_string() const
      r = __datamunge__(350, obj);
    end
    function r = to_tex (obj, varargin)
      % to_tex  Render the native reduced rational function as raw TeX, omitting a unit denominator.
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
      %   C++: std::string datamunge::octave_bindings::rational_function_to_tex(const datamunge::algebra::RationalFunction & self, const std::string & variable = "x", int precision = 7)
      r = __datamunge__(351, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::algebra::RationalFunction. Do not edit.
