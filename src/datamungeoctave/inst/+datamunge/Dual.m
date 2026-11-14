classdef Dual < datamunge.NativeObject
  % Dual  SWIG-friendly facade for datamunge::autodiff::Dual -- a first-order forward-mode dual number. Build an expression out of named operations (starting from a seed with derivative=1) to read off an exact derivative alongside the value.
  %
  %   obj = datamunge.Dual (value_)
  %   obj = datamunge.Dual (value_, derivative)
  %
  %   Methods:
  %     value  The real part: the function value.
  %     derivative  The dual part: the exact derivative with respect to the seed.
  %     add  Sum: this + `other`, with derivative a' + b'.
  %     subtract  Difference: this - `other`, with derivative a' - b'.
  %     multiply  Product: this * `other`, with derivative a'b + ab' (product rule).
  %     divide  Quotient: this / `other`, with derivative (a'b - ab')/b² (IEEE inf/NaN when other is 0).
  %     negate  Negation -x, with its exact chain-rule derivative.
  %     add_scalar  Adds a constant `scalar`, derivative unchanged.
  %     subtract_scalar  Subtracts a constant `scalar`, derivative unchanged.
  %     multiply_scalar  Multiplies by a constant `scalar`, derivative scaled by it.
  %     divide_scalar  Divides by a constant `scalar`, derivative divided by it.
  %     pow  Power x^p for a constant exponent, with derivative p·x^(p-1).
  %     exp  Exponential e^x, with its exact chain-rule derivative.
  %     log  Natural logarithm ln x (NaN for x < 0, -inf at 0), with its exact chain-rule derivative.
  %     sqrt  Square root √x (NaN for x < 0; infinite derivative at 0), with its exact chain-rule derivative.
  %     sin  Sine (radians), with its exact chain-rule derivative.
  %     cos  Cosine (radians), with its exact chain-rule derivative.
  %     tan  Tangent (radians), with its exact chain-rule derivative.
  %     tanh  Hyperbolic tangent, with its exact chain-rule derivative.
  %     abs  Absolute value |x| (the derivative at 0 uses the x ≥ 0 branch), with its exact chain-rule derivative.
  %
  %   C++: class datamunge::Dual
  methods
    function obj = Dual (varargin)
      % Dual  Construct a C++ datamunge::Dual.
      %
      %   obj = datamunge.Dual (value_)
      %   obj = datamunge.Dual (value_, derivative)
      %
      %   Seed the variable you differentiate with derivative 1 and constants with 0; every
      %   operation then carries the exact derivative with respect to that seed.
      %
      %   Example:
      %     const dm::Dual x(2.0, 1.0);  // seed d/dx
      %     std::print("{}\n", x.multiply(x).derivative());  // 4
      %
      %   A dual number value + derivative·ε (ε² = 0).
      %
      %   Inputs:
      %     value_  double: Real part, the function value.
      %     derivative  double, default 0.0: Dual part, the derivative with respect to the seed (default 0).
      %
      %   C++: datamunge::Dual::Dual(double value, double derivative = 0.0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2880, false, varargin{:});
    end
    function r = value (obj)
      % value  The real part: the function value.
      %
      %   r = obj.value ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Dual::value() const
      r = __datamunge__(2860, obj);
    end
    function r = derivative (obj)
      % derivative  The dual part: the exact derivative with respect to the seed.
      %
      %   r = obj.derivative ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Dual::derivative() const
      r = __datamunge__(2861, obj);
    end
    function r = add (obj, other)
      % add  Sum: this + `other`, with derivative a' + b'.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.Dual: Right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual; neither operand changes.
      %
      %   C++: Dual datamunge::Dual::add(const Dual & other) const
      r = __datamunge__(2862, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  Difference: this - `other`, with derivative a' - b'.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.Dual: Right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual; neither operand changes.
      %
      %   C++: Dual datamunge::Dual::subtract(const Dual & other) const
      r = __datamunge__(2863, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  Product: this * `other`, with derivative a'b + ab' (product rule).
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.Dual: Right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual; neither operand changes.
      %
      %   C++: Dual datamunge::Dual::multiply(const Dual & other) const
      r = __datamunge__(2864, obj, other);
    end
    function r = divide (obj, other)
      % divide  Quotient: this / `other`, with derivative (a'b - ab')/b² (IEEE inf/NaN when other is 0).
      %
      %   r = obj.divide (other)
      %
      %   Inputs:
      %     other  datamunge.Dual: Right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual; neither operand changes.
      %
      %   C++: Dual datamunge::Dual::divide(const Dual & other) const
      r = __datamunge__(2865, obj, other);
    end
    function r = negate (obj)
      % negate  Negation -x, with its exact chain-rule derivative.
      %
      %   r = obj.negate ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::negate() const
      r = __datamunge__(2866, obj);
    end
    function r = add_scalar (obj, scalar)
      % add_scalar  Adds a constant `scalar`, derivative unchanged.
      %
      %   r = obj.add_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::add_scalar(double scalar) const
      r = __datamunge__(2867, obj, scalar);
    end
    function r = subtract_scalar (obj, scalar)
      % subtract_scalar  Subtracts a constant `scalar`, derivative unchanged.
      %
      %   r = obj.subtract_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::subtract_scalar(double scalar) const
      r = __datamunge__(2868, obj, scalar);
    end
    function r = multiply_scalar (obj, scalar)
      % multiply_scalar  Multiplies by a constant `scalar`, derivative scaled by it.
      %
      %   r = obj.multiply_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::multiply_scalar(double scalar) const
      r = __datamunge__(2869, obj, scalar);
    end
    function r = divide_scalar (obj, scalar)
      % divide_scalar  Divides by a constant `scalar`, derivative divided by it.
      %
      %   r = obj.divide_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::divide_scalar(double scalar) const
      r = __datamunge__(2870, obj, scalar);
    end
    function r = pow (obj, exponent)
      % pow  Power x^p for a constant exponent, with derivative p·x^(p-1).
      %
      %   r = obj.pow (exponent)
      %
      %   Inputs:
      %     exponent  double: Constant exponent p; non-integer p needs x ≥ 0 (NaN otherwise).
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::pow(double exponent) const
      r = __datamunge__(2871, obj, exponent);
    end
    function r = exp (obj)
      % exp  Exponential e^x, with its exact chain-rule derivative.
      %
      %   r = obj.exp ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::exp() const
      r = __datamunge__(2872, obj);
    end
    function r = log (obj)
      % log  Natural logarithm ln x (NaN for x < 0, -inf at 0), with its exact chain-rule derivative.
      %
      %   r = obj.log ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::log() const
      r = __datamunge__(2873, obj);
    end
    function r = sqrt (obj)
      % sqrt  Square root √x (NaN for x < 0; infinite derivative at 0), with its exact chain-rule derivative.
      %
      %   r = obj.sqrt ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::sqrt() const
      r = __datamunge__(2874, obj);
    end
    function r = sin (obj)
      % sin  Sine (radians), with its exact chain-rule derivative.
      %
      %   r = obj.sin ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::sin() const
      r = __datamunge__(2875, obj);
    end
    function r = cos (obj)
      % cos  Cosine (radians), with its exact chain-rule derivative.
      %
      %   r = obj.cos ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::cos() const
      r = __datamunge__(2876, obj);
    end
    function r = tan (obj)
      % tan  Tangent (radians), with its exact chain-rule derivative.
      %
      %   r = obj.tan ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::tan() const
      r = __datamunge__(2877, obj);
    end
    function r = tanh (obj)
      % tanh  Hyperbolic tangent, with its exact chain-rule derivative.
      %
      %   r = obj.tanh ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::tanh() const
      r = __datamunge__(2878, obj);
    end
    function r = abs (obj)
      % abs  Absolute value |x| (the derivative at 0 uses the x ≥ 0 branch), with its exact chain-rule derivative.
      %
      %   r = obj.abs ()
      %
      %   Output:
      %     r  datamunge.Dual: A new Dual.
      %
      %   C++: Dual datamunge::Dual::abs() const
      r = __datamunge__(2879, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Dual. Do not edit.
