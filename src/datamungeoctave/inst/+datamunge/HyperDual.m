classdef HyperDual < datamunge.NativeObject
  % HyperDual  SWIG-friendly facade for datamunge::autodiff::HyperDual -- a second-order forward-mode dual number carrying two independent derivative directions plus their exact mixed second partial, for computing Hessian entries.
  %
  %   obj = datamunge.HyperDual (value_)
  %   obj = datamunge.HyperDual (value_, eps1)
  %   obj = datamunge.HyperDual (value_, eps1, eps2)
  %   obj = datamunge.HyperDual (value_, eps1, eps2, eps1eps2)
  %
  %   Methods:
  %     value  The real part: the function value.
  %     eps1  Derivative along the first seed direction (ε₁ part).
  %     eps2  Derivative along the second seed direction (ε₂ part).
  %     eps1eps2  The mixed second derivative along both seed directions (ε₁ε₂ part): a Hessian entry.
  %     add  Sum: this + `other`, with first and mixed second derivatives.
  %     subtract  Difference: this - `other`, with first and mixed second derivatives.
  %     multiply  Product: this * `other`, with first and mixed second derivatives.
  %     divide  Quotient: this / `other`, with first and mixed second derivatives (IEEE inf/NaN when other is 0).
  %     negate  Negation -x, with its exact chain-rule derivative.
  %     add_scalar  Adds a constant `scalar`, derivative parts unchanged.
  %     subtract_scalar  Subtracts a constant `scalar`, derivative parts unchanged.
  %     multiply_scalar  Multiplies by a constant `scalar`, derivative parts scaled accordingly.
  %     divide_scalar  Divides by a constant `scalar`, derivative parts scaled accordingly.
  %     pow  Power x^p for a constant exponent, with derivative p·x^(p-1).
  %     exp  Exponential e^x, with its exact chain-rule derivative.
  %     log  Natural logarithm ln x (NaN for x < 0, -inf at 0), with its exact chain-rule derivative.
  %     sqrt  Square root √x (NaN for x < 0; infinite derivative at 0), with its exact chain-rule derivative.
  %     sin  Sine (radians), with its exact chain-rule derivative.
  %     cos  Cosine (radians), with its exact chain-rule derivative.
  %     tan  Tangent (radians), with its exact chain-rule derivative.
  %     tanh  Hyperbolic tangent, with its exact chain-rule derivative.
  %
  %   C++: class datamunge::HyperDual
  methods
    function obj = HyperDual (varargin)
      % HyperDual  Construct a C++ datamunge::HyperDual.
      %
      %   obj = datamunge.HyperDual (value_)
      %   obj = datamunge.HyperDual (value_, eps1)
      %   obj = datamunge.HyperDual (value_, eps1, eps2)
      %   obj = datamunge.HyperDual (value_, eps1, eps2, eps1eps2)
      %
      %   Seed input i with eps1 = 1 and input j with eps2 = 1 (both on the same input for a
      %   diagonal entry); the result's eps1eps2() is then the exact ∂²f/∂xᵢ∂xⱼ.
      %
      %   Example:
      %     const dm::HyperDual x(3.0, 1.0, 1.0);
      %     std::print("{}\n", x.multiply(x).eps1eps2());  // d²(x²)/dx² = 2
      %
      %   A hyper-dual number value + eps1·ε₁ + eps2·ε₂ + eps1eps2·ε₁ε₂ (ε₁² = ε₂² = 0).
      %
      %   Inputs:
      %     value_  double: Real part, the function value.
      %     eps1  double, default 0.0: First derivative along the first seed direction (default 0).
      %     eps2  double, default 0.0: First derivative along the second seed direction (default 0).
      %     eps1eps2  double, default 0.0: Mixed second derivative part (default 0; leave 0 for inputs).
      %
      %   C++: datamunge::HyperDual::HyperDual(double value, double eps1 = 0.0, double eps2 = 0.0, double eps1eps2 = 0.0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2902, false, varargin{:});
    end
    function r = value (obj)
      % value  The real part: the function value.
      %
      %   r = obj.value ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::HyperDual::value() const
      r = __datamunge__(2881, obj);
    end
    function r = eps1 (obj)
      % eps1  Derivative along the first seed direction (ε₁ part).
      %
      %   r = obj.eps1 ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::HyperDual::eps1() const
      r = __datamunge__(2882, obj);
    end
    function r = eps2 (obj)
      % eps2  Derivative along the second seed direction (ε₂ part).
      %
      %   r = obj.eps2 ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::HyperDual::eps2() const
      r = __datamunge__(2883, obj);
    end
    function r = eps1eps2 (obj)
      % eps1eps2  The mixed second derivative along both seed directions (ε₁ε₂ part): a Hessian entry.
      %
      %   r = obj.eps1eps2 ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::HyperDual::eps1eps2() const
      r = __datamunge__(2884, obj);
    end
    function r = add (obj, other)
      % add  Sum: this + `other`, with first and mixed second derivatives.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.HyperDual: Right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual; neither operand changes.
      %
      %   C++: HyperDual datamunge::HyperDual::add(const HyperDual & other) const
      r = __datamunge__(2885, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  Difference: this - `other`, with first and mixed second derivatives.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.HyperDual: Right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual; neither operand changes.
      %
      %   C++: HyperDual datamunge::HyperDual::subtract(const HyperDual & other) const
      r = __datamunge__(2886, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  Product: this * `other`, with first and mixed second derivatives.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.HyperDual: Right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual; neither operand changes.
      %
      %   C++: HyperDual datamunge::HyperDual::multiply(const HyperDual & other) const
      r = __datamunge__(2887, obj, other);
    end
    function r = divide (obj, other)
      % divide  Quotient: this / `other`, with first and mixed second derivatives (IEEE inf/NaN when other is 0).
      %
      %   r = obj.divide (other)
      %
      %   Inputs:
      %     other  datamunge.HyperDual: Right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual; neither operand changes.
      %
      %   C++: HyperDual datamunge::HyperDual::divide(const HyperDual & other) const
      r = __datamunge__(2888, obj, other);
    end
    function r = negate (obj)
      % negate  Negation -x, with its exact chain-rule derivative.
      %
      %   r = obj.negate ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::negate() const
      r = __datamunge__(2889, obj);
    end
    function r = add_scalar (obj, scalar)
      % add_scalar  Adds a constant `scalar`, derivative parts unchanged.
      %
      %   r = obj.add_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::add_scalar(double scalar) const
      r = __datamunge__(2890, obj, scalar);
    end
    function r = subtract_scalar (obj, scalar)
      % subtract_scalar  Subtracts a constant `scalar`, derivative parts unchanged.
      %
      %   r = obj.subtract_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::subtract_scalar(double scalar) const
      r = __datamunge__(2891, obj, scalar);
    end
    function r = multiply_scalar (obj, scalar)
      % multiply_scalar  Multiplies by a constant `scalar`, derivative parts scaled accordingly.
      %
      %   r = obj.multiply_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::multiply_scalar(double scalar) const
      r = __datamunge__(2892, obj, scalar);
    end
    function r = divide_scalar (obj, scalar)
      % divide_scalar  Divides by a constant `scalar`, derivative parts scaled accordingly.
      %
      %   r = obj.divide_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::divide_scalar(double scalar) const
      r = __datamunge__(2893, obj, scalar);
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
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::pow(double exponent) const
      r = __datamunge__(2894, obj, exponent);
    end
    function r = exp (obj)
      % exp  Exponential e^x, with its exact chain-rule derivative.
      %
      %   r = obj.exp ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::exp() const
      r = __datamunge__(2895, obj);
    end
    function r = log (obj)
      % log  Natural logarithm ln x (NaN for x < 0, -inf at 0), with its exact chain-rule derivative.
      %
      %   r = obj.log ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::log() const
      r = __datamunge__(2896, obj);
    end
    function r = sqrt (obj)
      % sqrt  Square root √x (NaN for x < 0; infinite derivative at 0), with its exact chain-rule derivative.
      %
      %   r = obj.sqrt ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::sqrt() const
      r = __datamunge__(2897, obj);
    end
    function r = sin (obj)
      % sin  Sine (radians), with its exact chain-rule derivative.
      %
      %   r = obj.sin ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::sin() const
      r = __datamunge__(2898, obj);
    end
    function r = cos (obj)
      % cos  Cosine (radians), with its exact chain-rule derivative.
      %
      %   r = obj.cos ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::cos() const
      r = __datamunge__(2899, obj);
    end
    function r = tan (obj)
      % tan  Tangent (radians), with its exact chain-rule derivative.
      %
      %   r = obj.tan ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::tan() const
      r = __datamunge__(2900, obj);
    end
    function r = tanh (obj)
      % tanh  Hyperbolic tangent, with its exact chain-rule derivative.
      %
      %   r = obj.tanh ()
      %
      %   Output:
      %     r  datamunge.HyperDual: A new HyperDual.
      %
      %   C++: HyperDual datamunge::HyperDual::tanh() const
      r = __datamunge__(2901, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::HyperDual. Do not edit.
