classdef Var < datamunge.NativeObject
  % Var  SWIG-friendly facade for datamunge::autodiff::Var -- a handle into a Tape.
  %
  %   obj = datamunge.Var (tape, value_)
  %
  %   Methods:
  %     value  The value of this node.
  %     index  This node's index on its Tape: where to read its gradient in Tape::backward()'s result.
  %     add  Sum: this + `other`, with the node recorded on the shared Tape.
  %     subtract  Difference: this - `other`, with the node recorded on the shared Tape.
  %     multiply  Product: this * `other`, with the node recorded on the shared Tape.
  %     divide  Quotient: this / `other`, with the node recorded on the shared Tape (IEEE inf/NaN when other is 0).
  %     negate  Negation -x, with its exact chain-rule derivative.
  %     add_scalar  Adds a constant `scalar`, recorded on the Tape.
  %     subtract_scalar  Subtracts a constant `scalar`, recorded on the Tape.
  %     multiply_scalar  Multiplies by a constant `scalar`, recorded on the Tape.
  %     divide_scalar  Divides by a constant `scalar`, recorded on the Tape.
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
  %   C++: class datamunge::Var
  methods
    function obj = Var (varargin)
      % Var  Construct a C++ datamunge::Var.
      %
      %   obj = datamunge.Var (tape, value_)
      %
      %   Records a new independent leaf variable on `tape`.
      %
      %   Inputs:
      %     tape  datamunge.Tape: The tape that records this variable and every expression built from it; it must outlive the Var, and operands of one expression must share it.
      %     value_  double: The leaf's value.
      %
      %   C++: datamunge::Var::Var(Tape & tape, double value)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2923, false, varargin{:});
    end
    function r = value (obj)
      % value  The value of this node.
      %
      %   r = obj.value ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Var::value() const
      r = __datamunge__(2903, obj);
    end
    function r = index (obj)
      % index  This node's index on its Tape: where to read its gradient in Tape::backward()'s result.
      %
      %   r = obj.index ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Var::index() const
      r = __datamunge__(2904, obj);
    end
    function r = add (obj, other)
      % add  Sum: this + `other`, with the node recorded on the shared Tape.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.Var: Right operand (recorded on the same Tape).
      %
      %   Output:
      %     r  datamunge.Var: A new Var; neither operand changes.
      %
      %   C++: Var datamunge::Var::add(const Var & other) const
      r = __datamunge__(2905, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  Difference: this - `other`, with the node recorded on the shared Tape.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.Var: Right operand (recorded on the same Tape).
      %
      %   Output:
      %     r  datamunge.Var: A new Var; neither operand changes.
      %
      %   C++: Var datamunge::Var::subtract(const Var & other) const
      r = __datamunge__(2906, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  Product: this * `other`, with the node recorded on the shared Tape.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.Var: Right operand (recorded on the same Tape).
      %
      %   Output:
      %     r  datamunge.Var: A new Var; neither operand changes.
      %
      %   C++: Var datamunge::Var::multiply(const Var & other) const
      r = __datamunge__(2907, obj, other);
    end
    function r = divide (obj, other)
      % divide  Quotient: this / `other`, with the node recorded on the shared Tape (IEEE inf/NaN when other is 0).
      %
      %   r = obj.divide (other)
      %
      %   Inputs:
      %     other  datamunge.Var: Right operand (recorded on the same Tape).
      %
      %   Output:
      %     r  datamunge.Var: A new Var; neither operand changes.
      %
      %   C++: Var datamunge::Var::divide(const Var & other) const
      r = __datamunge__(2908, obj, other);
    end
    function r = negate (obj)
      % negate  Negation -x, with its exact chain-rule derivative.
      %
      %   r = obj.negate ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::negate() const
      r = __datamunge__(2909, obj);
    end
    function r = add_scalar (obj, scalar)
      % add_scalar  Adds a constant `scalar`, recorded on the Tape.
      %
      %   r = obj.add_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::add_scalar(double scalar) const
      r = __datamunge__(2910, obj, scalar);
    end
    function r = subtract_scalar (obj, scalar)
      % subtract_scalar  Subtracts a constant `scalar`, recorded on the Tape.
      %
      %   r = obj.subtract_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::subtract_scalar(double scalar) const
      r = __datamunge__(2911, obj, scalar);
    end
    function r = multiply_scalar (obj, scalar)
      % multiply_scalar  Multiplies by a constant `scalar`, recorded on the Tape.
      %
      %   r = obj.multiply_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::multiply_scalar(double scalar) const
      r = __datamunge__(2912, obj, scalar);
    end
    function r = divide_scalar (obj, scalar)
      % divide_scalar  Divides by a constant `scalar`, recorded on the Tape.
      %
      %   r = obj.divide_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double: Constant right operand.
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::divide_scalar(double scalar) const
      r = __datamunge__(2913, obj, scalar);
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
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::pow(double exponent) const
      r = __datamunge__(2914, obj, exponent);
    end
    function r = exp (obj)
      % exp  Exponential e^x, with its exact chain-rule derivative.
      %
      %   r = obj.exp ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::exp() const
      r = __datamunge__(2915, obj);
    end
    function r = log (obj)
      % log  Natural logarithm ln x (NaN for x < 0, -inf at 0), with its exact chain-rule derivative.
      %
      %   r = obj.log ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::log() const
      r = __datamunge__(2916, obj);
    end
    function r = sqrt (obj)
      % sqrt  Square root √x (NaN for x < 0; infinite derivative at 0), with its exact chain-rule derivative.
      %
      %   r = obj.sqrt ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::sqrt() const
      r = __datamunge__(2917, obj);
    end
    function r = sin (obj)
      % sin  Sine (radians), with its exact chain-rule derivative.
      %
      %   r = obj.sin ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::sin() const
      r = __datamunge__(2918, obj);
    end
    function r = cos (obj)
      % cos  Cosine (radians), with its exact chain-rule derivative.
      %
      %   r = obj.cos ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::cos() const
      r = __datamunge__(2919, obj);
    end
    function r = tan (obj)
      % tan  Tangent (radians), with its exact chain-rule derivative.
      %
      %   r = obj.tan ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::tan() const
      r = __datamunge__(2920, obj);
    end
    function r = tanh (obj)
      % tanh  Hyperbolic tangent, with its exact chain-rule derivative.
      %
      %   r = obj.tanh ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::tanh() const
      r = __datamunge__(2921, obj);
    end
    function r = abs (obj)
      % abs  Absolute value |x| (the derivative at 0 uses the x ≥ 0 branch), with its exact chain-rule derivative.
      %
      %   r = obj.abs ()
      %
      %   Output:
      %     r  datamunge.Var: A new Var.
      %
      %   C++: Var datamunge::Var::abs() const
      r = __datamunge__(2922, obj);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Var. Do not edit.
