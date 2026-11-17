classdef NnTensor < datamunge.NativeObject
  % NnTensor  C++ class datamunge::nn::NnTensor.
  %
  %   obj = datamunge.NnTensor ()
  %   obj = datamunge.NnTensor (values, shape)
  %   obj = datamunge.NnTensor (values, shape, dtype)
  %
  %   Methods:
  %     save_npy  C++ datamunge::nn::NnTensor::save_npy.
  %     defined  C++ datamunge::nn::NnTensor::defined.
  %     values  C++ datamunge::nn::NnTensor::values.
  %     shape  C++ datamunge::nn::NnTensor::shape.
  %     dtype  C++ datamunge::nn::NnTensor::dtype.
  %     ndim  C++ datamunge::nn::NnTensor::ndim.
  %     numel_  C++ datamunge::nn::NnTensor::numel.
  %     item  The value of a one-element tensor.
  %     at  The element at a full index.
  %     str  A short printable description with the values.
  %     astype  C++ datamunge::nn::NnTensor::astype.
  %     reshape  C++ datamunge::nn::NnTensor::reshape.
  %     transpose_  Permutes axes; an empty `perm` reverses them.
  %     slice  x[start:stop] along `axis`.
  %     concat  C++ datamunge::nn::NnTensor::concat.
  %     add  C++ datamunge::nn::NnTensor::add.
  %     sub  C++ datamunge::nn::NnTensor::sub.
  %     mul  C++ datamunge::nn::NnTensor::mul.
  %     div  C++ datamunge::nn::NnTensor::div.
  %     add_scalar  C++ datamunge::nn::NnTensor::add_scalar.
  %     mul_scalar  C++ datamunge::nn::NnTensor::mul_scalar.
  %     pow  C++ datamunge::nn::NnTensor::pow.
  %     neg  C++ datamunge::nn::NnTensor::neg.
  %     matmul  C++ datamunge::nn::NnTensor::matmul.
  %     exp  C++ datamunge::nn::NnTensor::exp.
  %     log  C++ datamunge::nn::NnTensor::log.
  %     sqrt  C++ datamunge::nn::NnTensor::sqrt.
  %     abs  C++ datamunge::nn::NnTensor::abs.
  %     square  C++ datamunge::nn::NnTensor::square.
  %     tanh  C++ datamunge::nn::NnTensor::tanh.
  %     sigmoid  C++ datamunge::nn::NnTensor::sigmoid.
  %     relu  C++ datamunge::nn::NnTensor::relu.
  %     activate  Any activation by name: "relu", "gelu", "silu", "softplus", "elu", "leaky_relu", ...
  %     softmax  C++ datamunge::nn::NnTensor::softmax.
  %     log_softmax  C++ datamunge::nn::NnTensor::log_softmax.
  %     sum  C++ datamunge::nn::NnTensor::sum.
  %     mean  C++ datamunge::nn::NnTensor::mean.
  %     max  C++ datamunge::nn::NnTensor::max.
  %     min  C++ datamunge::nn::NnTensor::min.
  %     argmax  C++ datamunge::nn::NnTensor::argmax.
  %     set_requires_grad  Marks a float tensor as a leaf whose gradient `backward` accumulates.
  %     requires_grad  C++ datamunge::nn::NnTensor::requires_grad.
  %     backward  Back-propagates from this one-element tensor into every leaf's grad().
  %     grad  The accumulated gradient (undefined if none).
  %     zero_grad  C++ datamunge::nn::NnTensor::zero_grad.
  %     detach  The same values without gradient history.
  %     to_string  C++ datamunge::octave_bindings::nn_tensor_repr.
  %
  %   Static methods:
  %     zeros  C++ datamunge::nn::NnTensor::zeros.
  %     ones  C++ datamunge::nn::NnTensor::ones.
  %     full  C++ datamunge::nn::NnTensor::full.
  %     arange  start, start + step, ... below stop.
  %     randn  Seeded N(mean, std^2) and U(low, high) samples.
  %     uniform  C++ datamunge::nn::NnTensor::uniform.
  %     load_npy  NumPy .npy files.
  %
  %   C++: class datamunge::nn::NnTensor
  methods
    function obj = NnTensor (varargin)
      % NnTensor  Construct a C++ datamunge::nn::NnTensor.
      %
      %   obj = datamunge.NnTensor ()
      %   obj = datamunge.NnTensor (values, shape)
      %   obj = datamunge.NnTensor (values, shape, dtype)
      %
      %   An empty (undefined) tensor.
      %
      %   C++: datamunge::nn::NnTensor::NnTensor()
      %   C++: datamunge::nn::NnTensor::NnTensor(const std::vector<double> & values, const std::vector<int> & shape, const std::string & dtype = "float32")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3134, false, varargin{:});
    end
    function save_npy (obj, path)
      % save_npy  C++ datamunge::nn::NnTensor::save_npy.
      %
      %   obj.save_npy (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   C++: void datamunge::nn::NnTensor::save_npy(const std::string & path) const
      __datamunge__(3087, obj, path);
    end
    function r = defined (obj)
      % defined  C++ datamunge::nn::NnTensor::defined.
      %
      %   r = obj.defined ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::nn::NnTensor::defined() const
      r = __datamunge__(3088, obj);
    end
    function r = values (obj)
      % values  C++ datamunge::nn::NnTensor::values.
      %
      %   r = obj.values ()
      %
      %   Output:
      %     r  double vector
      %
      %   C++: std::vector<double> datamunge::nn::NnTensor::values() const
      r = __datamunge__(3089, obj);
    end
    function r = shape (obj)
      % shape  C++ datamunge::nn::NnTensor::shape.
      %
      %   r = obj.shape ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<int> datamunge::nn::NnTensor::shape() const
      r = __datamunge__(3090, obj);
    end
    function r = dtype (obj)
      % dtype  C++ datamunge::nn::NnTensor::dtype.
      %
      %   r = obj.dtype ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::nn::NnTensor::dtype() const
      r = __datamunge__(3091, obj);
    end
    function r = ndim (obj)
      % ndim  C++ datamunge::nn::NnTensor::ndim.
      %
      %   r = obj.ndim ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::nn::NnTensor::ndim() const
      r = __datamunge__(3092, obj);
    end
    function r = numel_ (obj)
      % numel_  C++ datamunge::nn::NnTensor::numel.
      %
      %   r = obj.numel_ ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::nn::NnTensor::numel() const
      r = __datamunge__(3093, obj);
    end
    function r = item (obj)
      % item  The value of a one-element tensor.
      %
      %   r = obj.item ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::nn::NnTensor::item() const
      r = __datamunge__(3094, obj);
    end
    function r = at (obj, index)
      % at  The element at a full index.
      %
      %   r = obj.at (index)
      %
      %   Inputs:
      %     index  integer vector
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::nn::NnTensor::at(const std::vector<int> & index) const
      r = __datamunge__(3095, obj, index);
    end
    function r = str (obj)
      % str  A short printable description with the values.
      %
      %   r = obj.str ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::nn::NnTensor::str() const
      r = __datamunge__(3096, obj);
    end
    function r = astype (obj, dtype)
      % astype  C++ datamunge::nn::NnTensor::astype.
      %
      %   r = obj.astype (dtype)
      %
      %   Inputs:
      %     dtype  char row
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::astype(const std::string & dtype) const
      r = __datamunge__(3097, obj, dtype);
    end
    function r = reshape (obj, shape)
      % reshape  C++ datamunge::nn::NnTensor::reshape.
      %
      %   r = obj.reshape (shape)
      %
      %   Inputs:
      %     shape  integer vector
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::reshape(const std::vector<int> & shape) const
      r = __datamunge__(3098, obj, shape);
    end
    function r = transpose_ (obj, varargin)
      % transpose_  Permutes axes; an empty `perm` reverses them.
      %
      %   r = obj.transpose_ ()
      %   r = obj.transpose_ (perm)
      %
      %   Inputs:
      %     perm  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::transpose(const std::vector<int> & perm = {}) const
      r = __datamunge__(3099, obj, varargin{:});
    end
    function r = slice (obj, axis, start, stop)
      % slice  x[start:stop] along `axis`.
      %
      %   r = obj.slice (axis, start, stop)
      %
      %   Inputs:
      %     axis  integer
      %     start  integer
      %     stop  integer
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::slice(int axis, int start, int stop) const
      r = __datamunge__(3100, obj, axis, start, stop);
    end
    function r = concat (obj, other, axis)
      % concat  C++ datamunge::nn::NnTensor::concat.
      %
      %   r = obj.concat (other, axis)
      %
      %   Inputs:
      %     other  datamunge.NnTensor
      %     axis  integer
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::concat(const NnTensor & other, int axis) const
      r = __datamunge__(3101, obj, other, axis);
    end
    function r = add (obj, other)
      % add  C++ datamunge::nn::NnTensor::add.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.NnTensor
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::add(const NnTensor & other) const
      r = __datamunge__(3102, obj, other);
    end
    function r = sub (obj, other)
      % sub  C++ datamunge::nn::NnTensor::sub.
      %
      %   r = obj.sub (other)
      %
      %   Inputs:
      %     other  datamunge.NnTensor
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::sub(const NnTensor & other) const
      r = __datamunge__(3103, obj, other);
    end
    function r = mul (obj, other)
      % mul  C++ datamunge::nn::NnTensor::mul.
      %
      %   r = obj.mul (other)
      %
      %   Inputs:
      %     other  datamunge.NnTensor
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::mul(const NnTensor & other) const
      r = __datamunge__(3104, obj, other);
    end
    function r = div (obj, other)
      % div  C++ datamunge::nn::NnTensor::div.
      %
      %   r = obj.div (other)
      %
      %   Inputs:
      %     other  datamunge.NnTensor
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::div(const NnTensor & other) const
      r = __datamunge__(3105, obj, other);
    end
    function r = add_scalar (obj, value_)
      % add_scalar  C++ datamunge::nn::NnTensor::add_scalar.
      %
      %   r = obj.add_scalar (value_)
      %
      %   Inputs:
      %     value_  double
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::add_scalar(double value) const
      r = __datamunge__(3106, obj, value_);
    end
    function r = mul_scalar (obj, value_)
      % mul_scalar  C++ datamunge::nn::NnTensor::mul_scalar.
      %
      %   r = obj.mul_scalar (value_)
      %
      %   Inputs:
      %     value_  double
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::mul_scalar(double value) const
      r = __datamunge__(3107, obj, value_);
    end
    function r = pow (obj, exponent)
      % pow  C++ datamunge::nn::NnTensor::pow.
      %
      %   r = obj.pow (exponent)
      %
      %   Inputs:
      %     exponent  double
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::pow(double exponent) const
      r = __datamunge__(3108, obj, exponent);
    end
    function r = neg (obj)
      % neg  C++ datamunge::nn::NnTensor::neg.
      %
      %   r = obj.neg ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::neg() const
      r = __datamunge__(3109, obj);
    end
    function r = matmul (obj, varargin)
      % matmul  C++ datamunge::nn::NnTensor::matmul.
      %
      %   r = obj.matmul (other)
      %   r = obj.matmul (other, transpose_a)
      %   r = obj.matmul (other, transpose_a, transpose_b)
      %
      %   Inputs:
      %     other  datamunge.NnTensor
      %     transpose_a  logical, default false
      %     transpose_b  logical, default false
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::matmul(const NnTensor & other, bool transpose_a = false, bool transpose_b = false) const
      r = __datamunge__(3110, obj, varargin{:});
    end
    function r = exp (obj)
      % exp  C++ datamunge::nn::NnTensor::exp.
      %
      %   r = obj.exp ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::exp() const
      r = __datamunge__(3111, obj);
    end
    function r = log (obj)
      % log  C++ datamunge::nn::NnTensor::log.
      %
      %   r = obj.log ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::log() const
      r = __datamunge__(3112, obj);
    end
    function r = sqrt (obj)
      % sqrt  C++ datamunge::nn::NnTensor::sqrt.
      %
      %   r = obj.sqrt ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::sqrt() const
      r = __datamunge__(3113, obj);
    end
    function r = abs (obj)
      % abs  C++ datamunge::nn::NnTensor::abs.
      %
      %   r = obj.abs ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::abs() const
      r = __datamunge__(3114, obj);
    end
    function r = square (obj)
      % square  C++ datamunge::nn::NnTensor::square.
      %
      %   r = obj.square ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::square() const
      r = __datamunge__(3115, obj);
    end
    function r = tanh (obj)
      % tanh  C++ datamunge::nn::NnTensor::tanh.
      %
      %   r = obj.tanh ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::tanh() const
      r = __datamunge__(3116, obj);
    end
    function r = sigmoid (obj)
      % sigmoid  C++ datamunge::nn::NnTensor::sigmoid.
      %
      %   r = obj.sigmoid ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::sigmoid() const
      r = __datamunge__(3117, obj);
    end
    function r = relu (obj)
      % relu  C++ datamunge::nn::NnTensor::relu.
      %
      %   r = obj.relu ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::relu() const
      r = __datamunge__(3118, obj);
    end
    function r = activate (obj, activation)
      % activate  Any activation by name: "relu", "gelu", "silu", "softplus", "elu", "leaky_relu", ...
      %
      %   r = obj.activate (activation)
      %
      %   Inputs:
      %     activation  char row
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::activate(const std::string & activation) const
      r = __datamunge__(3119, obj, activation);
    end
    function r = softmax (obj, varargin)
      % softmax  C++ datamunge::nn::NnTensor::softmax.
      %
      %   r = obj.softmax ()
      %   r = obj.softmax (axis)
      %
      %   Inputs:
      %     axis  integer, default -1
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::softmax(int axis = -1) const
      r = __datamunge__(3120, obj, varargin{:});
    end
    function r = log_softmax (obj, varargin)
      % log_softmax  C++ datamunge::nn::NnTensor::log_softmax.
      %
      %   r = obj.log_softmax ()
      %   r = obj.log_softmax (axis)
      %
      %   Inputs:
      %     axis  integer, default -1
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::log_softmax(int axis = -1) const
      r = __datamunge__(3121, obj, varargin{:});
    end
    function r = sum (obj, varargin)
      % sum  C++ datamunge::nn::NnTensor::sum.
      %
      %   r = obj.sum ()
      %   r = obj.sum (axes)
      %   r = obj.sum (axes, keepdims)
      %
      %   Inputs:
      %     axes  integer vector, default the C++ default
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::sum(const std::vector<int> & axes = {}, bool keepdims = false) const
      r = __datamunge__(3122, obj, varargin{:});
    end
    function r = mean (obj, varargin)
      % mean  C++ datamunge::nn::NnTensor::mean.
      %
      %   r = obj.mean ()
      %   r = obj.mean (axes)
      %   r = obj.mean (axes, keepdims)
      %
      %   Inputs:
      %     axes  integer vector, default the C++ default
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::mean(const std::vector<int> & axes = {}, bool keepdims = false) const
      r = __datamunge__(3123, obj, varargin{:});
    end
    function r = max (obj, varargin)
      % max  C++ datamunge::nn::NnTensor::max.
      %
      %   r = obj.max ()
      %   r = obj.max (axes)
      %   r = obj.max (axes, keepdims)
      %
      %   Inputs:
      %     axes  integer vector, default the C++ default
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::max(const std::vector<int> & axes = {}, bool keepdims = false) const
      r = __datamunge__(3124, obj, varargin{:});
    end
    function r = min (obj, varargin)
      % min  C++ datamunge::nn::NnTensor::min.
      %
      %   r = obj.min ()
      %   r = obj.min (axes)
      %   r = obj.min (axes, keepdims)
      %
      %   Inputs:
      %     axes  integer vector, default the C++ default
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::min(const std::vector<int> & axes = {}, bool keepdims = false) const
      r = __datamunge__(3125, obj, varargin{:});
    end
    function r = argmax (obj, varargin)
      % argmax  C++ datamunge::nn::NnTensor::argmax.
      %
      %   r = obj.argmax ()
      %   r = obj.argmax (axis)
      %
      %   Inputs:
      %     axis  integer, default -1
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::argmax(int axis = -1) const
      r = __datamunge__(3126, obj, varargin{:});
    end
    function set_requires_grad (obj, varargin)
      % set_requires_grad  Marks a float tensor as a leaf whose gradient `backward` accumulates.
      %
      %   obj.set_requires_grad ()
      %   obj.set_requires_grad (requires_grad)
      %
      %   Inputs:
      %     requires_grad  logical, default true
      %
      %   C++: void datamunge::nn::NnTensor::set_requires_grad(bool requires_grad = true)
      __datamunge__(3127, obj, varargin{:});
    end
    function r = requires_grad (obj)
      % requires_grad  C++ datamunge::nn::NnTensor::requires_grad.
      %
      %   r = obj.requires_grad ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::nn::NnTensor::requires_grad() const
      r = __datamunge__(3128, obj);
    end
    function backward (obj)
      % backward  Back-propagates from this one-element tensor into every leaf's grad().
      %
      %   obj.backward ()
      %
      %   C++: void datamunge::nn::NnTensor::backward()
      __datamunge__(3129, obj);
    end
    function r = grad (obj)
      % grad  The accumulated gradient (undefined if none).
      %
      %   r = obj.grad ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::grad() const
      r = __datamunge__(3130, obj);
    end
    function zero_grad (obj)
      % zero_grad  C++ datamunge::nn::NnTensor::zero_grad.
      %
      %   obj.zero_grad ()
      %
      %   C++: void datamunge::nn::NnTensor::zero_grad()
      __datamunge__(3131, obj);
    end
    function r = detach (obj)
      % detach  The same values without gradient history.
      %
      %   r = obj.detach ()
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::detach() const
      r = __datamunge__(3132, obj);
    end
    function r = to_string (obj)
      % to_string  C++ datamunge::octave_bindings::nn_tensor_repr.
      %
      %   r = obj.to_string ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::octave_bindings::nn_tensor_repr(const datamunge::nn::NnTensor & self)
      r = __datamunge__(3133, obj);
    end
  end
  methods (Static)
    function r = zeros (varargin)
      % zeros  C++ datamunge::nn::NnTensor::zeros.
      %
      %   r = datamunge.NnTensor.zeros (shape)
      %   r = datamunge.NnTensor.zeros (shape, dtype)
      %
      %   Inputs:
      %     shape  integer vector
      %     dtype  char row, default 'float32'
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::zeros(const std::vector<int> & shape, const std::string & dtype = "float32")
      r = __datamunge__(3080, varargin{:});
    end
    function r = ones (varargin)
      % ones  C++ datamunge::nn::NnTensor::ones.
      %
      %   r = datamunge.NnTensor.ones (shape)
      %   r = datamunge.NnTensor.ones (shape, dtype)
      %
      %   Inputs:
      %     shape  integer vector
      %     dtype  char row, default 'float32'
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::ones(const std::vector<int> & shape, const std::string & dtype = "float32")
      r = __datamunge__(3081, varargin{:});
    end
    function r = full (varargin)
      % full  C++ datamunge::nn::NnTensor::full.
      %
      %   r = datamunge.NnTensor.full (shape, value_)
      %   r = datamunge.NnTensor.full (shape, value_, dtype)
      %
      %   Inputs:
      %     shape  integer vector
      %     value_  double
      %     dtype  char row, default 'float32'
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::full(const std::vector<int> & shape, double value, const std::string & dtype = "float32")
      r = __datamunge__(3082, varargin{:});
    end
    function r = arange (varargin)
      % arange  start, start + step, ... below stop.
      %
      %   r = datamunge.NnTensor.arange (start, stop)
      %   r = datamunge.NnTensor.arange (start, stop, step)
      %   r = datamunge.NnTensor.arange (start, stop, step, dtype)
      %
      %   Inputs:
      %     start  double
      %     stop  double
      %     step  double, default 1.0
      %     dtype  char row, default 'float32'
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::arange(double start, double stop, double step = 1.0, const std::string & dtype = "float32")
      r = __datamunge__(3083, varargin{:});
    end
    function r = randn (varargin)
      % randn  Seeded N(mean, std^2) and U(low, high) samples.
      %
      %   r = datamunge.NnTensor.randn (shape, seed)
      %   r = datamunge.NnTensor.randn (shape, seed, mean)
      %   r = datamunge.NnTensor.randn (shape, seed, mean, std)
      %   r = datamunge.NnTensor.randn (shape, seed, mean, std, dtype)
      %
      %   Inputs:
      %     shape  integer vector
      %     seed  integer
      %     mean  double, default 0.0
      %     std  double, default 1.0
      %     dtype  char row, default 'float32'
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::randn(const std::vector<int> & shape, unsigned int seed, double mean = 0.0, double std = 1.0, const std::string & dtype = "float32")
      r = __datamunge__(3084, varargin{:});
    end
    function r = uniform (varargin)
      % uniform  C++ datamunge::nn::NnTensor::uniform.
      %
      %   r = datamunge.NnTensor.uniform (shape, seed)
      %   r = datamunge.NnTensor.uniform (shape, seed, low)
      %   r = datamunge.NnTensor.uniform (shape, seed, low, high)
      %   r = datamunge.NnTensor.uniform (shape, seed, low, high, dtype)
      %
      %   Inputs:
      %     shape  integer vector
      %     seed  integer
      %     low  double, default 0.0
      %     high  double, default 1.0
      %     dtype  char row, default 'float32'
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::uniform(const std::vector<int> & shape, unsigned int seed, double low = 0.0, double high = 1.0, const std::string & dtype = "float32")
      r = __datamunge__(3085, varargin{:});
    end
    function r = load_npy (path)
      % load_npy  NumPy .npy files.
      %
      %   r = datamunge.NnTensor.load_npy (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnTensor::load_npy(const std::string & path)
      r = __datamunge__(3086, path);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::nn::NnTensor. Do not edit.
