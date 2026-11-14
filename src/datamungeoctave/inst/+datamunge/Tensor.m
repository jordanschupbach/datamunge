classdef Tensor < datamunge.NativeObject
  % Tensor  SWIG-friendly facade for datamunge::linalg::Tensor -- a dense, row-major, N-dimensional array that can hold float64, bool, or string elements (chosen at construction).
  %
  %   obj = datamunge.Tensor (shape)
  %   obj = datamunge.Tensor (shape, dtype)
  %
  %   Methods:
  %     ndim  C++ datamunge::Tensor::ndim.
  %     shape  C++ datamunge::Tensor::shape.
  %     size_  C++ datamunge::Tensor::size.
  %     dtype_name  C++ datamunge::Tensor::dtype_name.
  %     at  C++ datamunge::Tensor::at.
  %     set  C++ datamunge::Tensor::set.
  %     string_at  C++ datamunge::Tensor::string_at.
  %     set_string  C++ datamunge::Tensor::set_string.
  %     at_flat  C++ datamunge::Tensor::at_flat.
  %     set_flat  C++ datamunge::Tensor::set_flat.
  %     string_at_flat  C++ datamunge::Tensor::string_at_flat.
  %     set_string_flat  C++ datamunge::Tensor::set_string_flat.
  %     reshape  C++ datamunge::Tensor::reshape.
  %     flatten  C++ datamunge::Tensor::flatten.
  %     transpose_  C++ datamunge::Tensor::transpose.
  %     squeeze  C++ datamunge::Tensor::squeeze.
  %     squeeze_axis  C++ datamunge::Tensor::squeeze_axis.
  %     expand_dims  C++ datamunge::Tensor::expand_dims.
  %     slice  C++ datamunge::Tensor::slice.
  %     index_select  C++ datamunge::Tensor::index_select.
  %     add  C++ datamunge::Tensor::add.
  %     subtract  C++ datamunge::Tensor::subtract.
  %     multiply  C++ datamunge::Tensor::multiply.
  %     divide  C++ datamunge::Tensor::divide.
  %     power_  C++ datamunge::Tensor::power.
  %     add_scalar  C++ datamunge::Tensor::add_scalar.
  %     subtract_scalar  C++ datamunge::Tensor::subtract_scalar.
  %     multiply_scalar  C++ datamunge::Tensor::multiply_scalar.
  %     divide_scalar  C++ datamunge::Tensor::divide_scalar.
  %     power_scalar  C++ datamunge::Tensor::power_scalar.
  %     negate  C++ datamunge::Tensor::negate.
  %     abs  C++ datamunge::Tensor::abs.
  %     sqrt  C++ datamunge::Tensor::sqrt.
  %     exp  C++ datamunge::Tensor::exp.
  %     log  C++ datamunge::Tensor::log.
  %     apply  Applies a user-supplied Callback elementwise. Requires a numeric dtype; result is float64.
  %     equal  C++ datamunge::Tensor::equal.
  %     not_equal  C++ datamunge::Tensor::not_equal.
  %     less  C++ datamunge::Tensor::less.
  %     less_equal  C++ datamunge::Tensor::less_equal.
  %     greater  C++ datamunge::Tensor::greater.
  %     greater_equal  C++ datamunge::Tensor::greater_equal.
  %     sum  C++ datamunge::Tensor::sum.
  %     mean  C++ datamunge::Tensor::mean.
  %     max  C++ datamunge::Tensor::max.
  %     min  C++ datamunge::Tensor::min.
  %     prod  C++ datamunge::Tensor::prod.
  %     argmax  C++ datamunge::Tensor::argmax.
  %     argmin  C++ datamunge::Tensor::argmin.
  %     all  C++ datamunge::Tensor::all.
  %     any  C++ datamunge::Tensor::any.
  %     sum_axis  C++ datamunge::Tensor::sum_axis.
  %     mean_axis  C++ datamunge::Tensor::mean_axis.
  %     max_axis  C++ datamunge::Tensor::max_axis.
  %     min_axis  C++ datamunge::Tensor::min_axis.
  %     prod_axis  C++ datamunge::Tensor::prod_axis.
  %     argmax_axis  C++ datamunge::Tensor::argmax_axis.
  %     argmin_axis  C++ datamunge::Tensor::argmin_axis.
  %     matmul  C++ datamunge::Tensor::matmul.
  %     dot  C++ datamunge::Tensor::dot.
  %     outer  C++ datamunge::Tensor::outer.
  %     max_pool2d  C++ datamunge::Tensor::max_pool2d.
  %     avg_pool2d  C++ datamunge::Tensor::avg_pool2d.
  %     relu  C++ datamunge::Tensor::relu.
  %     sigmoid  C++ datamunge::Tensor::sigmoid.
  %     softmax  C++ datamunge::Tensor::softmax.
  %     to_string  C++ datamunge::Tensor::to_string.
  %
  %   Static methods:
  %     zeros  C++ datamunge::Tensor::zeros.
  %     ones  C++ datamunge::Tensor::ones.
  %     full  C++ datamunge::Tensor::full.
  %     from_values  C++ datamunge::Tensor::from_values.
  %     from_bool_values  C++ datamunge::Tensor::from_bool_values.
  %     from_string_values  C++ datamunge::Tensor::from_string_values.
  %     arange  C++ datamunge::Tensor::arange.
  %     eye  C++ datamunge::Tensor::eye.
  %     concatenate2  Concatenates two tensors along an existing axis (shapes must match on every other axis).
  %     stack2  Stacks two same-shaped tensors along a new axis inserted at position `axis`.
  %     from_image  The Tensor's own image_to_tensor() bridge: `img` (already directly SWIG-bindable, no facade needed) as a [channels, height, width] tensor normalized to [0, 1].
  %     conv2d  Basic (inference-only) neural-network building blocks -- see datamunge::cv::conv2d/max_pool2d/avg_pool2d/relu/sigmoid/softmax for the underlying implementation and full documentation of shapes/semantics.
  %
  %   C++: class datamunge::Tensor
  methods
    function obj = Tensor (varargin)
      % Tensor  Construct a C++ datamunge::Tensor.
      %
      %   obj = datamunge.Tensor (shape)
      %   obj = datamunge.Tensor (shape, dtype)
      %
      %   Inputs:
      %     shape  integer vector
      %     dtype  char row, default 'float64': One of "float64", "bool", or "string".
      %
      %   C++: datamunge::Tensor::Tensor(const std::vector<std::size_t> & shape, const std::string & dtype = "float64")
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2859, false, varargin{:});
    end
    function r = ndim (obj)
      % ndim  C++ datamunge::Tensor::ndim.
      %
      %   r = obj.ndim ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Tensor::ndim() const
      r = __datamunge__(2788, obj);
    end
    function r = shape (obj)
      % shape  C++ datamunge::Tensor::shape.
      %
      %   r = obj.shape ()
      %
      %   Output:
      %     r  integer vector
      %
      %   C++: std::vector<std::size_t> datamunge::Tensor::shape() const
      r = __datamunge__(2789, obj);
    end
    function r = size_ (obj)
      % size_  C++ datamunge::Tensor::size.
      %
      %   r = obj.size_ ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Tensor::size() const
      r = __datamunge__(2790, obj);
    end
    function r = dtype_name (obj)
      % dtype_name  C++ datamunge::Tensor::dtype_name.
      %
      %   r = obj.dtype_name ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Tensor::dtype_name() const
      r = __datamunge__(2791, obj);
    end
    function r = at (obj, index)
      % at  C++ datamunge::Tensor::at.
      %
      %   r = obj.at (index)
      %
      %   Inputs:
      %     index  integer vector
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::at(const std::vector<std::size_t> & index) const
      r = __datamunge__(2792, obj, index);
    end
    function set (obj, index, value_)
      % set  C++ datamunge::Tensor::set.
      %
      %   obj.set (index, value_)
      %
      %   Inputs:
      %     index  integer vector
      %     value_  double
      %
      %   C++: void datamunge::Tensor::set(const std::vector<std::size_t> & index, double value)
      __datamunge__(2793, obj, index, value_);
    end
    function r = string_at (obj, index)
      % string_at  C++ datamunge::Tensor::string_at.
      %
      %   r = obj.string_at (index)
      %
      %   Inputs:
      %     index  integer vector
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Tensor::string_at(const std::vector<std::size_t> & index) const
      r = __datamunge__(2794, obj, index);
    end
    function set_string (obj, index, value_)
      % set_string  C++ datamunge::Tensor::set_string.
      %
      %   obj.set_string (index, value_)
      %
      %   Inputs:
      %     index  integer vector
      %     value_  char row
      %
      %   C++: void datamunge::Tensor::set_string(const std::vector<std::size_t> & index, const std::string & value)
      __datamunge__(2795, obj, index, value_);
    end
    function r = at_flat (obj, i)
      % at_flat  C++ datamunge::Tensor::at_flat.
      %
      %   r = obj.at_flat (i)
      %
      %   Inputs:
      %     i  integer
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::at_flat(std::size_t i) const
      r = __datamunge__(2796, obj, i);
    end
    function set_flat (obj, i, value_)
      % set_flat  C++ datamunge::Tensor::set_flat.
      %
      %   obj.set_flat (i, value_)
      %
      %   Inputs:
      %     i  integer
      %     value_  double
      %
      %   C++: void datamunge::Tensor::set_flat(std::size_t i, double value)
      __datamunge__(2797, obj, i, value_);
    end
    function r = string_at_flat (obj, i)
      % string_at_flat  C++ datamunge::Tensor::string_at_flat.
      %
      %   r = obj.string_at_flat (i)
      %
      %   Inputs:
      %     i  integer
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Tensor::string_at_flat(std::size_t i) const
      r = __datamunge__(2798, obj, i);
    end
    function set_string_flat (obj, i, value_)
      % set_string_flat  C++ datamunge::Tensor::set_string_flat.
      %
      %   obj.set_string_flat (i, value_)
      %
      %   Inputs:
      %     i  integer
      %     value_  char row
      %
      %   C++: void datamunge::Tensor::set_string_flat(std::size_t i, const std::string & value)
      __datamunge__(2799, obj, i, value_);
    end
    function r = reshape (obj, new_shape)
      % reshape  C++ datamunge::Tensor::reshape.
      %
      %   r = obj.reshape (new_shape)
      %
      %   Inputs:
      %     new_shape  integer vector
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::reshape(const std::vector<std::size_t> & new_shape) const
      r = __datamunge__(2800, obj, new_shape);
    end
    function r = flatten (obj)
      % flatten  C++ datamunge::Tensor::flatten.
      %
      %   r = obj.flatten ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::flatten() const
      r = __datamunge__(2801, obj);
    end
    function r = transpose_ (obj, varargin)
      % transpose_  C++ datamunge::Tensor::transpose.
      %
      %   r = obj.transpose_ ()
      %   r = obj.transpose_ (permutation)
      %
      %   Inputs:
      %     permutation  integer vector, default the C++ default
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::transpose(const std::vector<std::size_t> & permutation = {}) const
      r = __datamunge__(2802, obj, varargin{:});
    end
    function r = squeeze (obj)
      % squeeze  C++ datamunge::Tensor::squeeze.
      %
      %   r = obj.squeeze ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::squeeze() const
      r = __datamunge__(2803, obj);
    end
    function r = squeeze_axis (obj, axis)
      % squeeze_axis  C++ datamunge::Tensor::squeeze_axis.
      %
      %   r = obj.squeeze_axis (axis)
      %
      %   Inputs:
      %     axis  integer
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::squeeze_axis(std::size_t axis) const
      r = __datamunge__(2804, obj, axis);
    end
    function r = expand_dims (obj, axis)
      % expand_dims  C++ datamunge::Tensor::expand_dims.
      %
      %   r = obj.expand_dims (axis)
      %
      %   Inputs:
      %     axis  integer
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::expand_dims(std::size_t axis) const
      r = __datamunge__(2805, obj, axis);
    end
    function r = slice (obj, varargin)
      % slice  C++ datamunge::Tensor::slice.
      %
      %   r = obj.slice (axis, start, stop)
      %   r = obj.slice (axis, start, stop, step)
      %
      %   Inputs:
      %     axis  integer
      %     start  integer
      %     stop  integer
      %     step  integer, default 1
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::slice(std::size_t axis, std::size_t start, std::size_t stop, std::size_t step = 1) const
      r = __datamunge__(2806, obj, varargin{:});
    end
    function r = index_select (obj, axis, indices)
      % index_select  C++ datamunge::Tensor::index_select.
      %
      %   r = obj.index_select (axis, indices)
      %
      %   Inputs:
      %     axis  integer
      %     indices  integer vector
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::index_select(std::size_t axis, const std::vector<std::size_t> & indices) const
      r = __datamunge__(2807, obj, axis, indices);
    end
    function r = add (obj, other)
      % add  C++ datamunge::Tensor::add.
      %
      %   r = obj.add (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::add(const Tensor & other) const
      r = __datamunge__(2810, obj, other);
    end
    function r = subtract (obj, other)
      % subtract  C++ datamunge::Tensor::subtract.
      %
      %   r = obj.subtract (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::subtract(const Tensor & other) const
      r = __datamunge__(2811, obj, other);
    end
    function r = multiply (obj, other)
      % multiply  C++ datamunge::Tensor::multiply.
      %
      %   r = obj.multiply (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::multiply(const Tensor & other) const
      r = __datamunge__(2812, obj, other);
    end
    function r = divide (obj, other)
      % divide  C++ datamunge::Tensor::divide.
      %
      %   r = obj.divide (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::divide(const Tensor & other) const
      r = __datamunge__(2813, obj, other);
    end
    function r = power_ (obj, other)
      % power_  C++ datamunge::Tensor::power.
      %
      %   r = obj.power_ (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::power(const Tensor & other) const
      r = __datamunge__(2814, obj, other);
    end
    function r = add_scalar (obj, scalar)
      % add_scalar  C++ datamunge::Tensor::add_scalar.
      %
      %   r = obj.add_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::add_scalar(double scalar) const
      r = __datamunge__(2815, obj, scalar);
    end
    function r = subtract_scalar (obj, scalar)
      % subtract_scalar  C++ datamunge::Tensor::subtract_scalar.
      %
      %   r = obj.subtract_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::subtract_scalar(double scalar) const
      r = __datamunge__(2816, obj, scalar);
    end
    function r = multiply_scalar (obj, scalar)
      % multiply_scalar  C++ datamunge::Tensor::multiply_scalar.
      %
      %   r = obj.multiply_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::multiply_scalar(double scalar) const
      r = __datamunge__(2817, obj, scalar);
    end
    function r = divide_scalar (obj, scalar)
      % divide_scalar  C++ datamunge::Tensor::divide_scalar.
      %
      %   r = obj.divide_scalar (scalar)
      %
      %   Inputs:
      %     scalar  double
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::divide_scalar(double scalar) const
      r = __datamunge__(2818, obj, scalar);
    end
    function r = power_scalar (obj, exponent)
      % power_scalar  C++ datamunge::Tensor::power_scalar.
      %
      %   r = obj.power_scalar (exponent)
      %
      %   Inputs:
      %     exponent  double
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::power_scalar(double exponent) const
      r = __datamunge__(2819, obj, exponent);
    end
    function r = negate (obj)
      % negate  C++ datamunge::Tensor::negate.
      %
      %   r = obj.negate ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::negate() const
      r = __datamunge__(2820, obj);
    end
    function r = abs (obj)
      % abs  C++ datamunge::Tensor::abs.
      %
      %   r = obj.abs ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::abs() const
      r = __datamunge__(2821, obj);
    end
    function r = sqrt (obj)
      % sqrt  C++ datamunge::Tensor::sqrt.
      %
      %   r = obj.sqrt ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::sqrt() const
      r = __datamunge__(2822, obj);
    end
    function r = exp (obj)
      % exp  C++ datamunge::Tensor::exp.
      %
      %   r = obj.exp ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::exp() const
      r = __datamunge__(2823, obj);
    end
    function r = log (obj)
      % log  C++ datamunge::Tensor::log.
      %
      %   r = obj.log ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::log() const
      r = __datamunge__(2824, obj);
    end
    function r = apply (obj, callback)
      % apply  Applies a user-supplied Callback elementwise. Requires a numeric dtype; result is float64.
      %
      %   r = obj.apply (callback)
      %
      %   Inputs:
      %     callback  datamunge.Callback or []
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::apply(Callback * callback) const
      r = __datamunge__(2825, obj, callback);
    end
    function r = equal (obj, other)
      % equal  C++ datamunge::Tensor::equal.
      %
      %   r = obj.equal (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::equal(const Tensor & other) const
      r = __datamunge__(2826, obj, other);
    end
    function r = not_equal (obj, other)
      % not_equal  C++ datamunge::Tensor::not_equal.
      %
      %   r = obj.not_equal (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::not_equal(const Tensor & other) const
      r = __datamunge__(2827, obj, other);
    end
    function r = less (obj, other)
      % less  C++ datamunge::Tensor::less.
      %
      %   r = obj.less (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::less(const Tensor & other) const
      r = __datamunge__(2828, obj, other);
    end
    function r = less_equal (obj, other)
      % less_equal  C++ datamunge::Tensor::less_equal.
      %
      %   r = obj.less_equal (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::less_equal(const Tensor & other) const
      r = __datamunge__(2829, obj, other);
    end
    function r = greater (obj, other)
      % greater  C++ datamunge::Tensor::greater.
      %
      %   r = obj.greater (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::greater(const Tensor & other) const
      r = __datamunge__(2830, obj, other);
    end
    function r = greater_equal (obj, other)
      % greater_equal  C++ datamunge::Tensor::greater_equal.
      %
      %   r = obj.greater_equal (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::greater_equal(const Tensor & other) const
      r = __datamunge__(2831, obj, other);
    end
    function r = sum (obj)
      % sum  C++ datamunge::Tensor::sum.
      %
      %   r = obj.sum ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::sum() const
      r = __datamunge__(2832, obj);
    end
    function r = mean (obj)
      % mean  C++ datamunge::Tensor::mean.
      %
      %   r = obj.mean ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::mean() const
      r = __datamunge__(2833, obj);
    end
    function r = max (obj)
      % max  C++ datamunge::Tensor::max.
      %
      %   r = obj.max ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::max() const
      r = __datamunge__(2834, obj);
    end
    function r = min (obj)
      % min  C++ datamunge::Tensor::min.
      %
      %   r = obj.min ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::min() const
      r = __datamunge__(2835, obj);
    end
    function r = prod (obj)
      % prod  C++ datamunge::Tensor::prod.
      %
      %   r = obj.prod ()
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::prod() const
      r = __datamunge__(2836, obj);
    end
    function r = argmax (obj)
      % argmax  C++ datamunge::Tensor::argmax.
      %
      %   r = obj.argmax ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Tensor::argmax() const
      r = __datamunge__(2837, obj);
    end
    function r = argmin (obj)
      % argmin  C++ datamunge::Tensor::argmin.
      %
      %   r = obj.argmin ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: std::size_t datamunge::Tensor::argmin() const
      r = __datamunge__(2838, obj);
    end
    function r = all (obj)
      % all  C++ datamunge::Tensor::all.
      %
      %   r = obj.all ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::Tensor::all() const
      r = __datamunge__(2839, obj);
    end
    function r = any (obj)
      % any  C++ datamunge::Tensor::any.
      %
      %   r = obj.any ()
      %
      %   Output:
      %     r  logical
      %
      %   C++: bool datamunge::Tensor::any() const
      r = __datamunge__(2840, obj);
    end
    function r = sum_axis (obj, varargin)
      % sum_axis  C++ datamunge::Tensor::sum_axis.
      %
      %   r = obj.sum_axis (axis)
      %   r = obj.sum_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::sum_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2841, obj, varargin{:});
    end
    function r = mean_axis (obj, varargin)
      % mean_axis  C++ datamunge::Tensor::mean_axis.
      %
      %   r = obj.mean_axis (axis)
      %   r = obj.mean_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::mean_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2842, obj, varargin{:});
    end
    function r = max_axis (obj, varargin)
      % max_axis  C++ datamunge::Tensor::max_axis.
      %
      %   r = obj.max_axis (axis)
      %   r = obj.max_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::max_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2843, obj, varargin{:});
    end
    function r = min_axis (obj, varargin)
      % min_axis  C++ datamunge::Tensor::min_axis.
      %
      %   r = obj.min_axis (axis)
      %   r = obj.min_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::min_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2844, obj, varargin{:});
    end
    function r = prod_axis (obj, varargin)
      % prod_axis  C++ datamunge::Tensor::prod_axis.
      %
      %   r = obj.prod_axis (axis)
      %   r = obj.prod_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::prod_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2845, obj, varargin{:});
    end
    function r = argmax_axis (obj, varargin)
      % argmax_axis  C++ datamunge::Tensor::argmax_axis.
      %
      %   r = obj.argmax_axis (axis)
      %   r = obj.argmax_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::argmax_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2846, obj, varargin{:});
    end
    function r = argmin_axis (obj, varargin)
      % argmin_axis  C++ datamunge::Tensor::argmin_axis.
      %
      %   r = obj.argmin_axis (axis)
      %   r = obj.argmin_axis (axis, keepdims)
      %
      %   Inputs:
      %     axis  integer
      %     keepdims  logical, default false
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::argmin_axis(std::size_t axis, bool keepdims = false) const
      r = __datamunge__(2847, obj, varargin{:});
    end
    function r = matmul (obj, other)
      % matmul  C++ datamunge::Tensor::matmul.
      %
      %   r = obj.matmul (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::matmul(const Tensor & other) const
      r = __datamunge__(2848, obj, other);
    end
    function r = dot (obj, other)
      % dot  C++ datamunge::Tensor::dot.
      %
      %   r = obj.dot (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::Tensor::dot(const Tensor & other) const
      r = __datamunge__(2849, obj, other);
    end
    function r = outer (obj, other)
      % outer  C++ datamunge::Tensor::outer.
      %
      %   r = obj.outer (other)
      %
      %   Inputs:
      %     other  datamunge.Tensor
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::outer(const Tensor & other) const
      r = __datamunge__(2850, obj, other);
    end
    function r = max_pool2d (obj, varargin)
      % max_pool2d  C++ datamunge::Tensor::max_pool2d.
      %
      %   r = obj.max_pool2d (pool_size)
      %   r = obj.max_pool2d (pool_size, stride)
      %
      %   Inputs:
      %     pool_size  integer
      %     stride  integer, default -1
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::max_pool2d(int pool_size, int stride = -1) const
      r = __datamunge__(2853, obj, varargin{:});
    end
    function r = avg_pool2d (obj, varargin)
      % avg_pool2d  C++ datamunge::Tensor::avg_pool2d.
      %
      %   r = obj.avg_pool2d (pool_size)
      %   r = obj.avg_pool2d (pool_size, stride)
      %
      %   Inputs:
      %     pool_size  integer
      %     stride  integer, default -1
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::avg_pool2d(int pool_size, int stride = -1) const
      r = __datamunge__(2854, obj, varargin{:});
    end
    function r = relu (obj)
      % relu  C++ datamunge::Tensor::relu.
      %
      %   r = obj.relu ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::relu() const
      r = __datamunge__(2855, obj);
    end
    function r = sigmoid (obj)
      % sigmoid  C++ datamunge::Tensor::sigmoid.
      %
      %   r = obj.sigmoid ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::sigmoid() const
      r = __datamunge__(2856, obj);
    end
    function r = softmax (obj)
      % softmax  C++ datamunge::Tensor::softmax.
      %
      %   r = obj.softmax ()
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::softmax() const
      r = __datamunge__(2857, obj);
    end
    function r = to_string (obj, varargin)
      % to_string  C++ datamunge::Tensor::to_string.
      %
      %   r = obj.to_string ()
      %   r = obj.to_string (max_elements)
      %
      %   Inputs:
      %     max_elements  integer, default 100
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::Tensor::to_string(std::size_t max_elements = 100) const
      r = __datamunge__(2858, obj, varargin{:});
    end
  end
  methods (Static)
    function r = zeros (shape)
      % zeros  C++ datamunge::Tensor::zeros.
      %
      %   r = datamunge.Tensor.zeros (shape)
      %
      %   Inputs:
      %     shape  integer vector
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::zeros(const std::vector<std::size_t> & shape)
      r = __datamunge__(2780, shape);
    end
    function r = ones (shape)
      % ones  C++ datamunge::Tensor::ones.
      %
      %   r = datamunge.Tensor.ones (shape)
      %
      %   Inputs:
      %     shape  integer vector
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::ones(const std::vector<std::size_t> & shape)
      r = __datamunge__(2781, shape);
    end
    function r = full (shape, value_)
      % full  C++ datamunge::Tensor::full.
      %
      %   r = datamunge.Tensor.full (shape, value_)
      %
      %   Inputs:
      %     shape  integer vector
      %     value_  double
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::full(const std::vector<std::size_t> & shape, double value)
      r = __datamunge__(2782, shape, value_);
    end
    function r = from_values (shape, values)
      % from_values  C++ datamunge::Tensor::from_values.
      %
      %   r = datamunge.Tensor.from_values (shape, values)
      %
      %   Inputs:
      %     shape  integer vector
      %     values  double vector
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::from_values(const std::vector<std::size_t> & shape, const std::vector<double> & values)
      r = __datamunge__(2783, shape, values);
    end
    function r = from_bool_values (shape, values)
      % from_bool_values  C++ datamunge::Tensor::from_bool_values.
      %
      %   r = datamunge.Tensor.from_bool_values (shape, values)
      %
      %   Inputs:
      %     shape  integer vector
      %     values  integer vector
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::from_bool_values(const std::vector<std::size_t> & shape, const std::vector<int> & values)
      r = __datamunge__(2784, shape, values);
    end
    function r = from_string_values (shape, values)
      % from_string_values  C++ datamunge::Tensor::from_string_values.
      %
      %   r = datamunge.Tensor.from_string_values (shape, values)
      %
      %   Inputs:
      %     shape  integer vector
      %     values  cell array of char
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::from_string_values(const std::vector<std::size_t> & shape, const std::vector<std::string> & values)
      r = __datamunge__(2785, shape, values);
    end
    function r = arange (varargin)
      % arange  C++ datamunge::Tensor::arange.
      %
      %   r = datamunge.Tensor.arange (start, stop)
      %   r = datamunge.Tensor.arange (start, stop, step)
      %
      %   Inputs:
      %     start  double
      %     stop  double
      %     step  double, default 1.0
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::arange(double start, double stop, double step = 1.0)
      r = __datamunge__(2786, varargin{:});
    end
    function r = eye (n)
      % eye  C++ datamunge::Tensor::eye.
      %
      %   r = datamunge.Tensor.eye (n)
      %
      %   Inputs:
      %     n  integer
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::eye(std::size_t n)
      r = __datamunge__(2787, n);
    end
    function r = concatenate2 (a, b, axis)
      % concatenate2  Concatenates two tensors along an existing axis (shapes must match on every other axis).
      %
      %   r = datamunge.Tensor.concatenate2 (a, b, axis)
      %
      %   Inputs:
      %     a  datamunge.Tensor
      %     b  datamunge.Tensor
      %     axis  integer
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::concatenate2(const Tensor & a, const Tensor & b, std::size_t axis)
      r = __datamunge__(2808, a, b, axis);
    end
    function r = stack2 (a, b, axis)
      % stack2  Stacks two same-shaped tensors along a new axis inserted at position `axis`.
      %
      %   r = datamunge.Tensor.stack2 (a, b, axis)
      %
      %   Inputs:
      %     a  datamunge.Tensor
      %     b  datamunge.Tensor
      %     axis  integer
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::stack2(const Tensor & a, const Tensor & b, std::size_t axis)
      r = __datamunge__(2809, a, b, axis);
    end
    function r = from_image (img)
      % from_image  The Tensor's own image_to_tensor() bridge: `img` (already directly SWIG-bindable, no facade needed) as a [channels, height, width] tensor normalized to [0, 1].
      %
      %   r = datamunge.Tensor.from_image (img)
      %
      %   Inputs:
      %     img  datamunge.Image
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::from_image(const datamunge::image::Image & img)
      r = __datamunge__(2851, img);
    end
    function r = conv2d (varargin)
      % conv2d  Basic (inference-only) neural-network building blocks -- see datamunge::cv::conv2d/max_pool2d/avg_pool2d/relu/sigmoid/softmax for the underlying implementation and full documentation of shapes/semantics.
      %
      %   r = datamunge.Tensor.conv2d (input, kernel, bias)
      %   r = datamunge.Tensor.conv2d (input, kernel, bias, stride)
      %   r = datamunge.Tensor.conv2d (input, kernel, bias, stride, padding)
      %
      %   Inputs:
      %     input  datamunge.Tensor
      %     kernel  datamunge.Tensor
      %     bias  datamunge.Tensor
      %     stride  integer, default 1
      %     padding  integer, default 0
      %
      %   Output:
      %     r  datamunge.Tensor or []
      %
      %   C++: Tensor * datamunge::Tensor::conv2d(const Tensor & input, const Tensor & kernel, const Tensor & bias, int stride = 1, int padding = 0)
      r = __datamunge__(2852, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::Tensor. Do not edit.
