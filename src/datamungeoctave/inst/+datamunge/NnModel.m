classdef NnModel < datamunge.NativeObject
  % NnModel  A sequential network. Layers are added by spec and own parameters named "<kind><index>/..." (e.g. "dense0/W", "conv1/b"); batch-norm moving statistics live in a second store. Parameters are created on the first forward pass, when the input width is known.
  %
  %   obj = datamunge.NnModel ()
  %   obj = datamunge.NnModel (seed)
  %
  %   Methods:
  %     add_dense  Fully connected layer; activation "none", "relu", "tanh", "sigmoid", "gelu", ...
  %     add_conv2d  2-D convolution on NHWC input; padding "same" or "valid".
  %     add_max_pool2d  C++ datamunge::nn::NnModel::add_max_pool2d.
  %     add_avg_pool2d  C++ datamunge::nn::NnModel::add_avg_pool2d.
  %     add_global_avg_pool2d  C++ datamunge::nn::NnModel::add_global_avg_pool2d.
  %     add_batch_norm  Batch normalization over the last axis (momentum 0.99, eps 1e-3, TF defaults).
  %     add_layer_norm  C++ datamunge::nn::NnModel::add_layer_norm.
  %     add_dropout  Inverted dropout, active only when training.
  %     add_activation  C++ datamunge::nn::NnModel::add_activation.
  %     add_flatten  Collapses every axis but the first.
  %     add_reshape  Reshapes each example (the batch axis is kept): e.g. {28, 28, 1}.
  %     num_layers  C++ datamunge::nn::NnModel::num_layers.
  %     summary  One line per layer with its parameter shapes (after the first forward pass).
  %     forward  Runs the network. `training` enables dropout and batch statistics (and updates batch-norm moving averages).
  %     predict  forward(x, false) without recording gradients.
  %     loss  Loss of the network on (x, y): "mse", "softmax_cross_entropy" (y holds integer class labels) or "sigmoid_cross_entropy" (y in [0, 1]).
  %     compute_gradients  Computes the loss in training mode and stores the gradient of every parameter (read them with `gradient`). Returns the loss.
  %     gradient  C++ datamunge::nn::NnModel::gradient.
  %     set_optimizer  Optimizer for `train_step`: "sgd", "momentum", "adam", "adamw", "rmsprop", "adagrad", ...
  %     train_step  compute_gradients followed by one optimizer update. Returns the loss.
  %     parameter_names  C++ datamunge::nn::NnModel::parameter_names.
  %     parameter  C++ datamunge::nn::NnModel::parameter.
  %     set_parameter  C++ datamunge::nn::NnModel::set_parameter.
  %     parameter_count  C++ datamunge::nn::NnModel::parameter_count.
  %     save  Parameters and batch-norm statistics as one .npz archive ("state/..." for statistics).
  %     load  Loads an archive written by save() (or any .npz with the same names).
  %
  %   C++: class datamunge::nn::NnModel
  methods
    function obj = NnModel (varargin)
      % NnModel  Construct a C++ datamunge::nn::NnModel.
      %
      %   obj = datamunge.NnModel ()
      %   obj = datamunge.NnModel (seed)
      %
      %   Inputs:
      %     seed  integer, default 0
      %
      %   C++: datamunge::nn::NnModel::NnModel(unsigned int seed = 0)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 3161, false, varargin{:});
    end
    function add_dense (obj, varargin)
      % add_dense  Fully connected layer; activation "none", "relu", "tanh", "sigmoid", "gelu", ...
      %
      %   obj.add_dense (units)
      %   obj.add_dense (units, activation)
      %
      %   Inputs:
      %     units  integer
      %     activation  char row, default 'none'
      %
      %   C++: void datamunge::nn::NnModel::add_dense(int units, const std::string & activation = "none")
      __datamunge__(3135, obj, varargin{:});
    end
    function add_conv2d (obj, varargin)
      % add_conv2d  2-D convolution on NHWC input; padding "same" or "valid".
      %
      %   obj.add_conv2d (filters, kernel)
      %   obj.add_conv2d (filters, kernel, stride)
      %   obj.add_conv2d (filters, kernel, stride, padding)
      %   obj.add_conv2d (filters, kernel, stride, padding, activation)
      %
      %   Inputs:
      %     filters  integer
      %     kernel  integer
      %     stride  integer, default 1
      %     padding  char row, default 'same'
      %     activation  char row, default 'none'
      %
      %   C++: void datamunge::nn::NnModel::add_conv2d(int filters, int kernel, int stride = 1, const std::string & padding = "same", const std::string & activation = "none")
      __datamunge__(3136, obj, varargin{:});
    end
    function add_max_pool2d (obj, varargin)
      % add_max_pool2d  C++ datamunge::nn::NnModel::add_max_pool2d.
      %
      %   obj.add_max_pool2d (window, stride)
      %   obj.add_max_pool2d (window, stride, padding)
      %
      %   Inputs:
      %     window  integer
      %     stride  integer
      %     padding  char row, default 'valid'
      %
      %   C++: void datamunge::nn::NnModel::add_max_pool2d(int window, int stride, const std::string & padding = "valid")
      __datamunge__(3137, obj, varargin{:});
    end
    function add_avg_pool2d (obj, varargin)
      % add_avg_pool2d  C++ datamunge::nn::NnModel::add_avg_pool2d.
      %
      %   obj.add_avg_pool2d (window, stride)
      %   obj.add_avg_pool2d (window, stride, padding)
      %
      %   Inputs:
      %     window  integer
      %     stride  integer
      %     padding  char row, default 'valid'
      %
      %   C++: void datamunge::nn::NnModel::add_avg_pool2d(int window, int stride, const std::string & padding = "valid")
      __datamunge__(3138, obj, varargin{:});
    end
    function add_global_avg_pool2d (obj)
      % add_global_avg_pool2d  C++ datamunge::nn::NnModel::add_global_avg_pool2d.
      %
      %   obj.add_global_avg_pool2d ()
      %
      %   C++: void datamunge::nn::NnModel::add_global_avg_pool2d()
      __datamunge__(3139, obj);
    end
    function add_batch_norm (obj, varargin)
      % add_batch_norm  Batch normalization over the last axis (momentum 0.99, eps 1e-3, TF defaults).
      %
      %   obj.add_batch_norm ()
      %   obj.add_batch_norm (momentum)
      %   obj.add_batch_norm (momentum, eps)
      %
      %   Inputs:
      %     momentum  double, default 0.99
      %     eps  double, default 0.001
      %
      %   C++: void datamunge::nn::NnModel::add_batch_norm(double momentum = 0.98999999999999999, double eps = 0.001)
      __datamunge__(3140, obj, varargin{:});
    end
    function add_layer_norm (obj, varargin)
      % add_layer_norm  C++ datamunge::nn::NnModel::add_layer_norm.
      %
      %   obj.add_layer_norm ()
      %   obj.add_layer_norm (eps)
      %
      %   Inputs:
      %     eps  double, default 1e-05
      %
      %   C++: void datamunge::nn::NnModel::add_layer_norm(double eps = 1.0000000000000001E-5)
      __datamunge__(3141, obj, varargin{:});
    end
    function add_dropout (obj, rate)
      % add_dropout  Inverted dropout, active only when training.
      %
      %   obj.add_dropout (rate)
      %
      %   Inputs:
      %     rate  double
      %
      %   C++: void datamunge::nn::NnModel::add_dropout(double rate)
      __datamunge__(3142, obj, rate);
    end
    function add_activation (obj, activation)
      % add_activation  C++ datamunge::nn::NnModel::add_activation.
      %
      %   obj.add_activation (activation)
      %
      %   Inputs:
      %     activation  char row
      %
      %   C++: void datamunge::nn::NnModel::add_activation(const std::string & activation)
      __datamunge__(3143, obj, activation);
    end
    function add_flatten (obj)
      % add_flatten  Collapses every axis but the first.
      %
      %   obj.add_flatten ()
      %
      %   C++: void datamunge::nn::NnModel::add_flatten()
      __datamunge__(3144, obj);
    end
    function add_reshape (obj, shape)
      % add_reshape  Reshapes each example (the batch axis is kept): e.g. {28, 28, 1}.
      %
      %   obj.add_reshape (shape)
      %
      %   Inputs:
      %     shape  integer vector
      %
      %   C++: void datamunge::nn::NnModel::add_reshape(const std::vector<int> & shape)
      __datamunge__(3145, obj, shape);
    end
    function r = num_layers (obj)
      % num_layers  C++ datamunge::nn::NnModel::num_layers.
      %
      %   r = obj.num_layers ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::nn::NnModel::num_layers() const
      r = __datamunge__(3146, obj);
    end
    function r = summary (obj)
      % summary  One line per layer with its parameter shapes (after the first forward pass).
      %
      %   r = obj.summary ()
      %
      %   Output:
      %     r  char row
      %
      %   C++: std::string datamunge::nn::NnModel::summary() const
      r = __datamunge__(3147, obj);
    end
    function r = forward (obj, varargin)
      % forward  Runs the network. `training` enables dropout and batch statistics (and updates batch-norm moving averages).
      %
      %   r = obj.forward (x)
      %   r = obj.forward (x, training)
      %
      %   Inputs:
      %     x  datamunge.NnTensor
      %     training  logical, default false
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnModel::forward(const NnTensor & x, bool training = false)
      r = __datamunge__(3148, obj, varargin{:});
    end
    function r = predict (obj, x)
      % predict  forward(x, false) without recording gradients.
      %
      %   r = obj.predict (x)
      %
      %   Inputs:
      %     x  datamunge.NnTensor
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnModel::predict(const NnTensor & x)
      r = __datamunge__(3149, obj, x);
    end
    function r = loss (obj, varargin)
      % loss  Loss of the network on (x, y): "mse", "softmax_cross_entropy" (y holds integer class labels) or "sigmoid_cross_entropy" (y in [0, 1]).
      %
      %   r = obj.loss (x, y)
      %   r = obj.loss (x, y, loss)
      %   r = obj.loss (x, y, loss, training)
      %
      %   Inputs:
      %     x  datamunge.NnTensor
      %     y  datamunge.NnTensor
      %     loss  char row, default 'mse'
      %     training  logical, default false
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::nn::NnModel::loss(const NnTensor & x, const NnTensor & y, const std::string & loss = "mse", bool training = false)
      r = __datamunge__(3150, obj, varargin{:});
    end
    function r = compute_gradients (obj, varargin)
      % compute_gradients  Computes the loss in training mode and stores the gradient of every parameter (read them with `gradient`). Returns the loss.
      %
      %   r = obj.compute_gradients (x, y)
      %   r = obj.compute_gradients (x, y, loss)
      %
      %   Inputs:
      %     x  datamunge.NnTensor
      %     y  datamunge.NnTensor
      %     loss  char row, default 'mse'
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::nn::NnModel::compute_gradients(const NnTensor & x, const NnTensor & y, const std::string & loss = "mse")
      r = __datamunge__(3151, obj, varargin{:});
    end
    function r = gradient (obj, name)
      % gradient  C++ datamunge::nn::NnModel::gradient.
      %
      %   r = obj.gradient (name)
      %
      %   Inputs:
      %     name  char row
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnModel::gradient(const std::string & name) const
      r = __datamunge__(3152, obj, name);
    end
    function set_optimizer (obj, varargin)
      % set_optimizer  Optimizer for `train_step`: "sgd", "momentum", "adam", "adamw", "rmsprop", "adagrad", ...
      %
      %   obj.set_optimizer ()
      %   obj.set_optimizer (kind)
      %   obj.set_optimizer (kind, learning_rate)
      %
      %   Inputs:
      %     kind  char row, default 'adam'
      %     learning_rate  double, default 0.001
      %
      %   C++: void datamunge::nn::NnModel::set_optimizer(const std::string & kind = "adam", double learning_rate = 0.001)
      __datamunge__(3153, obj, varargin{:});
    end
    function r = train_step (obj, varargin)
      % train_step  compute_gradients followed by one optimizer update. Returns the loss.
      %
      %   r = obj.train_step (x, y)
      %   r = obj.train_step (x, y, loss)
      %
      %   Inputs:
      %     x  datamunge.NnTensor
      %     y  datamunge.NnTensor
      %     loss  char row, default 'mse'
      %
      %   Output:
      %     r  double
      %
      %   C++: double datamunge::nn::NnModel::train_step(const NnTensor & x, const NnTensor & y, const std::string & loss = "mse")
      r = __datamunge__(3154, obj, varargin{:});
    end
    function r = parameter_names (obj)
      % parameter_names  C++ datamunge::nn::NnModel::parameter_names.
      %
      %   r = obj.parameter_names ()
      %
      %   Output:
      %     r  cell array of char
      %
      %   C++: std::vector<std::string> datamunge::nn::NnModel::parameter_names() const
      r = __datamunge__(3155, obj);
    end
    function r = parameter (obj, name)
      % parameter  C++ datamunge::nn::NnModel::parameter.
      %
      %   r = obj.parameter (name)
      %
      %   Inputs:
      %     name  char row
      %
      %   Output:
      %     r  datamunge.NnTensor
      %
      %   C++: NnTensor datamunge::nn::NnModel::parameter(const std::string & name) const
      r = __datamunge__(3156, obj, name);
    end
    function set_parameter (obj, name, value_)
      % set_parameter  C++ datamunge::nn::NnModel::set_parameter.
      %
      %   obj.set_parameter (name, value_)
      %
      %   Inputs:
      %     name  char row
      %     value_  datamunge.NnTensor
      %
      %   C++: void datamunge::nn::NnModel::set_parameter(const std::string & name, const NnTensor & value)
      __datamunge__(3157, obj, name, value_);
    end
    function r = parameter_count (obj)
      % parameter_count  C++ datamunge::nn::NnModel::parameter_count.
      %
      %   r = obj.parameter_count ()
      %
      %   Output:
      %     r  integer
      %
      %   C++: int datamunge::nn::NnModel::parameter_count() const
      r = __datamunge__(3158, obj);
    end
    function save (obj, path)
      % save  Parameters and batch-norm statistics as one .npz archive ("state/..." for statistics).
      %
      %   obj.save (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   C++: void datamunge::nn::NnModel::save(const std::string & path) const
      __datamunge__(3159, obj, path);
    end
    function load (obj, path)
      % load  Loads an archive written by save() (or any .npz with the same names).
      %
      %   obj.load (path)
      %
      %   Inputs:
      %     path  char row
      %
      %   C++: void datamunge::nn::NnModel::load(const std::string & path)
      __datamunge__(3160, obj, path);
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::nn::NnModel. Do not edit.
