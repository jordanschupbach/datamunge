\name{nn}
\alias{nn}
\title{Functional C++ neural network API}
\description{SWIG bindings to datamunge::nn. R code composes models and owns
training and evaluation loops; C++ implements tensor operations and differentiation.}
\format{An environment containing tensor factories, named ParameterStore,
functional layers, gradients, Optimizer, BatchOrder, datasets and tensor I/O.}
\details{Use \code{nn$ParameterStore(seed)}, \code{nn$dense}, \code{nn$conv2d},
\code{nn$grad} and \code{nn$Optimizer} in R model and training code.
Enums are grouped under \code{nn$DType}, \code{nn$Activation},
\code{nn$Padding} and \code{nn$OptimizerKind}. Initializers are under \code{nn$init}.
\code{nn$no_grad(function() ...)} restores gradient mode even on errors.
Shapes and tensor row indices follow the zero-based C++ API.
\code{nn$load_npz} and \code{nn$save_npz} use named lists of tensors.
\code{nn$tensor_list} and \code{nn$tensor_vector} convert native tensor vectors.
R's reserved word next is exposed as \code{batch$`_next`()} on BatchOrder.
Tensor references retain their owning containers through garbage collection.}
