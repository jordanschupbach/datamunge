\name{nn_examples}
\alias{nn_example_names}
\alias{nn_example_run}
\title{Run canonical neural network experiments}
\description{Optional SWIG entry points for the canonical C++ CNN and autoencoder experiments.}
\usage{
nn_example_names()
nn_example_run(name, arguments = character())
}
\arguments{
\item{name}{One name from \code{nn_example_names()}.}
\item{arguments}{Character vector of original experiment options, including
\code{key=value}, \code{--quick}, or \code{--full}.}
}
\value{The catalogue as a character vector, or the experiment's integer exit status.}
\details{Experiments execute in process. Quick mode is the default. Dataset and
output paths respect DATAMUNGE_DATA_DIR and DATAMUNGE_OUTPUT_DIR. R outputs
otherwise use output/nn/r/quick or output/nn/r/full. Environment changes are
restored after each call. Full mode can require substantial training time.}
