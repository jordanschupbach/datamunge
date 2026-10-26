\name{nn_examples}
\alias{nn_example_names}
\alias{nn_example_run}
\title{Run native R neural network examples}
\description{List and execute R example scripts through the functional SWIG API.
Models and training loops are implemented in the scripts, not compiled experiments.}
\usage{
nn_example_names()
nn_example_run(name, arguments = character())
}
\arguments{
\item{name}{One name returned by \code{nn_example_names()}.}
\item{arguments}{Character vector of original example options, including
\code{key=value}, \code{--quick}, and \code{--full}.}
}
\value{A character vector of implemented example names, or an integer exit status.}
\details{Scripts are loaded from the current checkout when available, otherwise
from the installed package. DATAMUNGE_EXAMPLES_ROOT can select a checkout.
Dataset and output directories respect DATAMUNGE_DATA_DIR and DATAMUNGE_OUTPUT_DIR.
Temporary environment settings are restored even on errors. Unported C++ examples
are not included in the native R catalogue. No Python runner is involved.}
