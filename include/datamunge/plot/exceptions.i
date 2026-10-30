// Translate failures from the constructor-based plotting workflow at the native
// language boundary. Keep this shared by Python/R so invalid input cannot let a
// C++ exception escape into either interpreter.
%include <exception.i>

%define DATAMUNGE_PLOT_EXCEPTIONS(Symbol)
%exception Symbol {
  try {
    $action
  } catch (const std::invalid_argument& error) {
    SWIG_exception_fail(SWIG_ValueError, error.what());
  } catch (const std::exception& error) {
    SWIG_exception_fail(SWIG_RuntimeError, error.what());
  }
}
%enddef

DATAMUNGE_PLOT_EXCEPTIONS(datamunge::graphics::RPlot::RPlot)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::graphics::RPlot::hist)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::graphics::RPlot::set_col)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::runif)

// Dataset loading: unknown names, missing/malformed files and failed downloads.
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::DataFrame::diamonds)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::DataFrame::california_housing)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::DataFrame::dataset)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::DataFrame::read_csv)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::DataFrame::parse_csv)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::CsvReader::CsvReader)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::CsvReader::next)
DATAMUNGE_PLOT_EXCEPTIONS(datamunge::CsvReader::set_batch_size)
