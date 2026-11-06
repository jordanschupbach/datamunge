// Native algebra TeX serializers return owned text in Python/R. Their rendering
// stays in the canonical C++ methods; report delimiters are ordinary strings.
%include <exception.i>
%exception {
  try {
    $action
  } catch (const std::invalid_argument& error) {
    SWIG_exception_fail(SWIG_ValueError, error.what());
  } catch (const std::exception& error) {
    SWIG_exception_fail(SWIG_RuntimeError, error.what());
  }
}
%rename(to_tex) datamunge::algebra::Expr::tex_source;
%rename(to_tex) datamunge::algebra::Polynomial::tex_source;
%rename(to_tex) datamunge::algebra::RationalFunction::tex_source;
%extend datamunge::algebra::Expr {
  /// Render the expression using native grouping and precision; return raw TeX without math delimiters.
  std::string tex_source(int precision = 7) const { return $self->to_tex(precision).str(); }
}
%extend datamunge::algebra::Polynomial {
  /// Render descending powers using a raw TeX symbol and 1-17 significant digits; return raw TeX.
  std::string tex_source(const std::string& variable = "x", int precision = 7) const {
    return $self->to_tex(variable, precision).str();
  }
}
%extend datamunge::algebra::RationalFunction {
  /// Render the native reduced rational function as raw TeX, omitting a unit denominator.
  std::string tex_source(const std::string& variable = "x", int precision = 7) const {
    return $self->to_tex(variable, precision).str();
  }
}
%exception;
