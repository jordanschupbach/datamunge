{
  lib,
  stdenv,
  python,
  fetchPypi,
  setuptools,
  buildPythonPackage,
  libxml2,
  pkg-config,
}:
buildPythonPackage rec {
  pname = "pyoctra";
  version = "0.0.1";
  pyproject = true;
  src = lib.cleanSource ../.;
  build-system = [ setuptools ];
  meta = {
    description = "Python bindings to the octra library.";
    homepage = "https://github.com/jordanschupbach/octra";
    license = lib.licenses.unlicense;
  };
  buildInputs = [
    pkg-config
  ];
  nativeBuildInputs = [
    pkg-config
  ];
}
