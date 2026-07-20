{
  lib,
  stdenv,
  python,
  fetchPypi,
  setuptools,
  buildPythonPackage,
  libxml2,
  pkg-config,
  zlib,
}:
buildPythonPackage rec {
  pname = "pydatamunge";
  version = "0.0.1";
  pyproject = true;
  src = lib.cleanSource ../.;
  build-system = [ setuptools ];
  meta = {
    description = "Python bindings to the datamunge library.";
    homepage = "https://github.com/jordanschupbach/datamunge";
    license = lib.licenses.unlicense;
  };
  buildInputs = [
    pkg-config
    zlib
  ];
  nativeBuildInputs = [
    pkg-config
  ];
}
