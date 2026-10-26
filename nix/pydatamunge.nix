{
  lib,
  stdenv,
  python,
  fetchPypi,
  setuptools,
  buildPythonPackage,
  libxml2,
  pkg-config,
  datamunge,
  swigPkg,
}:
buildPythonPackage rec {
  pname = "pydatamunge";
  version = "0.0.1";
  pyproject = true;
  src = import ./source.nix { inherit lib; };
  build-system = [ setuptools ];
  meta = {
    description = "Python bindings to the datamunge library.";
    homepage = "https://github.com/jordanschupbach/datamunge";
    license = lib.licenses.unlicense;
  };
  buildInputs = [
    pkg-config
    datamunge
  ];
  # Link the nix-built core instead of recompiling it here (see setup.py).
  env.DATAMUNGE_USE_SYSTEM_LIB = "1";
  preBuild = "python scripts/prebuild_bindings.py python";
  nativeBuildInputs = [
    swigPkg
    pkg-config
  ];
}
