{
  lib,
  buildNpmPackage,
  libxml2,
  pkg-config,
  datamunge,
  ...
}:

buildNpmPackage (finalAttrs: {

  name = "datamungejs";
  packageName = "datamungejs";
  version = "0.0.1";
  src = import ./source.nix { inherit lib; };
  # npmDepsHash = "sha256-xY8C8qEWDw+4HtFbLI2j4liIAZ6cP7JDS5dXT5N/te8=";
  npmDepsHash = "sha256-3AVJuVdQXXQ9oYoT0Zh9s0hwQMDTFsXyd90sCBTO4aw=";
  # Add native build inputs if needed
  nativeBuildInputs = [
    pkg-config
    libxml2
  ];
  buildInputs = [
    pkg-config
    libxml2
    datamunge
  ];
  # Link the nix-built core instead of recompiling it here (see setup.js).
  env.DATAMUNGE_USE_SYSTEM_LIB = "1";
  meta = {
    description = "JavaScript (N-API) bindings for the datamunge C++ library.";
    license = lib.licenses.unlicense;
  };
  production = true;
  bypassCache = true;
  reconstructLock = true;
  # buildCommands = [ "npm run build" ];
  # postPatch = "npm run build && cp -r build $out";

})
