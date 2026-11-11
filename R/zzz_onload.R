# Native constants and class statics are filled in once the shared library is loaded.
.onLoad <- function(libname, pkgname) .rosetta_onload(asNamespace(pkgname))
