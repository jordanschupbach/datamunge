# Tcl package index for the SWIG-generated Octra extension.
#
# Notes:
# - The SWIG-generated init function is `Octra_Init` (loaded via `load ... Octra`).
# - SWIG's Tcl backend provides `package provide octra 0.0` and installs commands
#   in the global namespace (e.g. `hello`), which doesn't match this repo's
#   intended contract (`package require Octra 0.0.1` and `octra::...`).
# - This shim normalizes the package/version and exports a minimal `octra::`
#   namespace API expected by `examples/` and `tests/` (formerly `bindings_tests/`).

package ifneeded Octra 0.0.1 [list apply {{dir} {
  load [file join $dir "Octra[info sharedlibextension]"] Octra

  namespace eval octra {}
  foreach {src dst} {
    hello octra::hello
    new_DVector octra::new_DVector
    DVector_size octra::DVector_size
    DVector_get octra::DVector_get
    DVector_set octra::DVector_set
    DVector_push octra::DVector_push
    DVector_pop octra::DVector_pop
    DVector_push octra::DVector_push_back
    delete_DVector octra::delete_DVector
    new_DPair octra::new_DPair
    DPair_first_get octra::DPair_first_get
    DPair_second_get octra::DPair_second_get
    DPair_first_set octra::DPair_first_set
    DPair_second_set octra::DPair_second_set
    delete_DPair octra::delete_DPair
  } {
    if {[llength [info commands $src]] && ![llength [info commands $dst]]} {
      interp alias {} $dst {} $src
    }
  }

  package provide Octra 0.0.1
}} $dir]
