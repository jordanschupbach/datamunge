# Tcl package index for the SWIG-generated Datamunge extension.
#
# Notes:
# - The SWIG-generated init function is `Datamunge_Init` (loaded via `load ... Datamunge`).
# - SWIG's Tcl backend provides `package provide datamunge 0.0` and installs commands
#   in the global namespace (e.g. `hello`), which doesn't match this repo's
#   intended contract (`package require Datamunge 0.0.1` and `datamunge::...`).
# - This shim normalizes the package/version and exports the full `datamunge::`
#   namespace API expected by `examples/` and `tests/` (formerly `bindings_tests/`):
#   every command the SWIG extension registers in the global namespace is aliased
#   into `datamunge::` under the same name (diffed before/after `load`, so this stays
#   correct as the bound API grows -- no per-symbol hand-maintenance needed).

package ifneeded Datamunge 0.0.1 [list apply {{dir} {
  set before [info commands]
  load [file join $dir "Datamunge[info sharedlibextension]"] Datamunge
  set after [info commands]

  namespace eval datamunge {}
  foreach src $after {
    if {$src ni $before} {
      set dst "datamunge::$src"
      if {![llength [info commands $dst]]} {
        interp alias {} $dst {} $src
      }
      # SWIG's std_vector.i renames push_back to "push" (e.g. DVector_push, not
      # DVector_push_back) -- also expose the *_push_back spelling for readability.
      if {[string match {*_push} $src]} {
        set alt "datamunge::${src}_back"
        if {![llength [info commands $alt]]} {
          interp alias {} $alt {} $src
        }
      }
    }
  }

  package provide Datamunge 0.0.1
}} $dir]
