# Smoke tests of the datamunge Tcl binding (just test-tcl). Argument: the generated package file
# (default src/datamungetcl/pkg/datamunge/datamunge.tcl), for the documentation checks.
package require Tcl 8.6-
package require datamunge

set checks 0
# check CONDITION MESSAGE: CONDITION is an expr evaluated in the caller.
proc check {condition message} {
    incr ::checks
    if {![uplevel 1 [list expr $condition]]} {
        puts stderr "check failed: $message"
        exit 1
    }
}

# A Tcl subclass overriding the C++ Callback's virtual call() (a director).
oo::class create Doubler {
    superclass ::datamunge::Callback
    constructor {} { next }
    method call {x} { expr {2 * $x} }
}

# An override that raises a Tcl error with its own -errorcode.
oo::class create Failing {
    superclass ::datamunge::Callback
    constructor {} { next }
    method call {x} { return -code error -errorcode {TEST from_tcl} "from Tcl" }
}

# Smoke test: call into the native library.
datamunge::hello

# std::vector<double> and std::pair<double, double> round trips.
set v [datamunge::make_dvector 1.0 2.0 3.0]
check {$v eq {1.0 2.0 3.0}} "std::vector<double> is a Tcl list of doubles"
check {[datamunge::sum_dvector $v] == 6.0} "sum_dvector"
check {[datamunge::sum_dvector {0.5 1.5}] == 2.0} "lists convert to std::vector"
set p [datamunge::make_dpair 1.25 2.75]
check {$p eq {1.25 2.75}} "std::pair is a two-element list"
check {[datamunge::sum_dpair $p] == 4.0} "sum_dpair"

# The C++ Callback class, a Tcl subclass overriding its virtual call(), and a command prefix.
set cb [datamunge::Callback new]
check {[datamunge::call_with_callback 3.0 $cb] == 3.0} "native Callback"
check {[datamunge::sum_dvector [datamunge::map_dvector_with_callback {1.0 2.0 3.0} $cb]] == 6.0} "native map"
$cb destroy
set doubler [Doubler new]
check {[datamunge::call_with_callback 3.0 $doubler] == 6.0} "the Tcl override runs"
check {[datamunge::sum_dvector [datamunge::map_dvector_with_callback {1.0 2.0 3.0} $doubler]] == 12.0} \
    "override from C++ loops"

# A Tcl error inside an override reaches the caller unchanged.
set failing [Failing new]
check {[catch {datamunge::call_with_callback 1.0 $failing} msg opts] && [dict get $opts -errorcode] eq {TEST from_tcl}} \
    "the Tcl error propagates"

# C++ exceptions raise Tcl errors with -errorcode {DATAMUNGE kind cpp_type message}.
set code [catch {datamunge::runif 1.0 0.0 3} msg opts]
check {$code == 1 && [lrange [dict get $opts -errorcode] 0 1] eq {DATAMUNGE invalid_argument}} \
    "C++ exceptions raise DATAMUNGE errors"

# Classes and methods.
set frame [datamunge::DataFrame new]
$frame add_numeric_column x {1.0 2.0 3.0}
check {[$frame nrows] == 3} "DataFrame"
check {[$frame columns] eq {x}} "std::vector<std::string> is a Tcl list"
$frame destroy
check {[info commands $frame] eq ""} "destroy deletes the object command"

# Commented stubs in the package file carry the C++ documentation.
set path [expr {$argc > 0 ? [lindex $argv 0]
                : [file join [file dirname [info script]] .. .. src datamungetcl pkg datamunge datamunge.tcl]}]
set f [open $path r]
fconfigure $f -encoding utf-8
set source [read $f]
close $f
check {[regexp {((?:\n *#[^\n]*)+)\n *proc seql \{from to n\}} $source -> doc] && [string match {*evenly spaced*} $doc]} \
    "proc documentation"
check {[regexp {((?:\n *#[^\n]*)+)\n *method sort_by } $source -> mdoc] && [string match {*usage: $obj sort_by*} $mdoc]} \
    "method documentation"

puts "test_datamunge.tcl: all $checks checks passed"
