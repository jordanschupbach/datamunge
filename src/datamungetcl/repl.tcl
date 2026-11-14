# The datamunge Tcl console (just repl-tcl): tclsh with `package require datamunge` done. Started as
# a script, tclsh has no interactive loop of its own, so this reads commands from stdin until EOF.
package require datamunge
puts "datamunge [package present datamunge] loaded (Tcl [info patchlevel]); try:"
puts {  set df [datamunge::DataFrame new]; $df add_numeric_column x [datamunge::seql 0 1 5]; puts [$df to_string]}
if {[info exists argv0] && [file normalize $argv0] eq [file normalize [info script]]} {
    set tcl_interactive 1
    set command ""
    while {1} {
        puts -nonewline [expr {$command eq "" ? "% " : "> "}]
        flush stdout
        if {[gets stdin line] < 0} break
        append command $line\n
        if {![info complete $command]} continue
        if {[catch {uplevel #0 $command} result]} {
            puts stderr $result
        } elseif {$result ne ""} {
            puts $result
        }
        set command ""
    }
}
