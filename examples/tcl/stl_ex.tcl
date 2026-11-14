# CMD: just TARGET=stl_ex run-tcl
# C++ std::vector and std::pair values in Tcl: vectors are lists, pairs are two-element lists.
package require datamunge

# The integers 0 .. n-1 as a list.
proc range {n} {
    set result {}
    for {set i 0} {$i < $n} {incr i} {
        lappend result $i
    }
    return $result
}

set n 100
set v [lmap i [range $n] {expr {$i * 1.5}}]
foreach x $v {
    puts $x
}
set v2 [lmap i [range $n] {expr {entier(floor($i * 1.5))}}]
foreach x $v2 {
    puts $x
}

set p {3 4}
puts "p: ([lindex $p 0], [lindex $p 1])"

set p2 {10 20}
puts "p2: ([lindex $p2 0], [lindex $p2 1])"

# A C++ std::pair<double,double> comes back as a two-element list.
set p3 [datamunge::make_dpair 1.25 2.75]
puts "make_dpair(1.25, 2.75): ([join $p3 {, }]), sum_dpair = [datamunge::sum_dpair $p3]"
