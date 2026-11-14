# CMD: just TARGET=datamunge_ex run-tcl
# Smoke test of the basic bindings: vectors, pairs and a native Callback object.
package require datamunge

datamunge::hello

set v [datamunge::make_dvector 1.0 2.0 3.0]
puts "sum_dvector:\t[datamunge::sum_dvector $v]"

set p [datamunge::make_dpair 1.25 2.75]
puts "sum_dpair:\t[datamunge::sum_dpair $p]"

set cb [datamunge::Callback new]
puts "call_with_callback(3.0):\t[datamunge::call_with_callback 3.0 $cb]"
set v2 [datamunge::map_dvector_with_callback [datamunge::make_dvector 1.0 2.0 3.0] $cb]
puts "sum_dvector(map_dvector_with_callback(1,2,3)):\t[datamunge::sum_dvector $v2]"
$cb destroy
