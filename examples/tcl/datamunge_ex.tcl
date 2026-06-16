package require Datamunge 0.0.1

datamunge::hello

set v [datamunge::new_DVector]
datamunge::DVector_push_back $v 1.25
datamunge::DVector_push_back $v 2.5
puts "DVector size: [datamunge::DVector_size $v]"
puts [format {DVector[0]: %s} [datamunge::DVector_get $v 0]]
datamunge::delete_DVector $v

set p [datamunge::new_DPair 3.0 4.5]
puts "DPair: ([datamunge::DPair_first_get $p], [datamunge::DPair_second_get $p])"
datamunge::delete_DPair $p

puts "Datamunge Tcl example ran successfully."
