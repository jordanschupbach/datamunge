package require Octra 0.0.1

octra::hello

set v [octra::new_DVector]
octra::DVector_push_back $v 1.25
octra::DVector_push_back $v 2.5
puts "DVector size: [octra::DVector_size $v]"
puts [format {DVector[0]: %s} [octra::DVector_get $v 0]]
octra::delete_DVector $v

set p [octra::new_DPair 3.0 4.5]
puts "DPair: ([octra::DPair_first_get $p], [octra::DPair_second_get $p])"
octra::delete_DPair $p

puts "Octra Tcl example ran successfully."
