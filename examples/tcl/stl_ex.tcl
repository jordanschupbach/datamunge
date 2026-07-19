package require Datamunge 0.0.1

set n 100
set v [datamunge::new_DVector]
for {set i 0} {$i < $n} {incr i} {
  datamunge::DVector_push $v [expr {$i * 1.5}]
}
for {set i 0} {$i < $n} {incr i} {
  puts [datamunge::DVector_get $v $i]
}

set v2 [datamunge::new_IVector]
for {set i 0} {$i < $n} {incr i} {
  datamunge::IVector_push $v2 [expr {int($i * 1.5)}]
}
for {set i 0} {$i < $n} {incr i} {
  puts [datamunge::IVector_get $v2 $i]
}

set p [datamunge::new_IPair 3 4]
puts "p: ([datamunge::IPair_first_get $p], [datamunge::IPair_second_get $p])"

set p2 [datamunge::new_DPair 10.0 20.0]
puts "p2: ([datamunge::DPair_first_get $p2], [datamunge::DPair_second_get $p2])"
