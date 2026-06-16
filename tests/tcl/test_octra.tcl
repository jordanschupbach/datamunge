package require Tcl 8.6

if {[catch {package require Octra 0.0.1} err]} {
  puts stderr "FAILED: package require Octra 0.0.1: $err"
  exit 1
}

if {[catch {octra::hello} err]} {
  puts stderr "FAILED: octra::hello: $err"
  exit 1
}

if {[catch {
  set v [octra::new_DVector]
  octra::DVector_push_back $v 1.0
  octra::DVector_push_back $v 2.5
  if {[octra::DVector_size $v] != 2} { error "DVector_size != 2" }
  if {[octra::DVector_get $v 0] != 1.0} { error "DVector_get(0) != 1.0" }
  if {[octra::DVector_get $v 1] != 2.5} { error "DVector_get(1) != 2.5" }
  octra::delete_DVector $v
} err]} {
  puts stderr "FAILED: std::vector<double>: $err"
  exit 1
}

if {[catch {
  set p [octra::new_DPair 3.0 4.0]
  if {[octra::DPair_first_get $p] != 3.0} { error "DPair_first_get != 3.0" }
  if {[octra::DPair_second_get $p] != 4.0} { error "DPair_second_get != 4.0" }
  octra::DPair_first_set $p 9.5
  octra::DPair_second_set $p 10.25
  if {[octra::DPair_first_get $p] != 9.5} { error "DPair_first_get != 9.5" }
  if {[octra::DPair_second_get $p] != 10.25} { error "DPair_second_get != 10.25" }
  octra::delete_DPair $p
} err]} {
  puts stderr "FAILED: std::pair<double,double>: $err"
  exit 1
}

puts "ok"
