package require Datamunge 0.0.1

proc randn {} {
  set u1 [expr {rand()}]
  set u2 [expr {rand()}]
  return [expr {sqrt(-2.0 * log($u1)) * cos(2.0 * 3.14159265358979323846 * $u2)}]
}

puts "=================== Random intercept on a real dataset (penguins) ==================="
set penguins [datamunge::DataFrame_penguins]
set species_model [datamunge::new_LMM $penguins "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"]
datamunge::LMM_print_summary $species_model

puts "\n=================== Random intercept + slope on a simulated multi-school dataset ==================="
expr {srand(2024)}
set n_schools 30
set school_intercept {}
set school_slope {}
for {set s 0} {$s < $n_schools} {incr s} {
  lappend school_intercept [expr {[randn] * 6.0}]
  lappend school_slope [expr {[randn] * 1.2}]
}

set true_intercept 60.0
set true_slope 3.0
set school {}
set study_hours {}
set score {}
for {set s 0} {$s < $n_schools} {incr s} {
  set n_students [expr {15 + int(rand() * 21)}]
  for {set j 0} {$j < $n_students} {incr j} {
    set hours [expr {rand() * 10.0}]
    set noise [expr {[randn] * 4.0}]
    set s_val [expr {$true_intercept + [lindex $school_intercept $s] + ($true_slope + [lindex $school_slope $s]) * $hours + $noise}]
    lappend school $s
    lappend study_hours $hours
    lappend score $s_val
  }
}

set df [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $df "school" $school
datamunge::DataFrame_add_numeric_column $df "study_hours" $study_hours
datamunge::DataFrame_add_numeric_column $df "score" $score

set model [datamunge::new_LMM $df "score ~ study_hours + (1 + study_hours | school)"]
datamunge::LMM_print_summary $model

puts "\nTrue generating values: intercept=$true_intercept, slope=$true_slope, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0"

puts "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---"
set group_labels [datamunge::LMM_group_labels $model]
for {set idx 0} {$idx < 3} {incr idx} {
  set re [datamunge::LMM_random_effects_for_group $model $idx]
  puts "school [lindex $group_labels $idx]: intercept shift=[lindex $re 0], slope shift=[lindex $re 1]"
}

puts "\n--- Prediction: population-level vs. school-adjusted ---"
set newdata_population [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newdata_population "study_hours" {5.0}
set newdata_school0 [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newdata_school0 "study_hours" {5.0}
datamunge::DataFrame_add_numeric_column $newdata_school0 "school" {0.0}
set pred_pop [datamunge::LMM_predict $model $newdata_population]
set pred_s0 [datamunge::LMM_predict $model $newdata_school0]
puts "5 study hours, unseen school:      [lindex $pred_pop 0] (fixed effects only)"
puts "5 study hours, school 0 (known):    [lindex $pred_s0 0] (fixed effects + school 0's BLUP)"
