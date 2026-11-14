# CMD: just TARGET=lmm_ex run-tcl
# Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
package require datamunge

# Tcl's own generator, so the simulated data (and the fit below) differ from other languages'.
expr {srand(2024)}

# Box-Muller on (0, 1] uniforms (avoids log(0)).
proc gauss {} {
    expr {sqrt(-2 * log(1 - rand())) * cos(2 * acos(-1) * (1 - rand()))}
}

puts "=================== Random intercept on a real dataset (penguins) ==================="
set penguin_model [datamunge::LMM new [datamunge::Data::penguins] \
    "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)"]
$penguin_model print_summary

puts "\n=================== Random intercept + slope on a simulated multi-school dataset ==================="
# 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
# and the return on an extra hour of study vary by school -- the textbook case for a
# random-intercept-and-slope model.
set n_schools 30
set school_intercept [lmap _ [lrepeat $n_schools 0] {expr {[gauss] * 6}}]
set school_slope [lmap _ [lrepeat $n_schools 0] {expr {[gauss] * 1.2}}]
set true_intercept 60.0
set true_slope 3.0

set school {}
set study_hours {}
set score {}
for {set s 0} {$s < $n_schools} {incr s} {
    set n_students [expr {15 + int(rand() * 21)}]
    for {set j 0} {$j < $n_students} {incr j} {
        set hours [expr {rand() * 10}]
        lappend school [expr {double($s)}]
        lappend study_hours $hours
        lappend score [expr {$true_intercept + [lindex $school_intercept $s]
            + ($true_slope + [lindex $school_slope $s]) * $hours + [gauss] * 4}]
    }
}
set df [datamunge::DataFrame new]
$df add_numeric_column school $school
$df add_numeric_column study_hours $study_hours
$df add_numeric_column score $score

set model [datamunge::LMM new $df "score ~ study_hours + (1 + study_hours | school)"]
$model print_summary

puts "\nTrue generating values: intercept=$true_intercept, slope=$true_slope,\
    random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0"

puts "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---"
set labels [$model group_labels]
for {set idx 0} {$idx < 3} {incr idx} {
    lassign [$model random_effects_for_group $idx] intercept_shift slope_shift
    puts "school [lindex $labels $idx]: intercept shift=$intercept_shift, slope shift=$slope_shift"
}

puts "\n--- Prediction: population-level vs. school-adjusted ---"
set population [datamunge::DataFrame new]
set school0 [datamunge::DataFrame new]
$population add_numeric_column study_hours {5.0}
$school0 add_numeric_column study_hours {5.0}
$school0 add_numeric_column school {0.0}
puts "5 study hours, unseen school:      [lindex [$model predict $population] 0] (fixed effects only)"
puts "5 study hours, school 0 (known):    [lindex [$model predict $school0] 0] (fixed effects + school 0's BLUP)"
