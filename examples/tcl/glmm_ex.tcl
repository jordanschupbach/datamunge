# CMD: just TARGET=glmm_ex run-tcl
# Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
package require datamunge

# Tcl's own generator, so the simulated data (and the fit below) differ from other languages'.
expr {srand(4242)}

# Box-Muller; 1 - rand() keeps log() away from 0.
proc gauss {} {
    expr {sqrt(-2 * log(1 - rand())) * cos(2 * acos(-1) * rand())}
}

# Knuth's Poisson sampler.
proc rpois {lambda} {
    set threshold [expr {exp(-$lambda)}]
    set k 0
    set p 1.0
    while 1 {
        incr k
        set p [expr {$p * rand()}]
        if {$p <= $threshold} {
            return [expr {$k - 1}]
        }
    }
}

puts "=================== Binomial (logistic) mixed model on a real dataset (penguins) ==================="
# Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
# confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
# is not tied to location, so this is a well-behaved fit.
set penguins [datamunge::Data::penguins]
set rows {}
for {set i 0} {$i < [$penguins nrows]} {incr i} {
    if {![$penguins is_null sex $i] && ![$penguins is_null body_mass_g $i]
            && ![$penguins is_null island $i]} {
        lappend rows $i
    }
}
set sex_df [datamunge::DataFrame new]
$sex_df add_numeric_column is_male \
    [lmap i $rows {expr {[$penguins string_at sex $i] eq "male" ? 1.0 : 0.0}}]
$sex_df add_numeric_column body_mass_g [lmap i $rows {$penguins numeric_at body_mass_g $i}]
$sex_df add_string_column island [lmap i $rows {$penguins string_at island $i}]
set sex_model [datamunge::GLMM new $sex_df "is_male ~ body_mass_g + (1 | island)" binomial]
$sex_model print_summary

puts "\n=================== Poisson mixed model on simulated multi-site count data ==================="
# 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
# the baseline traffic and the promo's effectiveness vary by store.
set n_stores 25
set store_effect [lmap _ [lrepeat $n_stores 0] {expr {[gauss] * 0.4}}]
set true_intercept 2.0
set true_slope 0.3

set store {}
set promo {}
set visits {}
for {set s 0} {$s < $n_stores} {incr s} {
    set n_days [expr {15 + int(rand() * 11)}]
    for {set day 0} {$day < $n_days} {incr day} {
        set promo_intensity [expr {rand() * 3}]
        set lambda [expr {exp($true_intercept + [lindex $store_effect $s] + $true_slope * $promo_intensity)}]
        lappend store [expr {double($s)}]
        lappend promo $promo_intensity
        lappend visits [expr {double([rpois $lambda])}]
    }
}
set df [datamunge::DataFrame new]
$df add_numeric_column store $store
$df add_numeric_column promo $promo
$df add_numeric_column visits $visits
set store_model [datamunge::GLMM new $df "visits ~ promo + (1 | store)" poisson]

$store_model print_summary

puts "\nTrue generating values: intercept=$true_intercept, slope=$true_slope,\
    random-intercept SD (log scale)=0.4"

puts "\n--- BLUPs for a few stores ---"
set labels [$store_model group_labels]
for {set idx 0} {$idx < 3} {incr idx} {
    puts "store [lindex $labels $idx]: intercept shift=[lindex [$store_model random_effects_for_group $idx] 0]"
}

puts "\n--- Prediction: population-level vs. store-adjusted ---"
set population [datamunge::DataFrame new]
set store0 [datamunge::DataFrame new]
$population add_numeric_column promo {1.5}
$store0 add_numeric_column promo {1.5}
$store0 add_numeric_column store {0.0}
puts "promo=1.5, unseen store:   [lindex [$store_model predict $population] 0] expected visits (fixed effects only)"
puts "promo=1.5, store 0 (known): [lindex [$store_model predict $store0] 0] expected visits (fixed effects + store 0's BLUP)"
