package require Datamunge 0.0.1

proc randn {} {
  set u1 [expr {rand()}]
  set u2 [expr {rand()}]
  return [expr {sqrt(-2.0 * log($u1)) * cos(2.0 * 3.14159265358979323846 * $u2)}]
}

puts "=================== Binomial (logistic) mixed model on a real dataset (penguins) ==================="
set penguins [datamunge::DataFrame_penguins]
set is_male {}
set body_mass {}
set island {}
set n [datamunge::DataFrame_nrows $penguins]
for {set i 0} {$i < $n} {incr i} {
  if {![datamunge::DataFrame_is_null $penguins "sex" $i] &&
      ![datamunge::DataFrame_is_null $penguins "body_mass_g" $i] &&
      ![datamunge::DataFrame_is_null $penguins "island" $i]} {
    lappend is_male [expr {[datamunge::DataFrame_string_at $penguins "sex" $i] eq "male" ? 1.0 : 0.0}]
    lappend body_mass [datamunge::DataFrame_numeric_at $penguins "body_mass_g" $i]
    lappend island [datamunge::DataFrame_string_at $penguins "island" $i]
  }
}

set sex_df [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $sex_df "is_male" $is_male
datamunge::DataFrame_add_numeric_column $sex_df "body_mass_g" $body_mass
datamunge::DataFrame_add_string_column $sex_df "island" $island

set sex_model [datamunge::new_GLMM $sex_df "is_male ~ body_mass_g + (1 | island)" "binomial"]
datamunge::GLMM_print_summary $sex_model

puts "\n=================== Poisson mixed model on simulated multi-site count data ==================="
expr {srand(4242)}
set n_stores 25
set store_effect {}
for {set s 0} {$s < $n_stores} {incr s} {
  lappend store_effect [expr {[randn] * 0.4}]
}

set true_intercept 2.0
set true_slope 0.3
set store {}
set promo {}
set visits {}
for {set s 0} {$s < $n_stores} {incr s} {
  set n_days [expr {15 + int(rand() * 11)}]
  for {set d 0} {$d < $n_days} {incr d} {
    set promo_intensity [expr {rand() * 3.0}]
    set lam [expr {exp($true_intercept + [lindex $store_effect $s] + $true_slope * $promo_intensity)}]
    set l_thresh [expr {exp(-$lam)}]
    set k 0
    set p 1.0
    while {1} {
      incr k
      set p [expr {$p * rand()}]
      if {$p <= $l_thresh} { break }
    }
    lappend store $s
    lappend promo $promo_intensity
    lappend visits [expr {$k - 1}]
  }
}

set df [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $df "store" $store
datamunge::DataFrame_add_numeric_column $df "promo" $promo
datamunge::DataFrame_add_numeric_column $df "visits" $visits

set store_model [datamunge::new_GLMM $df "visits ~ promo + (1 | store)" "poisson"]
datamunge::GLMM_print_summary $store_model

puts "\nTrue generating values: intercept=$true_intercept, slope=$true_slope, random-intercept SD (log scale)=0.4"

puts "\n--- BLUPs for a few stores ---"
set group_labels [datamunge::GLMM_group_labels $store_model]
for {set idx 0} {$idx < 3} {incr idx} {
  set re [datamunge::GLMM_random_effects_for_group $store_model $idx]
  puts "store [lindex $group_labels $idx]: intercept shift=[lindex $re 0]"
}

puts "\n--- Prediction: population-level vs. store-adjusted ---"
set newdata_population [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newdata_population "promo" {1.5}
set newdata_store0 [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newdata_store0 "promo" {1.5}
datamunge::DataFrame_add_numeric_column $newdata_store0 "store" {0.0}
set pred_pop [datamunge::GLMM_predict $store_model $newdata_population]
set pred_s0 [datamunge::GLMM_predict $store_model $newdata_store0]
puts "promo=1.5, unseen store:   [lindex $pred_pop 0] expected visits (fixed effects only)"
puts "promo=1.5, store 0 (known): [lindex $pred_s0 0] expected visits (fixed effects + store 0's BLUP)"
