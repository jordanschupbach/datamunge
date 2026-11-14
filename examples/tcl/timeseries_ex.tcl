# CMD: just TARGET=timeseries_ex run-tcl
# Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
package require datamunge

set pi [expr {acos(-1)}]

# A standard normal draw by Box-Muller on (0, 1] uniforms (avoids log(0)); reseed with srand().
proc gauss {} {
    global pi
    expr {sqrt(-2 * log(1 - rand())) * cos(2 * $pi * (1 - rand()))}
}

# A new options object of the given datamunge class with the given fields set.
proc options {class args} {
    set options [datamunge::$class new]
    if {[llength $args]} {
        $options configure {*}$args
    }
    return $options
}

puts "=================== ARIMA(1,1,1) on a simulated random walk with drift ==================="
expr {srand(42)}
set level 100.0
set prev_shock 0.0
set y {}
for {set i 0} {$i < 150} {incr i} {
    set shock [gauss]
    set level [expr {$level + 0.3 + 0.4 * $prev_shock + $shock}]  ;# integrated ARMA(0,1) increments with drift
    set prev_shock $shock
    lappend y $level
}

set model [datamunge::ARIMA new $y [options ARIMAOptions \
    -p 1 -d 1 -q 1 -de_population_size 80 -de_max_generations 400]]

puts "AR coefficient: [lindex [$model ar_coefficients] 0]"
puts "MA coefficient: [lindex [$model ma_coefficients] 0]"
puts "sigma^2: [$model sigma2], AIC: [$model aic], BIC: [$model bic]"

lassign [$model forecast_with_intervals 6] forecast_values std_errors
puts "6-step forecast: [join $forecast_values {, }]"
puts "forecast std. errors: [join $std_errors {, }]"

puts "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ==================="
expr {srand(7)}
set prev 0.0
set s {}
for {set i 0} {$i < 120} {incr i} {
    set prev [expr {0.5 * $prev + [gauss]}]
    lappend s [expr {20 + 0.2 * $i + 5 * sin(2 * $pi * $i / 12) + $prev}]
}

set model2 [datamunge::ARIMA new $s [options ARIMAOptions \
    -p 1 -seasonal_p 1 -seasonal_d 1 -seasonal_period 12 \
    -de_population_size 100 -de_max_generations 500]]
puts "AR coefficient: [lindex [$model2 ar_coefficients] 0],\
    seasonal AR coefficient: [lindex [$model2 seasonal_ar_coefficients] 0]"
puts "12-step forecast: [join [$model2 forecast 12] {, }]"

puts "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ==================="
set seasonal_shape {0.8 0.75 0.9 0.95 1.0 1.05 1.1 1.05 1.0 1.1 1.3 1.6}
expr {srand(11)}
set y2 {}
for {set i 0} {$i < 48} {incr i} {
    lappend y2 [expr {(100 + 2 * $i) * [lindex $seasonal_shape [expr {$i % 12}]] + [gauss] * 3}]
}

set es_model [datamunge::ExponentialSmoothing new $y2 [options ExponentialSmoothingOptions \
    -trend Additive -seasonal Multiplicative -seasonal_period 12 \
    -de_population_size 60 -de_max_generations 300]]
puts "alpha=[$es_model alpha] beta=[$es_model beta] gamma=[$es_model gamma]"
puts "sigma^2: [$es_model sigma2], AIC: [$es_model aic]"
puts "12-month forecast: [join [$es_model forecast 12] {, }]"

puts "\n=================== Simple exponential smoothing vs. Holt's linear trend ==================="
set flat {50.2 49.8 50.5 49.6 50.1 50.3 49.9 50.0 50.4 49.7}

set ses [datamunge::ExponentialSmoothing new $flat [options ExponentialSmoothingOptions]]
puts "SES alpha= [$ses alpha]"
puts "SES 5-step forecast: [join [$ses forecast 5] {, }]"

set holt [datamunge::ExponentialSmoothing new $flat [options ExponentialSmoothingOptions -trend Additive]]
puts "Holt alpha=[$holt alpha] beta=[$holt beta]"
puts "Holt 5-step forecast: [join [$holt forecast 5] {, }]"
