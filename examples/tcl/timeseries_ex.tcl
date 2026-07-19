package require Datamunge 0.0.1

proc randn {} {
  set u1 [expr {rand()}]
  set u2 [expr {rand()}]
  return [expr {sqrt(-2.0 * log($u1)) * cos(2.0 * 3.14159265358979323846 * $u2)}]
}

puts "=================== ARIMA(1,1,1) on a simulated random walk with drift ==================="
expr {srand(42)}
set y {}
set level 100.0
set prev_shock 0.0
for {set i 0} {$i < 150} {incr i} {
  set shock [randn]
  set level [expr {$level + 0.3 + 0.4 * $prev_shock + $shock}]
  lappend y $level
  set prev_shock $shock
}

set options [datamunge::new_ARIMAOptions]
datamunge::ARIMAOptions_p_set $options 1
datamunge::ARIMAOptions_d_set $options 1
datamunge::ARIMAOptions_q_set $options 1
datamunge::ARIMAOptions_de_population_size_set $options 80
datamunge::ARIMAOptions_de_max_generations_set $options 400
set model [datamunge::new_ARIMA $y $options]

set ar [datamunge::ARIMA_ar_coefficients $model]
set ma [datamunge::ARIMA_ma_coefficients $model]
puts "AR coefficient: [lindex $ar 0]"
puts "MA coefficient: [lindex $ma 0]"
puts "sigma^2: [datamunge::ARIMA_sigma2 $model], AIC: [datamunge::ARIMA_aic $model], BIC: [datamunge::ARIMA_bic $model]"

set pair [datamunge::ARIMA_forecast_with_intervals $model 6]
puts "6-step forecast: [join [datamunge::DVectorPair_first_get $pair] ", "]"
puts "forecast std. errors: [join [datamunge::DVectorPair_second_get $pair] ", "]"

puts "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ==================="
expr {srand(7)}
set s {}
set prev 0.0
for {set i 0} {$i < 120} {incr i} {
  set prev [expr {0.5 * $prev + [randn]}]
  lappend s [expr {20.0 + 0.2 * $i + 5.0 * sin(2.0 * 3.14159265358979323846 * $i / 12.0) + $prev}]
}

set options2 [datamunge::new_ARIMAOptions]
datamunge::ARIMAOptions_p_set $options2 1
datamunge::ARIMAOptions_seasonal_p_set $options2 1
datamunge::ARIMAOptions_seasonal_d_set $options2 1
datamunge::ARIMAOptions_seasonal_period_set $options2 12
datamunge::ARIMAOptions_de_population_size_set $options2 100
datamunge::ARIMAOptions_de_max_generations_set $options2 500
set model2 [datamunge::new_ARIMA $s $options2]
set ar2 [datamunge::ARIMA_ar_coefficients $model2]
set sar2 [datamunge::ARIMA_seasonal_ar_coefficients $model2]
puts "AR coefficient: [lindex $ar2 0], seasonal AR coefficient: [lindex $sar2 0]"
puts "12-step forecast: [join [datamunge::ARIMA_forecast $model2 12] ", "]"

puts "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ==================="
set seasonal_shape {0.8 0.75 0.9 0.95 1.0 1.05 1.1 1.05 1.0 1.1 1.3 1.6}
expr {srand(11)}
set y2 {}
for {set i 0} {$i < 48} {incr i} {
  set lvl [expr {100.0 + 2.0 * $i}]
  lappend y2 [expr {$lvl * [lindex $seasonal_shape [expr {$i % 12}]] + [randn] * 3.0}]
}

set es_options [datamunge::new_ExponentialSmoothingOptions]
datamunge::ExponentialSmoothingOptions_trend_set $es_options $::TrendType_Additive
datamunge::ExponentialSmoothingOptions_seasonal_set $es_options $::SeasonalType_Multiplicative
datamunge::ExponentialSmoothingOptions_seasonal_period_set $es_options 12
datamunge::ExponentialSmoothingOptions_de_population_size_set $es_options 60
datamunge::ExponentialSmoothingOptions_de_max_generations_set $es_options 300
set es_model [datamunge::new_ExponentialSmoothing $y2 $es_options]
puts "alpha=[datamunge::ExponentialSmoothing_alpha $es_model] beta=[datamunge::ExponentialSmoothing_beta $es_model] gamma=[datamunge::ExponentialSmoothing_gamma $es_model]"
puts "sigma^2: [datamunge::ExponentialSmoothing_sigma2 $es_model], AIC: [datamunge::ExponentialSmoothing_aic $es_model]"
puts "12-month forecast: [join [datamunge::ExponentialSmoothing_forecast $es_model 12] ", "]"

puts "\n=================== Simple exponential smoothing vs. Holt's linear trend ==================="
set flat {50.2 49.8 50.5 49.6 50.1 50.3 49.9 50.0 50.4 49.7}

set ses [datamunge::new_ExponentialSmoothing $flat [datamunge::new_ExponentialSmoothingOptions]]
puts "SES alpha= [datamunge::ExponentialSmoothing_alpha $ses]"
puts "SES 5-step forecast: [join [datamunge::ExponentialSmoothing_forecast $ses 5] ", "]"

set holt_options [datamunge::new_ExponentialSmoothingOptions]
datamunge::ExponentialSmoothingOptions_trend_set $holt_options $::TrendType_Additive
set holt [datamunge::new_ExponentialSmoothing $flat $holt_options]
puts "Holt alpha=[datamunge::ExponentialSmoothing_alpha $holt] beta=[datamunge::ExponentialSmoothing_beta $holt]"
puts "Holt 5-step forecast: [join [datamunge::ExponentialSmoothing_forecast $holt 5] ", "]"
