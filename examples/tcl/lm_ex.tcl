package require Datamunge 0.0.1

set hp {110.0 110.0 93.0 110.0 175.0 105.0 245.0 62.0 95.0 123.0}
set wt {2.62 2.875 2.32 3.215 3.44 3.46 3.57 3.19 3.15 3.44}
set transmission {manual manual manual automatic automatic automatic automatic automatic automatic automatic}
set mpg {21.0 21.0 22.8 21.4 18.7 18.1 14.3 24.4 22.8 19.2}

set cars [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $cars "hp" $hp
datamunge::DataFrame_add_numeric_column $cars "wt" $wt
datamunge::DataFrame_add_string_column $cars "transmission" $transmission
datamunge::DataFrame_add_numeric_column $cars "mpg" $mpg

puts "Fitting: mpg ~ hp + wt + transmission\n"
set model [datamunge::new_LM $cars "mpg ~ hp + wt + transmission"]
datamunge::LM_print_summary $model

puts "\nSequential ANOVA:"
puts "[datamunge::DataFrame_to_string [datamunge::LM_anova $model]]"

set newcars [datamunge::DataFrame_empty]
datamunge::DataFrame_add_numeric_column $newcars "hp" {150.0 90.0}
datamunge::DataFrame_add_numeric_column $newcars "wt" {3.0 2.5}
datamunge::DataFrame_add_string_column $newcars "transmission" {manual automatic}

set frame [datamunge::LM_predict_frame $model $newcars "confidence"]
puts "\nPredictions with 95% confidence intervals:"
puts "[datamunge::DataFrame_to_string $frame]"

datamunge::LM_save_diagnostic_plots $model "lm_ex_diagnostics"
puts "\nSaved diagnostic plots as lm_ex_diagnostics_*.svg"
