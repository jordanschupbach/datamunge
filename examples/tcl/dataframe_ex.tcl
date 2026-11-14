# CMD: just TARGET=dataframe_ex run-tcl
# DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
package require datamunge

set sales [datamunge::DataFrame new]
$sales add_string_column region {west west east south south south}
$sales add_string_column product {widget widget widget gizmo gizmo gizmo}
$sales add_numeric_column sales {10.0 10.0 14.0 8.0 0.0 11.0} {1 1 1 1 0 1}
$sales add_string_column quarter {Q1 Q1 Q1 Q2 Q2 {}} {1 1 1 1 1 0}

puts "raw data\n[$sales to_string]\n"

set cleaned [$sales drop_duplicates {region product sales quarter}]
$cleaned fill_null_string quarter unknown
$cleaned fill_null_numeric sales 0.0
puts "after drop_duplicates + fill_null\n[$cleaned to_string]\n"

set selected [[$cleaned select {region sales quarter}] sort_by sales false]
puts "selected + sorted\n[$selected to_string]\n"

set grouped [[$cleaned group_by_sum {region} {sales}] sort_by sales false]
puts "group_by_sum(region)\n[$grouped to_string]\n"

set targets [datamunge::DataFrame new]
$targets add_string_column region {west east south}
$targets add_numeric_column target {18.0 12.0 25.0}
puts "joined with targets\n[[$grouped join $targets region region left] to_string]\n"

lassign [$cleaned shape] rows cols
puts "shape = ($rows, $cols)"
puts "sales count = [$cleaned numeric_count sales]"
puts "sales nulls = [$cleaned numeric_null_count sales]"
puts "sales sum = [$cleaned numeric_sum sales]"
puts "sales mean = [$cleaned numeric_mean sales]"
