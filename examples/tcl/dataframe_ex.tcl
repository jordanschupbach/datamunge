package require Datamunge 0.0.1

set sales [datamunge::DataFrame_empty]
datamunge::DataFrame_add_string_column $sales "region" {west west east south south south}
datamunge::DataFrame_add_string_column $sales "product" {widget widget widget gizmo gizmo gizmo}
datamunge::DataFrame_add_numeric_column $sales "sales" {10.0 10.0 14.0 8.0 0.0 11.0} {1 1 1 1 0 1}
datamunge::DataFrame_add_string_column $sales "quarter" {Q1 Q1 Q1 Q2 Q2 {}} {1 1 1 1 1 0}

puts "raw data"
puts "[datamunge::DataFrame_to_string $sales]\n"

set cleaned [datamunge::DataFrame_drop_duplicates $sales {region product sales quarter}]
datamunge::DataFrame_fill_null_string $cleaned "quarter" "unknown"
datamunge::DataFrame_fill_null_numeric $cleaned "sales" 0.0
puts "after drop_duplicates + fill_null"
puts "[datamunge::DataFrame_to_string $cleaned]\n"

set selected [datamunge::DataFrame_sort_by [datamunge::DataFrame_select $cleaned {region sales quarter}] "sales" 0]
puts "selected + sorted"
puts "[datamunge::DataFrame_to_string $selected]\n"

set grouped [datamunge::DataFrame_sort_by [datamunge::DataFrame_group_by_sum $cleaned {region} {sales}] "sales" 0]
puts "group_by_sum(region)"
puts "[datamunge::DataFrame_to_string $grouped]\n"

set targets [datamunge::DataFrame_empty]
datamunge::DataFrame_add_string_column $targets "region" {west east south}
datamunge::DataFrame_add_numeric_column $targets "target" {18.0 12.0 25.0}
set joined [datamunge::DataFrame_join $grouped $targets "region" "region" 1]
puts "joined with targets"
puts "[datamunge::DataFrame_to_string $joined]\n"

set shape [datamunge::DataFrame_shape $cleaned]
puts "shape = ([lindex $shape 0], [lindex $shape 1])"
puts "sales count = [datamunge::DataFrame_numeric_count $cleaned sales]"
puts "sales nulls = [datamunge::DataFrame_numeric_null_count $cleaned sales]"
puts "sales sum = [datamunge::DataFrame_numeric_sum $cleaned sales]"
puts "sales mean = [datamunge::DataFrame_numeric_mean $cleaned sales]"
