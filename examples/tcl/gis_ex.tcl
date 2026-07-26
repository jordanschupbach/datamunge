package require Datamunge 0.0.1

# Demonstrates datamunge::ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
# drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
# repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one plain
# square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read() expects,
# then reads it back through the public API. Tcl's [binary format] mirrors Python's struct.pack:
#   i = 32-bit little-endian int   I = 32-bit big-endian int
#   s = 16-bit little-endian int   q = 64-bit little-endian (IEEE) double
#   a = null-padded char string    A = space-padded char string   x = null pad byte

proc lmin {lst} {
  set m [lindex $lst 0]
  foreach x $lst { if {$x < $m} { set m $x } }
  return $m
}
proc lmax {lst} {
  set m [lindex $lst 0]
  foreach x $lst { if {$x > $m} { set m $x } }
  return $m
}

proc polygon_record {rings} {
  set all_points {}
  foreach ring $rings {
    foreach p $ring { lappend all_points $p }
  }
  set xs {}
  set ys {}
  foreach p $all_points {
    lappend xs [lindex $p 0]
    lappend ys [lindex $p 1]
  }
  set out [binary format i 5]  ;# shape type: Polygon
  append out [binary format q4 [list [lmin $xs] [lmin $ys] [lmax $xs] [lmax $ys]]]
  append out [binary format i [llength $rings]]
  append out [binary format i [llength $all_points]]
  set start 0
  foreach ring $rings {
    append out [binary format i $start]
    incr start [llength $ring]
  }
  foreach ring $rings {
    foreach p $ring {
      append out [binary format q2 [list [lindex $p 0] [lindex $p 1]]]
    }
  }
  return $out
}

proc write_counties_shp {path shapes} {
  set contents {}
  foreach rings $shapes { lappend contents [polygon_record $rings] }
  set total_words 0
  foreach c $contents { incr total_words [expr {4 + [string length $c] / 2}] }
  set xs {}
  set ys {}
  foreach rings $shapes {
    foreach ring $rings {
      foreach p $ring {
        lappend xs [lindex $p 0]
        lappend ys [lindex $p 1]
      }
    }
  }
  set f [open $path w]
  fconfigure $f -translation binary -encoding binary
  puts -nonewline $f [binary format I 9994]
  puts -nonewline $f [binary format I5 {0 0 0 0 0}]
  puts -nonewline $f [binary format I [expr {50 + $total_words}]]
  puts -nonewline $f [binary format i 1000]
  puts -nonewline $f [binary format i 5]  ;# Polygon
  puts -nonewline $f [binary format q4 [list [lmin $xs] [lmin $ys] [lmax $xs] [lmax $ys]]]
  puts -nonewline $f [binary format q4 {0.0 0.0 0.0 0.0}]
  set i 1
  foreach content $contents {
    puts -nonewline $f [binary format I $i]
    puts -nonewline $f [binary format I [expr {[string length $content] / 2}]]
    puts -nonewline $f $content
    incr i
  }
  close $f
}

proc write_counties_dbf {path names populations} {
  set header_size [expr {32 + 2 * 32 + 1}]
  set record_size [expr {1 + 12 + 8}]
  set f [open $path w]
  fconfigure $f -translation binary -encoding binary
  puts -nonewline $f [binary format c4 {3 0 0 0}]
  puts -nonewline $f [binary format i [llength $names]]
  puts -nonewline $f [binary format s $header_size]
  puts -nonewline $f [binary format s $record_size]
  puts -nonewline $f [binary format x20]

  # write_field(name, field_type, length): 11-byte null-padded name, 1-byte type, 4 null bytes,
  # 1-byte length, 15 null bytes.
  proc write_field {f name field_type length} {
    puts -nonewline $f [binary format a11 $name]
    puts -nonewline $f [binary format a1 $field_type]
    puts -nonewline $f [binary format x4]
    puts -nonewline $f [binary format c $length]
    puts -nonewline $f [binary format x15]
  }

  write_field $f "NAME" "C" 12
  write_field $f "POP" "N" 8
  puts -nonewline $f [binary format c 13]  ;# 0x0D field-descriptor terminator

  foreach name $names pop $populations {
    puts -nonewline $f [binary format a1 " "]                          ;# record not-deleted flag
    puts -nonewline $f [binary format A12 [string range $name 0 11]]   ;# space-padded NAME
    puts -nonewline $f [binary format A8 [string range $pop 0 7]]      ;# space-padded POP
  }
  close $f
}

set base "datamunge_gis_ex_counties_tcl"

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
set plain {{{10 0} {10 10} {20 10} {20 0} {10 0}}}
set with_lake {
  {{0 0} {0 10} {10 10} {10 0} {0 0}}
  {{3 3} {4 3} {4 4} {3 4} {3 3}}
}

write_counties_shp "$base.shp" [list $with_lake $plain]
write_counties_dbf "$base.dbf" {Lakeside Plainview} {48231 19876}

set counties [datamunge::ShapeLayer_read $base]

puts "shapes = [datamunge::ShapeLayer_size $counties], shape_type = [datamunge::ShapeLayer_shape_type $counties]"
# bounds() returns std::vector<double>, delivered as a plain Tcl list by the binding.
set bounds [datamunge::ShapeLayer_bounds $counties]
set bt {}
foreach b $bounds { lappend bt [format %g $b] }
puts "bounds = \[[join $bt ", "]\]"
puts ""

set attributes [datamunge::ShapeLayer_attributes $counties]
puts "attributes"
puts [datamunge::DataFrame_to_string $attributes]
puts ""

for {set i 0} {$i < [datamunge::ShapeLayer_size $counties]} {incr i} {
  set name [datamunge::DataFrame_string_at $attributes "NAME" $i]
  puts "$name: [datamunge::ShapeLayer_num_parts $counties $i] ring(s)"
}

set svg_path "datamunge_gis_ex_map_tcl.svg"
datamunge::Plot_save_svg [datamunge::ShapeLayer_plot $counties] $svg_path
puts "\nmap saved to $svg_path"
