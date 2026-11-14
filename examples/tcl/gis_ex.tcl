# CMD: just TARGET=gis_ex run-tcl
# ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
package require datamunge

# Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
# writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
# layout ShapeLayer read expects, then reads them back through the public API.

# All points of a list of rings, flattened to {x y x y ...}.
proc ring_coords {rings} {
    concat {*}[concat {*}$rings]
}

# Bounding box {minx miny maxx maxy} of every point in rings.
proc ring_bounds {rings} {
    set xs {}
    set ys {}
    foreach {x y} [ring_coords $rings] {
        lappend xs $x
        lappend ys $y
    }
    list [tcl::mathfunc::min {*}$xs] [tcl::mathfunc::min {*}$ys] \
        [tcl::mathfunc::max {*}$xs] [tcl::mathfunc::max {*}$ys]
}

# Bytes of one Polygon record's contents (without the record header).
proc polygon_record {rings} {
    set starts {}
    set total 0
    foreach ring $rings {
        lappend starts $total
        incr total [llength $ring]
    }
    set coords [ring_coords $rings]
    # Shape type 5 = Polygon, then bounds, part/point counts, part starts and the points.
    binary format i1q4i2i*q* 5 [ring_bounds $rings] [list [llength $rings] $total] $starts $coords
}

proc write_counties_shp {path shapes} {
    set contents [lmap shape $shapes {polygon_record $shape}]
    set total_words 0
    foreach c $contents {
        incr total_words [expr {4 + [string length $c] / 2}]
    }
    set f [binary format I7 [list 9994 0 0 0 0 0 [expr {50 + $total_words}]]]
    append f [binary format i2 {1000 5}]                ;# version, Polygon
    append f [binary format q8 [concat [ring_bounds [concat {*}$shapes]] {0.0 0.0 0.0 0.0}]]
    set i 0
    foreach c $contents {
        append f [binary format I2 [list [incr i] [expr {[string length $c] / 2}]]] $c
    }
    set out [open $path wb]
    puts -nonewline $out $f
    close $out
}

# One 32-byte dBASE field descriptor.
proc dbf_field {name type length} {
    binary format a11a1ic1x15 $name $type 0 $length
}

proc write_counties_dbf {path names populations} {
    set header_size [expr {32 + 2 * 32 + 1}]
    set record_size [expr {1 + 12 + 8}]
    set f [binary format c4iss [list 3 0 0 0] [llength $names] $header_size $record_size]
    append f [binary format x20]
    append f [dbf_field NAME C 12] [dbf_field POP N 8] "\r"
    foreach name $names pop $populations {
        append f " " [format %-12.12s%-8.8s $name $pop]
    }
    set out [open $path wb]
    puts -nonewline $out $f
    close $out
}

# Roughly C's %g: integral doubles print without a fractional part.
proc fmt_g {x} {
    expr {$x == floor($x) ? [format %d [expr {entier($x)}]] : [format %.6f $x]}
}

set base_path datamunge_gis_ex_counties_tcl

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
set plain {{{10 0} {10 10} {20 10} {20 0} {10 0}}}
set with_lake {
    {{0 0} {0 10} {10 10} {10 0} {0 0}}
    {{3 3} {4 3} {4 4} {3 4} {3 3}}
}

write_counties_shp $base_path.shp [list $with_lake $plain]
write_counties_dbf $base_path.dbf {Lakeside Plainview} {48231 19876}

set counties [datamunge::ShapeLayer read $base_path]

puts "shapes = [$counties size], shape_type = [$counties shape_type]"
puts "bounds = \[[join [lmap x [$counties bounds] {fmt_g $x}] {, }]\]\n"

set attrs [$counties attributes]
puts "attributes\n[$attrs to_string]\n"

for {set i 0} {$i < [$counties size]} {incr i} {
    puts "[$attrs string_at NAME $i]: [$counties num_parts $i] ring(s)"
}

set svg_path datamunge_gis_ex_map_tcl.svg
[$counties plot] save_svg $svg_path
puts "\nmap saved to $svg_path"
