# CMD: just TARGET=gis_ex run-ruby
# ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
# Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
# writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
# layout ShapeLayer.read() expects, then reads them back through the public API.
require "datamunge"

# Bounding box [minx, miny, maxx, maxy] of every point in rings.
def ring_bounds(rings)
  xs, ys = rings.flatten(1).transpose
  [xs.min, ys.min, xs.max, ys.max]
end

# Bytes of one Polygon record's contents (without the record header).
def polygon_record(rings)
  starts = rings.map(&:size).each_with_object([0]) { |n, acc| acc << acc.last + n }[0...-1]
  [5].pack("l<") +                                   # shape type: Polygon
    ring_bounds(rings).pack("E4") +
    [rings.size, rings.sum(&:size), *starts].pack("l<*") +
    rings.flatten.pack("E*")
end

def write_counties_shp(path, shapes)
  contents = shapes.map { |s| polygon_record(s) }
  total_words = contents.sum { |c| 4 + c.bytesize / 2 }
  f = [9994, 0, 0, 0, 0, 0, 50 + total_words].pack("l>*")
  f << [1000, 5].pack("l<*")                          # version, Polygon
  f << (ring_bounds(shapes.flatten(1)) + [0.0] * 4).pack("E8")
  contents.each_with_index { |c, i| f << [i + 1, c.bytesize / 2].pack("l>2") << c }
  File.binwrite(path, f)
end

def write_counties_dbf(path, names, populations)
  header_size = 32 + 2 * 32 + 1
  record_size = 1 + 12 + 8
  f = [0x03, 0, 0, 0, names.size, header_size, record_size].pack("C4l<s<s<") + "\0" * 20
  field = ->(name, type, length) { [name, type, 0, length].pack("a11a1l<C") + "\0" * 15 }
  f << field.("NAME", "C", 12) << field.("POP", "N", 8) << "\r"
  names.zip(populations) { |name, pop| f << " " << name.ljust(12)[0, 12] << pop.ljust(8)[0, 8] }
  File.binwrite(path, f)
end

base_path = "datamunge_gis_ex_counties_ruby"

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
plain = [[[10, 0], [10, 10], [20, 10], [20, 0], [10, 0]]]
with_lake = [
  [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]],
  [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]],
]

write_counties_shp("#{base_path}.shp", [with_lake, plain])
write_counties_dbf("#{base_path}.dbf", %w[Lakeside Plainview], %w[48231 19876])

counties = Datamunge::ShapeLayer.read(base_path)

puts "shapes = #{counties.size}, shape_type = #{counties.shape_type}"
# Roughly C's %g: integral doubles print without a fractional part.
fmt_g = ->(x) { x == x.floor ? x.to_i.to_s : format("%.6f", x) }
puts "bounds = [#{counties.bounds.map(&fmt_g).join(", ")}]\n\n"

attrs = counties.attributes
puts "attributes\n#{attrs}\n\n"

counties.size.times { |i| puts "#{attrs.string_at("NAME", i)}: #{counties.num_parts(i)} ring(s)" }

svg_path = "datamunge_gis_ex_map_ruby.svg"
counties.plot.save_svg(svg_path)
puts "\nmap saved to #{svg_path}"
