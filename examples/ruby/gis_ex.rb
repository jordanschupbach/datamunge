require "octruby"
require "tmpdir"

# Demonstrates Datamunge::ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
# drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
# repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one
# plain square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read()
# expects, then reads it back through the public API. Ruby's Array#pack mirrors Python's
# struct.pack for the binary layout ("l<" = <i, "E" = <d little-endian double, "l>" = >i, etc.).

def polygon_record(rings)
  all_points = rings.flatten(1)
  xs = all_points.map { |p| p[0] }
  ys = all_points.map { |p| p[1] }
  out = [5].pack("l<") # shape type: Polygon
  out += [xs.min, ys.min, xs.max, ys.max].pack("E4")
  out += [rings.length].pack("l<")
  out += [all_points.length].pack("l<")
  start = 0
  rings.each do |ring|
    out += [start].pack("l<")
    start += ring.length
  end
  rings.each do |ring|
    ring.each { |x, y| out += [x, y].pack("E2") }
  end
  out
end

def write_counties_shp(path, shapes)
  contents = shapes.map { |rings| polygon_record(rings) }
  total_words = contents.sum { |c| 4 + c.bytesize / 2 }
  all_points = shapes.flat_map { |rings| rings.flatten(1) }
  xs = all_points.map { |p| p[0] }
  ys = all_points.map { |p| p[1] }

  File.open(path, "wb") do |out|
    out.write([9994].pack("l>"))
    out.write([0, 0, 0, 0, 0].pack("l>5"))
    out.write([50 + total_words].pack("l>"))
    out.write([1000].pack("l<"))
    out.write([5].pack("l<")) # Polygon
    out.write([xs.min, ys.min, xs.max, ys.max].pack("E4"))
    out.write([0.0, 0.0, 0.0, 0.0].pack("E4"))
    contents.each_with_index do |content, idx|
      out.write([idx + 1].pack("l>"))
      out.write([content.bytesize / 2].pack("l>"))
      out.write(content)
    end
  end
end

def write_counties_dbf(path, names, populations)
  header_size = 32 + 2 * 32 + 1
  record_size = 1 + 12 + 8
  File.open(path, "wb") do |out|
    out.write([0x03, 0, 0, 0].pack("C4"))
    out.write([names.length].pack("V")) # <I: uint32 little-endian
    out.write([header_size].pack("v"))  # <H: uint16 little-endian
    out.write([record_size].pack("v"))
    out.write("\0" * 20)

    write_field = lambda do |name, field_type, length|
      out.write(name.ljust(11, "\0"))
      out.write(field_type)
      out.write("\0" * 4)
      out.write([length].pack("C"))
      out.write("\0" * 15)
    end

    write_field.call("NAME", "C", 12)
    write_field.call("POP", "N", 8)
    out.write([0x0D].pack("C"))

    names.zip(populations).each do |name, pop|
      out.write(" ")
      out.write(name[0, 12].ljust(12, " "))
      out.write(pop[0, 8].ljust(8, " "))
    end
  end
end

base = File.join(Dir.tmpdir, "datamunge_gis_ex_counties_rb")

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
plain = [[[10, 0], [10, 10], [20, 10], [20, 0], [10, 0]]]
with_lake = [
  [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]],
  [[3, 3], [4, 3], [4, 4], [3, 4], [3, 3]],
]

write_counties_shp(base + ".shp", [with_lake, plain])
write_counties_dbf(base + ".dbf", ["Lakeside", "Plainview"], ["48231", "19876"])

counties = Datamunge::ShapeLayer.read(base)

puts "shapes = #{counties.size}, shape_type = #{counties.shape_type}"
puts "bounds = #{counties.bounds.to_a}"
puts

attributes = counties.attributes
puts "attributes"
puts attributes.to_string
puts

(0...counties.size).each do |i|
  name = attributes.string_at("NAME", i)
  puts "#{name}: #{counties.num_parts(i)} ring(s)"
end

svg_path = File.join(Dir.tmpdir, "datamunge_gis_ex_map_rb.svg")
counties.plot.save_svg(svg_path)
puts "\nmap saved to #{svg_path}"
