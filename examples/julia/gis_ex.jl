# CMD: just TARGET=gis_ex run-julia

# ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
# Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
# writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
# layout ShapeLayer.read() expects, then reads them back through the public API.
using Datamunge
using Printf

# Little- and big-endian packers that append to a byte buffer.
pack_i4_le(buf, value) = append!(buf, reinterpret(UInt8, [htol(Int32(value))]))
pack_i4_be(buf, value) = append!(buf, reinterpret(UInt8, [hton(Int32(value))]))
pack_i2_le(buf, value) = append!(buf, reinterpret(UInt8, [htol(Int16(value))]))
pack_double_le(buf, x) = append!(buf, reinterpret(UInt8, [htol(Float64(x))]))
pack_ascii(buf, s) = append!(buf, codeunits(s))
pack_zeros(buf, n) = append!(buf, zeros(UInt8, n))

"Bounding box (minx, miny, maxx, maxy) of every point in `rings`."
function ring_bounds(rings)
    points = reduce(vcat, rings)
    xs, ys = first.(points), last.(points)
    (minimum(xs), minimum(ys), maximum(xs), maximum(ys))
end

"Bytes of one Polygon record's contents (without the record header)."
function polygon_record(rings)
    points = reduce(vcat, rings)
    out = UInt8[]
    pack_i4_le(out, 5)  # shape type: Polygon
    foreach(b -> pack_double_le(out, b), ring_bounds(rings))
    pack_i4_le(out, length(rings))
    pack_i4_le(out, length(points))
    start = 0
    for ring in rings
        pack_i4_le(out, start)
        start += length(ring)
    end
    for (x, y) in points
        pack_double_le(out, x)
        pack_double_le(out, y)
    end
    out
end

function write_counties_shp(path, shapes)
    contents = polygon_record.(shapes)
    total_words = sum(4 + length(c) ÷ 2 for c in contents)
    f = UInt8[]
    pack_i4_be(f, 9994)
    for _ in 1:5 pack_i4_be(f, 0) end
    pack_i4_be(f, 50 + total_words)
    pack_i4_le(f, 1000)
    pack_i4_le(f, 5)  # Polygon
    foreach(b -> pack_double_le(f, b), ring_bounds(reduce(vcat, shapes)))
    for _ in 1:4 pack_double_le(f, 0.0) end
    for (i, content) in enumerate(contents)
        pack_i4_be(f, i)
        pack_i4_be(f, length(content) ÷ 2)
        append!(f, content)
    end
    write(path, f)
end

ljust(s, n, ch) = length(s) >= n ? s[1:n] : s * repeat(ch, n - length(s))

function write_counties_dbf(path, names, populations)
    header_size = 32 + 2 * 32 + 1
    record_size = 1 + 12 + 8
    f = UInt8[0x03, 0, 0, 0]
    pack_i4_le(f, length(names))
    pack_i2_le(f, header_size)
    pack_i2_le(f, record_size)
    pack_zeros(f, 20)
    function write_field(name, field_type, len)
        pack_ascii(f, ljust(name, 11, '\0'))
        pack_ascii(f, field_type)
        pack_zeros(f, 4)
        push!(f, UInt8(len))
        pack_zeros(f, 15)
    end
    write_field("NAME", "C", 12)
    write_field("POP", "N", 8)
    push!(f, 0x0D)
    for (name, pop) in zip(names, populations)
        pack_ascii(f, " ")
        pack_ascii(f, ljust(name, 12, ' '))
        pack_ascii(f, ljust(pop, 8, ' '))
    end
    write(path, f)
end

const base = "datamunge_gis_ex_counties_julia"

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
const plain = [[(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)]]
const with_lake = [[(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
                   [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)]]

write_counties_shp("$base.shp", [with_lake, plain])
write_counties_dbf("$base.dbf", ["Lakeside", "Plainview"], ["48231", "19876"])

const counties = read(ShapeLayer, base)

println("shapes = $(size(counties)), shape_type = $(shape_type(counties))")
"Roughly C's %g: integral doubles print without a fractional part."
fmt_g(x) = isinteger(x) ? string(Int(x)) : @sprintf("%.6f", x)
println("bounds = [$(join(fmt_g.(bounds(counties)), ", "))]\n")

const attrs = attributes(counties)
println("attributes\n$(to_string(attrs))\n")

for i in 0:size(counties)-1
    println("$(string_at(attrs, "NAME", i)): $(num_parts(counties, i)) ring(s)")
end

let svg_path = "datamunge_gis_ex_map_julia.svg"
    save_svg(plot(counties), svg_path)
    println("\nmap saved to $svg_path")
end
