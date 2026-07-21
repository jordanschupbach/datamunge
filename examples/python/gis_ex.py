# Demonstrates datamunge.ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and
# drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in this
# repo, so this example first writes a tiny synthetic shapefile by hand (two "counties": one
# plain square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer.read()
# expects, then reads it back through the public API.

import struct
import tempfile
from pathlib import Path

from pydatamunge import datamunge


def polygon_record(rings):
    all_points = [p for ring in rings for p in ring]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    out = struct.pack("<i", 5)  # shape type: Polygon
    out += struct.pack("<4d", min(xs), min(ys), max(xs), max(ys))
    out += struct.pack("<i", len(rings))
    out += struct.pack("<i", len(all_points))
    start = 0
    for ring in rings:
        out += struct.pack("<i", start)
        start += len(ring)
    for ring in rings:
        for x, y in ring:
            out += struct.pack("<2d", x, y)
    return out


def write_counties_shp(path, shapes):
    contents = [polygon_record(rings) for rings in shapes]
    total_words = sum(4 + len(c) // 2 for c in contents)
    all_points = [p for rings in shapes for ring in rings for p in ring]
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]

    with open(path, "wb") as out:
        out.write(struct.pack(">i", 9994))
        out.write(struct.pack(">5i", 0, 0, 0, 0, 0))
        out.write(struct.pack(">i", 50 + total_words))
        out.write(struct.pack("<i", 1000))
        out.write(struct.pack("<i", 5))  # Polygon
        out.write(struct.pack("<4d", min(xs), min(ys), max(xs), max(ys)))
        out.write(struct.pack("<4d", 0.0, 0.0, 0.0, 0.0))
        for i, content in enumerate(contents, start=1):
            out.write(struct.pack(">i", i))
            out.write(struct.pack(">i", len(content) // 2))
            out.write(content)


def write_counties_dbf(path, names, populations):
    header_size = 32 + 2 * 32 + 1
    record_size = 1 + 12 + 8
    with open(path, "wb") as out:
        out.write(bytes([0x03, 0, 0, 0]))
        out.write(struct.pack("<I", len(names)))
        out.write(struct.pack("<H", header_size))
        out.write(struct.pack("<H", record_size))
        out.write(bytes(20))

        def write_field(name, field_type, length):
            out.write(name.encode("ascii").ljust(11, b"\0"))
            out.write(field_type.encode("ascii"))
            out.write(bytes(4))
            out.write(bytes([length]))
            out.write(bytes(15))

        write_field("NAME", "C", 12)
        write_field("POP", "N", 8)
        out.write(bytes([0x0D]))

        for name, pop in zip(names, populations):
            out.write(b" ")
            out.write(name.encode("ascii")[:12].ljust(12, b" "))
            out.write(pop.encode("ascii")[:8].ljust(8, b" "))


tmp = Path(tempfile.gettempdir())
base = str(tmp / "datamunge_gis_ex_counties_py")

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
plain = [[(10, 0), (10, 10), (20, 10), (20, 0), (10, 0)]]
with_lake = [
    [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)],
    [(3, 3), (4, 3), (4, 4), (3, 4), (3, 3)],
]

write_counties_shp(base + ".shp", [with_lake, plain])
write_counties_dbf(base + ".dbf", ["Lakeside", "Plainview"], ["48231", "19876"])

counties = datamunge.ShapeLayer.read(base)

print(f"shapes = {counties.size()}, shape_type = {counties.shape_type()}")
print(f"bounds = {list(counties.bounds())}")
print()

attributes = counties.attributes()
print("attributes")
print(attributes.to_string())
print()

for i in range(counties.size()):
    name = attributes.string_at("NAME", i)
    print(f"{name}: {counties.num_parts(i)} ring(s)")

svg_path = str(tmp / "datamunge_gis_ex_map_py.svg")
counties.plot().save_svg(svg_path)
print(f"\nmap saved to {svg_path}")
