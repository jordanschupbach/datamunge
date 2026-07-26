-- Demonstrates datamunge.ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes)
-- and drawing it as a map. Real .shp/.dbf files are large binary bundles that don't belong in
-- this repo, so this example first writes a tiny synthetic shapefile by hand (two "counties":
-- one plain square, one with a lake-shaped hole) using the same ESRI byte layout
-- ShapeLayer.read() expects, then reads it back through the public API. Lua's string.pack
-- mirrors Python's struct.pack for the binary layout.
local dm = require("datamunge")

local function polygon_record(rings)
  local all_points = {}
  for _, ring in ipairs(rings) do
    for _, p in ipairs(ring) do all_points[#all_points + 1] = p end
  end
  local minx, miny, maxx, maxy = math.huge, math.huge, -math.huge, -math.huge
  for _, p in ipairs(all_points) do
    minx, maxx = math.min(minx, p[1]), math.max(maxx, p[1])
    miny, maxy = math.min(miny, p[2]), math.max(maxy, p[2])
  end
  local out = string.pack("<i4", 5) -- shape type: Polygon
  out = out .. string.pack("<dddd", minx, miny, maxx, maxy)
  out = out .. string.pack("<i4", #rings)
  out = out .. string.pack("<i4", #all_points)
  local start = 0
  for _, ring in ipairs(rings) do
    out = out .. string.pack("<i4", start)
    start = start + #ring
  end
  for _, ring in ipairs(rings) do
    for _, p in ipairs(ring) do
      out = out .. string.pack("<dd", p[1], p[2])
    end
  end
  return out
end

local function write_counties_shp(path, shapes)
  local contents = {}
  for i, rings in ipairs(shapes) do contents[i] = polygon_record(rings) end
  local total_words = 0
  for _, c in ipairs(contents) do total_words = total_words + 4 + (#c // 2) end
  local minx, miny, maxx, maxy = math.huge, math.huge, -math.huge, -math.huge
  for _, rings in ipairs(shapes) do
    for _, ring in ipairs(rings) do
      for _, p in ipairs(ring) do
        minx, maxx = math.min(minx, p[1]), math.max(maxx, p[1])
        miny, maxy = math.min(miny, p[2]), math.max(maxy, p[2])
      end
    end
  end
  local f = assert(io.open(path, "wb"))
  f:write(string.pack(">i4", 9994))
  f:write(string.pack(">i4i4i4i4i4", 0, 0, 0, 0, 0))
  f:write(string.pack(">i4", 50 + total_words))
  f:write(string.pack("<i4", 1000))
  f:write(string.pack("<i4", 5)) -- Polygon
  f:write(string.pack("<dddd", minx, miny, maxx, maxy))
  f:write(string.pack("<dddd", 0.0, 0.0, 0.0, 0.0))
  for i, content in ipairs(contents) do
    f:write(string.pack(">i4", i))
    f:write(string.pack(">i4", #content // 2))
    f:write(content)
  end
  f:close()
end

local function ljust(s, n, ch)
  if #s >= n then return s:sub(1, n) end
  return s .. string.rep(ch, n - #s)
end

local function write_counties_dbf(path, names, populations)
  local header_size = 32 + 2 * 32 + 1
  local record_size = 1 + 12 + 8
  local f = assert(io.open(path, "wb"))
  f:write(string.char(0x03, 0, 0, 0))
  f:write(string.pack("<I4", #names))
  f:write(string.pack("<I2", header_size))
  f:write(string.pack("<I2", record_size))
  f:write(string.rep("\0", 20))

  local function write_field(name, field_type, length)
    f:write(ljust(name, 11, "\0"))
    f:write(field_type)
    f:write(string.rep("\0", 4))
    f:write(string.char(length))
    f:write(string.rep("\0", 15))
  end

  write_field("NAME", "C", 12)
  write_field("POP", "N", 8)
  f:write(string.char(0x0D))

  for i, name in ipairs(names) do
    f:write(" ")
    f:write(ljust(name:sub(1, 12), 12, " "))
    f:write(ljust(populations[i]:sub(1, 8), 8, " "))
  end
  f:close()
end

local base = "datamunge_gis_ex_counties_lua"

-- Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
local plain = { { { 10, 0 }, { 10, 10 }, { 20, 10 }, { 20, 0 }, { 10, 0 } } }
local with_lake = {
  { { 0, 0 }, { 0, 10 }, { 10, 10 }, { 10, 0 }, { 0, 0 } },
  { { 3, 3 }, { 4, 3 }, { 4, 4 }, { 3, 4 }, { 3, 3 } },
}

write_counties_shp(base .. ".shp", { with_lake, plain })
write_counties_dbf(base .. ".dbf", { "Lakeside", "Plainview" }, { "48231", "19876" })

local counties = dm.ShapeLayer.read(base)

print("shapes = " .. counties:size() .. ", shape_type = " .. counties:shape_type())
local bounds = counties:bounds()
local bt = {}
for i = 0, bounds:size() - 1 do bt[#bt + 1] = string.format("%g", bounds[i]) end
print("bounds = [" .. table.concat(bt, ", ") .. "]")
print()

local attributes = counties:attributes()
print("attributes")
print(attributes:to_string())
print()

for i = 0, counties:size() - 1 do
  local name = attributes:string_at("NAME", i)
  print(name .. ": " .. counties:num_parts(i) .. " ring(s)")
end

local svg_path = "datamunge_gis_ex_map_lua.svg"
counties:plot():save_svg(svg_path)
print("\nmap saved to " .. svg_path)
