local dm = require("datamunge")

-- Plain Lua tables never auto-convert to vector<T> parameters in Lua's SWIG bindings --
-- build a real DVector/SVector/IVector via the sized constructor and 0-indexed assignment.
local function dv(t)
  local v = dm.DVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function sv(t)
  local v = dm.SVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local function iv(t)
  local v = dm.IVector(#t)
  for i, x in ipairs(t) do v[i - 1] = x end
  return v
end

local sales = dm.DataFrame()
sales:add_string_column("region", sv({"west", "west", "east", "south", "south", "south"}))
sales:add_string_column("product", sv({"widget", "widget", "widget", "gizmo", "gizmo", "gizmo"}))
sales:add_numeric_column("sales", dv({10, 10, 14, 8, 0, 11}), iv({1, 1, 1, 1, 0, 1}))
sales:add_string_column("quarter", sv({"Q1", "Q1", "Q1", "Q2", "Q2", ""}), iv({1, 1, 1, 1, 1, 0}))

print("raw data")
print(sales:to_string())
print()

local cleaned = sales:drop_duplicates(sv({"region", "product", "sales", "quarter"}))
cleaned:fill_null_string("quarter", "unknown")
cleaned:fill_null_numeric("sales", 0.0)
print("after drop_duplicates + fill_null")
print(cleaned:to_string())
print()

local selected = cleaned:select(sv({"region", "sales", "quarter"})):sort_by("sales", false)
print("selected + sorted")
print(selected:to_string())
print()

local grouped = cleaned:group_by_sum(sv({"region"}), sv({"sales"})):sort_by("sales", false)
print("group_by_sum(region)")
print(grouped:to_string())
print()

local targets = dm.DataFrame()
targets:add_string_column("region", sv({"west", "east", "south"}))
targets:add_numeric_column("target", dv({18, 12, 25}))
local joined = grouped:join(targets, "region", "region", "left")
print("joined with targets")
print(joined:to_string())
print()

local shape = cleaned:shape()
print("shape = (" .. shape[0] .. ", " .. shape[1] .. ")")
print("sales count = " .. cleaned:numeric_count("sales"))
print("sales nulls = " .. cleaned:numeric_null_count("sales"))
print("sales sum = " .. cleaned:numeric_sum("sales"))
print("sales mean = " .. cleaned:numeric_mean("sales"))
