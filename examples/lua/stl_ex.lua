-- CMD: just TARGET=stl_ex run-lua

local dm = require("datamunge")

-- C++ std::vector and std::pair values are plain Lua tables: vectors are 1-based arrays and
-- pairs are { first, second }.
local n = 100
local v = {}

for i = 1, n do v[i] = (i - 1) * 1.5 end
for i = 1, n do print(v[i]) end

local v2 = {}
for i = 1, n do v2[i] = math.floor((i - 1) * 1.5) end
for i = 1, n do print(v2[i]) end

local p = { 3, 4 }
print("p: (" .. p[1] .. ", " .. p[2] .. ")")

local p2 = { 10, 20 }
print("p2: (" .. p2[1] .. ", " .. p2[2] .. ")")
