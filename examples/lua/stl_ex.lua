local dm = require("datamunge")

local n = 100
local v = dm.DVector(n)

for i = 0, n - 1 do v[i] = i * 1.5 end
for i = 0, n - 1 do print(v[i]) end

local v2 = dm.IVector(n)
for i = 0, n - 1 do v2[i] = math.floor(i * 1.5) end
for i = 0, n - 1 do print(v2[i]) end

local p = dm.IPair(3, 4)
print("p: (" .. p.first .. ", " .. p.second .. ")")

local p2 = dm.DPair(10, 20)
print("p2: (" .. p2.first .. ", " .. p2.second .. ")")
