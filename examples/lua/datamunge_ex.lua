local datamunge = require("datamunge")
datamunge.hello()

local v = datamunge.make_dvector(1.0, 2.0, 3.0)
print("sum_dvector:", datamunge.sum_dvector(v))

local p = datamunge.make_dpair(1.25, 2.75)
print("sum_dpair:", datamunge.sum_dpair(p))

local cb = datamunge.Callback()
print("call_with_callback(3.0):", datamunge.call_with_callback(3.0, cb))
local v2 = datamunge.map_dvector_with_callback(datamunge.make_dvector(1.0, 2.0, 3.0), cb)
print("sum_dvector(map_dvector_with_callback(1,2,3)):", datamunge.sum_dvector(v2))
