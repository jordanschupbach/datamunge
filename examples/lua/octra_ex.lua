local octra = require("octra")
octra.hello()

local v = octra.make_dvector(1.0, 2.0, 3.0)
print("sum_dvector:", octra.sum_dvector(v))

local p = octra.make_dpair(1.25, 2.75)
print("sum_dpair:", octra.sum_dpair(p))

local cb = octra.Callback()
print("call_with_callback(3.0):", octra.call_with_callback(3.0, cb))
local v2 = octra.map_dvector_with_callback(octra.make_dvector(1.0, 2.0, 3.0), cb)
print("sum_dvector(map_dvector_with_callback(1,2,3)):", octra.sum_dvector(v2))
