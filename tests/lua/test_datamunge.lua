local ok, mod = pcall(require, "datamunge")
assert(ok, "failed to require 'datamunge': " .. tostring(mod))
assert(type(mod) == "table", "expected datamunge module to be a table")
assert(type(mod.hello) == "function", "expected datamunge.hello to be a function")

-- Smoke test: call into the native library.
mod.hello()

-- std::vector<double> roundtrip + by-const-ref usage
assert(type(mod.make_dvector) == "function", "expected datamunge.make_dvector to be a function")
assert(type(mod.sum_dvector) == "function", "expected datamunge.sum_dvector to be a function")
local v = mod.make_dvector(1.0, 2.0, 3.0)
local sumv = mod.sum_dvector(v)
assert(sumv == 6.0, ("expected sum_dvector=6.0, got %s"):format(tostring(sumv)))

-- std::pair<double,double> roundtrip + by-const-ref usage
assert(type(mod.make_dpair) == "function", "expected datamunge.make_dpair to be a function")
assert(type(mod.sum_dpair) == "function", "expected datamunge.sum_dpair to be a function")
local p = mod.make_dpair(1.25, 2.75)
local sump = mod.sum_dpair(p)
assert(sump == 4.0, ("expected sum_dpair=4.0, got %s"):format(tostring(sump)))

-- callback/director usage: implement Callback.call in Lua and pass to C++
assert(type(mod.Callback) == "table" or type(mod.Callback) == "function", "expected datamunge.Callback to exist")
local cb = mod.Callback()

local y = mod.call_with_callback(3.0, cb)
assert(y == 3.0, ("expected call_with_callback=3.0, got %s"):format(tostring(y)))

local out = mod.map_dvector_with_callback(mod.make_dvector(1.0, 2.0, 3.0), cb)
assert(mod.sum_dvector(out) == 6.0, "expected mapped sum to be 6.0")
