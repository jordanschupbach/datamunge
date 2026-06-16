-- Lua loader for the SWIG-generated native module `octra`.
-- This keeps Lua-side behavior stable across SWIG versions.

local name = "octra"

local so_path, search_err = package.searchpath(name, package.cpath)
if not so_path then
	error(("could not find native module '%s' in package.cpath: %s"):format(name, tostring(search_err)))
end

local open_sym = "luaopen_" .. name
local open_fn, load_err = package.loadlib(so_path, open_sym)
if not open_fn then
	error(("failed to load '%s' from %s: %s"):format(open_sym, so_path, tostring(load_err)))
end

return open_fn()
