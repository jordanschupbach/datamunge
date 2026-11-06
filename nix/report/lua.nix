{
  pkgs,
  lua,
  datamungelua,
  ...
}:
let
  v = lua.luaversion;
in
pkgs.mkShell {
  packages = [ lua ];
  shellHook = ''
    export LUA_PATH="${datamungelua}/share/lua/${v}/?.lua;${datamungelua}/share/lua/?.lua;;"
    export LUA_CPATH="${datamungelua}/lib/lua/${v}/?.so;${datamungelua}/lib/lua/?.so;${datamungelua}/lib64/lua/${v}/?.so;${datamungelua}/lib64/lua/?.so;;"
  '';
}
