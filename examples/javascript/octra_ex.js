// Local dev: build first (`just build-javascript`), then run this example.
const octra = require("../../index.js");

octra.hello();

// Note: SWIG's Node/JavaScript backend does not support directors/virtual-method
// overrides the same way as Python/Ruby/Perl. `Callback` can still be passed,
// but the default implementation is identity.
const cb = new octra.Callback();
console.log("call_with_callback(3.0) =", octra.call_with_callback(3.0, cb));
const v2 = octra.map_dvector_with_callback(
  octra.make_dvector(1.0, 2.0, 3.0),
  cb,
);
console.log("sum_dvector(map_dvector_with_callback) =", octra.sum_dvector(v2));

// Bridging: pass a JS function into native code (via C callback trampoline).
console.log(
  "call_with_function(3.0) =",
  octra.call_with_function(3.0, (x) => x * 2.0),
);
const out = octra.map_array_with_function([1.0, 2.0, 3.0], (x) => x * 2.0);
console.log("map_array_with_function([1,2,3]) =", out);
