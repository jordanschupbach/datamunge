// CMD: just TARGET=datamunge_ex run-javascript

// Local dev: build first (`just build-javascript`), then run this example.
const datamunge = require("../../index.js");

datamunge.hello();

// The default `Callback` is the identity.
const cb = new datamunge.Callback();
console.log("call_with_callback(3.0) =", datamunge.call_with_callback(3.0, cb));
const v2 = datamunge.map_dvector_with_callback(
  datamunge.make_dvector(1.0, 2.0, 3.0),
  cb,
);
console.log("sum_dvector(map_dvector_with_callback) =", datamunge.sum_dvector(v2));

// Subclass `Callback` in JavaScript: `call` overrides the C++ virtual method, so native
// code calls back into JS.
class Doubler extends datamunge.Callback {
  call(x) {
    return x * 2.0;
  }
}
const doubler = new Doubler();
console.log("call_with_callback(3.0, Doubler) =", datamunge.call_with_callback(3.0, doubler));
const out = datamunge.map_dvector_with_callback([1.0, 2.0, 3.0], doubler);
console.log("map_dvector_with_callback([1,2,3], Doubler) =", out);
