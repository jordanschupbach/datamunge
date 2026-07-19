<?php

hello();

$v = make_dvector(1.0, 2.0, 3.0);
print("sum_dvector: " . sum_dvector($v) . "\n");

$p = make_dpair(1.25, 2.75);
print("sum_dpair: " . sum_dpair($p) . "\n");

$cb = new Callback();
print("call_with_callback(3.0): " . call_with_callback(3.0, $cb) . "\n");
$v2 = map_dvector_with_callback(make_dvector(1.0, 2.0, 3.0), $cb);
print("sum_dvector(map_dvector_with_callback(1,2,3)): " . sum_dvector($v2) . "\n");

?>
