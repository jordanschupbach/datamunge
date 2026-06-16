<?php

// Run with:
//   php -d assert.exception=1 -d zend.assertions=1 --php-ini .user.ini tests/php/test_octra.php

assert(function_exists("hello"));
hello();

$p = new DPair(1.25, 2.5);
assert(abs($p->first - 1.25) < 1e-12);
assert(abs($p->second - 2.5) < 1e-12);

$v = new DVector(2);
$v->set(0, 3.0);
$v->set(1, 4.5);
assert(abs($v->get(0) - 3.0) < 1e-12);
assert(abs($v->get(1) - 4.5) < 1e-12);

print("ok\n");

?>
