<?php

$n = 100;
$v = new DVector($n);
for($i = 0; $i < 10; $i += 1) {
    $v->set($i, $i * 1.5);
}
for($i = 0; $i < 10; $i += 1) {
    print($v->get($i) . "\n");
}

$v2 = new IVector($n);
for($i = 0; $i < 10; $i += 1) {
    $v2->set($i, $i * 1.5); // Type coercion happens implicitly
}

for($i = 0; $i < 10; $i += 1) {
    print($v2->get($i) . "\n");
}



?>
