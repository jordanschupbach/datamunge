<?php

function dvector($values) {
    $out = new DVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function svector($values) {
    $out = new SVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function szvector($values) {
    $out = new SizeVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

function shape_str($shp) {
    $parts = array();
    for ($i = 0; $i < $shp->size(); $i++) $parts[] = $shp->get($i);
    return implode(", ", $parts);
}

print("=================== Construction ===================\n");
$z = Tensor::zeros(szvector(array(2, 3)));
print("zeros([2,3]): " . $z->to_string() . "\n");

$eyet = Tensor::eye(3);
print("eye(3): " . $eyet->to_string() . "\n");

$r = Tensor::arange(0.0, 12.0, 1.0)->reshape(szvector(array(3, 4)));
print("arange(0,12).reshape([3,4]): " . $r->to_string() . "\n");

print("\n=================== Shape ops ===================\n");
$rt = $r->transpose();
print("transpose -> shape [" . shape_str($rt->shape()) . "]\n");
$sliced = $r->slice(1, 1, 3);
print("slice(axis=1, start=1, stop=3): " . $sliced->to_string() . "\n");

print("\n=================== Broadcasting arithmetic ===================\n");
$col = Tensor::from_values(szvector(array(3, 1)), dvector(array(1, 2, 3)));
$row = Tensor::from_values(szvector(array(1, 4)), dvector(array(10, 20, 30, 40)));
$broadcast_sum = $col->add($row);
print("(3,1) + (1,4) -> " . $broadcast_sum->to_string() . "\n");

print("\n=================== Reductions ===================\n");
print("r.sum() = " . $r->sum() . ", r.mean() = " . $r->mean() . "\n");
$col_means = $r->mean_axis(0);
print("column means (axis=0): " . $col_means->to_string() . "\n");

print("\n=================== Linear algebra ===================\n");
$a = Tensor::from_values(szvector(array(2, 3)), dvector(array(1, 2, 3, 4, 5, 6)));
$b = Tensor::from_values(szvector(array(3, 2)), dvector(array(7, 8, 9, 10, 11, 12)));
print("matmul(2x3, 3x2) -> " . $a->matmul($b)->to_string() . "\n");

$v1 = Tensor::from_values(szvector(array(3)), dvector(array(1, 2, 3)));
$v2 = Tensor::from_values(szvector(array(3)), dvector(array(4, 5, 6)));
print("dot([1,2,3], [4,5,6]) = " . $v1->dot($v2) . "\n");
print("outer(v1, v2) -> " . $v1->outer($v2)->to_string() . "\n");

print("\n=================== Comparisons & masks ===================\n");
$mask = $r->greater_equal(Tensor::full(szvector(array(3, 4)), 6.0));
print("r >= 6 -> " . $mask->to_string() . "\n");
print("count(r >= 6) = " . $mask->sum() . "\n");

print("\n=================== A real dataset as a Tensor ===================\n");
$iris = DataFrame::iris();
$n = $iris->nrows();
$flat = array();
for ($i = 0; $i < $n; $i++) {
    $flat[] = $iris->numeric_at("Sepal.Length", $i);
    $flat[] = $iris->numeric_at("Sepal.Width", $i);
    $flat[] = $iris->numeric_at("Petal.Length", $i);
    $flat[] = $iris->numeric_at("Petal.Width", $i);
}
$x = Tensor::from_values(szvector(array($n, 4)), dvector($flat));
print("iris feature tensor shape: [" . shape_str($x->shape()) . "]\n");

$feature_means = $x->mean_axis(0);
$centered = $x->subtract($feature_means->reshape(szvector(array(1, 4))));
$scatter = $centered->transpose()->matmul($centered);
print("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " . $feature_means->to_string() . "\n");
print("(X-mean)^T (X-mean) [4x4 scatter matrix]: " . $scatter->to_string(16) . "\n");

$species = array();
for ($i = 0; $i < $n; $i++) $species[] = $iris->string_at("Species", $i);
$species_tensor = Tensor::from_string_values(szvector(array($n)), svector($species));
$setosa_mask = $species_tensor->equal(Tensor::from_string_values(szvector(array(1)), svector(array("setosa"))));
print("setosa count = " . $setosa_mask->sum() . " (of $n rows)\n");

?>
