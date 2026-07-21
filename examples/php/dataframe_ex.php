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

function ivector($values) {
    $out = new IVector(count($values));
    foreach ($values as $i => $v) $out->set($i, $v);
    return $out;
}

$sales = new DataFrame();
$sales->add_string_column("region", svector(array("west", "west", "east", "south", "south", "south")));
$sales->add_string_column("product", svector(array("widget", "widget", "widget", "gizmo", "gizmo", "gizmo")));
$sales->add_numeric_column("sales", dvector(array(10.0, 10.0, 14.0, 8.0, 0.0, 11.0)), ivector(array(1, 1, 1, 1, 0, 1)));
$sales->add_string_column("quarter", svector(array("Q1", "Q1", "Q1", "Q2", "Q2", "")), ivector(array(1, 1, 1, 1, 1, 0)));

print("raw data\n");
print($sales->to_string() . "\n\n");

$cleaned = $sales->drop_duplicates(svector(array("region", "product", "sales", "quarter")));
$cleaned->fill_null_string("quarter", "unknown");
$cleaned->fill_null_numeric("sales", 0.0);
print("after drop_duplicates + fill_null\n");
print($cleaned->to_string() . "\n\n");

$selected = $cleaned->select(svector(array("region", "sales", "quarter")))->sort_by("sales", false);
print("selected + sorted\n");
print($selected->to_string() . "\n\n");

$grouped = $cleaned->group_by_sum(svector(array("region")), svector(array("sales")))->sort_by("sales", false);
print("group_by_sum(region)\n");
print($grouped->to_string() . "\n\n");

$targets = new DataFrame();
$targets->add_string_column("region", svector(array("west", "east", "south")));
$targets->add_numeric_column("target", dvector(array(18.0, 12.0, 25.0)));
$joined = $grouped->join($targets, "region", "region", "left");
print("joined with targets\n");
print($joined->to_string() . "\n\n");

$shape = $cleaned->shape();
print("shape = (" . $shape->get(0) . ", " . $shape->get(1) . ")\n");
print("sales count = " . $cleaned->numeric_count("sales") . "\n");
print("sales nulls = " . $cleaned->numeric_null_count("sales") . "\n");
print("sales sum = " . $cleaned->numeric_sum("sales") . "\n");
print("sales mean = " . $cleaned->numeric_mean("sales") . "\n");

?>
