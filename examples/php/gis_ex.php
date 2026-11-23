<?php
// CMD: just TARGET=gis_ex run-php
// ShapeLayer: write a tiny synthetic shapefile (.shp + .dbf) by hand, read it back and draw it as a map.
// Real .shp/.dbf files are large binary bundles that don't belong in this repo, so this example
// writes two "counties" (one plain square, one with a lake-shaped hole) using the same ESRI byte
// layout ShapeLayer::read() expects, then reads them back through the public API.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\ShapeLayer;

const BASE_PATH = 'datamunge_gis_ex_counties_php';

// Roughly C's %g: integral doubles print without a fractional part.
function fmtG(float $x): string
{
    return $x == floor($x) ? (string) (int) $x : sprintf('%.6f', $x);
}

// Bounding box (minx, miny, maxx, maxy) of every point in rings.
/** @param list<list<array{float, float}>> $rings */
function ringBounds(array $rings): array
{
    $points = array_merge(...$rings);
    $xs = array_column($points, 0);
    $ys = array_column($points, 1);
    return [min($xs), min($ys), max($xs), max($ys)];
}

// Bytes of one Polygon record's contents (without the record header): pack() 'V' is a
// little-endian int32, 'N' big-endian, 'v' little-endian int16 and 'e' a little-endian double.
/** @param list<list<array{float, float}>> $rings */
function polygonRecord(array $rings): string
{
    $output = pack('V', 5);  // shape type: Polygon
    $output .= pack('e*', ...ringBounds($rings));
    $output .= pack('V', count($rings));
    $output .= pack('V', array_sum(array_map('count', $rings)));
    $start = 0;
    foreach ($rings as $ring) {
        $output .= pack('V', $start);
        $start += count($ring);
    }
    foreach (array_merge(...$rings) as [$x, $y]) {
        $output .= pack('e2', $x, $y);
    }
    return $output;
}

/** @param list<list<list<array{float, float}>>> $shapes */
function writeCountiesShp(string $path, array $shapes): void
{
    $contents = array_map('polygonRecord', $shapes);
    $totalWords = array_sum(array_map(fn(string $c): int => 4 + intdiv(strlen($c), 2), $contents));
    $f = pack('N', 9994);
    $f .= pack('N5', 0, 0, 0, 0, 0);
    $f .= pack('N', 50 + $totalWords);
    $f .= pack('V', 1000);
    $f .= pack('V', 5);  // Polygon
    $f .= pack('e*', ...ringBounds(array_merge(...$shapes)));
    $f .= pack('e4', 0.0, 0.0, 0.0, 0.0);
    foreach ($contents as $i => $content) {
        $f .= pack('N', $i + 1);
        $f .= pack('N', intdiv(strlen($content), 2));
        $f .= $content;
    }
    file_put_contents($path, $f);
}

function ljust(string $s, int $n, string $ch): string
{
    return strlen($s) >= $n ? substr($s, 0, $n) : str_pad($s, $n, $ch);
}

/**
 * @param list<string> $names
 * @param list<string> $populations
 */
function writeCountiesDbf(string $path, array $names, array $populations): void
{
    $headerSize = 32 + 2 * 32 + 1;
    $recordSize = 1 + 12 + 8;
    $f = "\x03\0\0\0";
    $f .= pack('V', count($names));
    $f .= pack('v', $headerSize);
    $f .= pack('v', $recordSize);
    $f .= str_repeat("\0", 20);
    $writeField = function (string $name, string $fieldType, int $length) use (&$f): void {
        $f .= ljust($name, 11, "\0");
        $f .= $fieldType;
        $f .= str_repeat("\0", 4);
        $f .= chr($length);
        $f .= str_repeat("\0", 15);
    };
    $writeField('NAME', 'C', 12);
    $writeField('POP', 'N', 8);
    $f .= "\x0D";
    foreach (array_map(null, $names, $populations) as [$name, $pop]) {
        $f .= ' ';
        $f .= ljust($name, 12, ' ');
        $f .= ljust($pop, 8, ' ');
    }
    file_put_contents($path, $f);
}

// Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
$plain = [[[10.0, 0.0], [10.0, 10.0], [20.0, 10.0], [20.0, 0.0], [10.0, 0.0]]];
$withLake = [
    [[0.0, 0.0], [0.0, 10.0], [10.0, 10.0], [10.0, 0.0], [0.0, 0.0]],
    [[3.0, 3.0], [4.0, 3.0], [4.0, 4.0], [3.0, 4.0], [3.0, 3.0]],
];

writeCountiesShp(BASE_PATH . '.shp', [$withLake, $plain]);
writeCountiesDbf(BASE_PATH . '.dbf', ['Lakeside', 'Plainview'], ['48231', '19876']);

$counties = ShapeLayer::read(BASE_PATH);

echo "shapes = {$counties->size()}, shape_type = {$counties->shapeType()}\n";
echo 'bounds = [' . implode(', ', array_map('fmtG', $counties->bounds())) . "]\n\n";

$attrs = $counties->attributes();
echo "attributes\n$attrs\n\n";

for ($i = 0; $i < $counties->size(); $i++) {
    echo "{$attrs->stringAt('NAME', $i)}: {$counties->numParts($i)} ring(s)\n";
}

const SVG_PATH = 'datamunge_gis_ex_map_php.svg';
$counties->plot()->saveSvg(SVG_PATH);
echo "\nmap saved to " . SVG_PATH . "\n";
