<?php
// Smoke tests of the Rosetta-generated PHP binding (just test-php):
//   php tests/php/test_datamunge.php  (with DATAMUNGE_PHP_LIBRARY set, as the dev shells do)
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{Callback, DataFrame, InvalidArgumentException, NativeError};
use function Datamunge\{call_with_callback, hello, make_dpair, make_dvector, map_dvector_with_callback, runif, seql,
    sum_dpair, sum_dvector};

$checks = 0;
function check(bool $ok, string $message): void
{
    global $checks;
    $checks++;
    if (!$ok) {
        throw new RuntimeException("check failed: $message");
    }
}

final class Doubler extends Callback
{
    public function call(float $x): float
    {
        return 2 * $x;
    }
}

final class FromPhp extends Exception
{
}

final class Failing extends Callback
{
    public function call(float $x): float
    {
        throw new FromPhp('from PHP');
    }
}

// Smoke test: call into the native library.
hello();

// std::vector<double> and std::pair<double,double> round trips.
$v = make_dvector(1.0, 2.0, 3.0);
check($v === [1.0, 2.0, 3.0], 'std::vector<double> is an array');
check(sum_dvector($v) === 6.0, 'sum_dvector');
check(sum_dvector([0.5, 1.5]) === 2.0, 'arrays convert to std::vector');
$p = make_dpair(1.25, 2.75);
check($p === [1.25, 2.75], 'std::pair is an array of two');
check(sum_dpair($p) === 4.0, 'sum_dpair');

// The C++ Callback class, and a PHP subclass overriding its virtual call() (a director).
$native = new Callback();
check(call_with_callback(3.0, $native) === 3.0, 'native Callback');
check(sum_dvector(map_dvector_with_callback([1.0, 2.0, 3.0], $native)) === 6.0, 'native map');
$doubler = new Doubler();
check(call_with_callback(3.0, $doubler) === 6.0, 'the PHP override runs');
check(map_dvector_with_callback([1.0, 2.0, 3.0], $doubler) === [2.0, 4.0, 6.0], 'override from C++ loops');

// A PHP exception inside an override reaches the caller unchanged.
try {
    call_with_callback(1.0, new Failing());
    check(false, 'the PHP exception propagates');
} catch (FromPhp $e) {
    check($e->getMessage() === 'from PHP', 'the PHP exception propagates');
}

// C++ exceptions become the matching SPL exception (std::invalid_argument here).
try {
    runif(1.0, 0.0, 3);
    check(false, 'C++ exceptions throw');
} catch (InvalidArgumentException $e) {
    check($e instanceof NativeError && $e instanceof \InvalidArgumentException, 'C++ exceptions throw');
}

// Classes, methods, the policy namespace Datamunge\Data and the PHPDoc documentation.
$frame = new DataFrame();
$frame->addNumericColumn('x', [1.0, 2.0, 3.0]);
check($frame->nrows() === 3, 'DataFrame');
check(seql(0.0, 1.0, 5) === [0.0, 0.25, 0.5, 0.75, 1.0], 'seql');
check(\Datamunge\Data\iris()->nrows() === 150, 'the Data namespace');
$doc = (new ReflectionFunction('Datamunge\seql'))->getDocComment();
check(is_string($doc) && str_contains($doc, 'evenly spaced') && str_contains($doc, '@param int $n'), 'PHPDoc documentation');

echo "test_datamunge.php: all $checks checks passed\n";
