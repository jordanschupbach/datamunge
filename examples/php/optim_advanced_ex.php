<?php
// CMD: just TARGET=optim_advanced_ex run-php
// Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
// AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from PHP.
//
// Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
// EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
// ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
// implement the objective. From PHP, extend the interface class and override its virtual
// methods; the optimizer then calls the PHP overrides. Coordinates are updated in place through
// a by-reference array parameter (the C++ parameter is a non-const std::vector<double>&).
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{AugmentedLagrangian, AugmentedLagrangianOptions, BayesianOptimization, BayesianOptimizationOptions, Fista,
    FistaOptions, HessianFunction, InteriorPoint, InteriorPointOptions, LevenbergMarquardt, LevenbergMarquardtOptions,
    NativeObject, Newton, NewtonOptions, ProximalGradient, ProximalGradientOptions, RbfGaussianProcessSurrogate,
    ResidualFunction, Sqp, SqpOptions, TrustRegionNewton, TrustRegionNewtonOptions};

/** A double the way C# prints it with "G<precision>" (like C's %g, but E+XX exponents). */
function fmtG(float $x, int $precision): string
{
    if ($x == 0.0) {
        return '0';
    }
    [$mant, $exp] = explode('e', sprintf('%.' . ($precision - 1) . 'e', $x));
    $exp = (int) $exp;
    $sign = $mant[0] === '-' ? '-' : '';
    $digits = rtrim(str_replace(['-', '.'], '', $mant), '0');
    if ($exp < -5 || $exp >= $precision) {
        return $sign . $digits[0] . (strlen($digits) > 1 ? '.' . substr($digits, 1) : '')
            . 'E' . ($exp < 0 ? '-' : '+') . sprintf('%02d', abs($exp));
    }
    if ($exp < 0) {
        return $sign . '0.' . str_repeat('0', -$exp - 1) . $digits;
    }
    if (strlen($digits) <= $exp + 1) {
        return $sign . $digits . str_repeat('0', $exp + 1 - strlen($digits));
    }
    return $sign . substr($digits, 0, $exp + 1) . '.' . substr($digits, $exp + 1);
}

/**
 * "CppClass()" for a wrapped object, as the C# binding's ToString() prints it: the C++ class
 * name comes from the generated PHPDoc ("Wraps the C++ class `ns::Name`").
 */
function describe(NativeObject $object): string
{
    $doc = (string) (new ReflectionClass($object))->getDocComment();
    $name = preg_match('/Wraps the C\+\+ class `(?:\w+::)*(\w+)`/', $doc, $m) ? $m[1]
        : (new ReflectionClass($object))->getShortName();
    return "$name()";
}

final class Quadratic extends HessianFunction
{
    public function evaluate(array $coordinates): float
    {
        return ($coordinates[0] - 1) ** 2 + 10 * ($coordinates[1] + 2) ** 2;
    }

    public function gradient(array $coordinates): array
    {
        return [2 * ($coordinates[0] - 1), 20 * ($coordinates[1] + 2)];
    }

    public function hessian(array $coordinates): array
    {
        return [[2.0, 0.0], [0.0, 20.0]];
    }
}

final class Exponential extends ResidualFunction
{
    /** @param list<float> $ts */
    public function __construct(private array $ts)
    {
        parent::__construct();
    }

    public function residuals(array $coordinates): array
    {
        [$a, $b] = $coordinates;
        return array_map(fn(float $t): float => $a * exp($b * $t) - 2 * exp(0.5 * $t), $this->ts);
    }

    public function jacobian(array $coordinates): array
    {
        [$a, $b] = $coordinates;
        return array_map(fn(float $t): array => [exp($b * $t), $a * $t * exp($b * $t)], $this->ts);
    }
}

final class Incomplete extends HessianFunction
{
    public function evaluate(array $coordinates): float
    {
        return 0.0;
    }
}

echo "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================\n";
// Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
// then check the acquisition function favors an unexplored region.
$xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];
$ys = array_map(fn(float $x): float => ($x - 1) * ($x - 1), $xs);
$surrogate = new RbfGaussianProcessSurrogate(1.0, 1e-6);
$surrogate->fit(array_map(fn(float $x): array => [$x], $xs), $ys);
$incumbent = min(0.0, min($ys));  // best observed value so far
echo 'incumbent (best observed y): ' . fmtG($incumbent, 17) . "\n";
foreach ([0.25, 1.0, 2.5] as $x) {
    printf("acquisition(x=%.2f) = %.6f\n", $x, $surrogate->acquisition([$x], $incumbent));
}

echo "\n=================== Options structs + optimizer construction (all 10 new classes) ===================\n";
// Every new Options struct and optimizer class constructs cleanly, proving
// the binding for the whole batch is wired correctly.
$pgOptions = new ProximalGradientOptions();
$pgOptions->stepSize = 0.05;
$pg = new ProximalGradient($pgOptions);
echo 'ProximalGradient: ' . describe($pg) . "\n";
$fista = new Fista(new FistaOptions());
echo 'FISTA: ' . describe($fista) . "\n";
$lmOptions = new LevenbergMarquardtOptions();
$lmOptions->maxIterations = 200;
$lm = new LevenbergMarquardt($lmOptions);
echo 'LevenbergMarquardt: ' . describe($lm) . "\n";
$newton = new Newton(new NewtonOptions());
echo 'Newton: ' . describe($newton) . "\n";
$trn = new TrustRegionNewton(new TrustRegionNewtonOptions());
echo 'TrustRegionNewton: ' . describe($trn) . "\n";
$al = new AugmentedLagrangian(new AugmentedLagrangianOptions());
echo 'AugmentedLagrangian: ' . describe($al) . "\n";
$sqp = new Sqp(new SqpOptions());
echo 'SQP: ' . describe($sqp) . "\n";
$ip = new InteriorPoint(new InteriorPointOptions());
echo 'InteriorPoint: ' . describe($ip) . "\n";
$boOptions = new BayesianOptimizationOptions();
$boOptions->initialSamples = 8;
$bo = new BayesianOptimization($boOptions);
echo 'BayesianOptimization: ' . describe($bo) . "\n";

echo "\n=================== PHP objectives (subclassing the C++ interfaces) ===================\n";
// Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
$start = [5.0, 5.0];
$value = $newton->optimize(new Quadratic(), $start);
printf("Newton: f = %s at (%.6f, %.6f) (exact minimum 0 at (1, -2))\n", fmtG($value, 6), $start[0], $start[1]);

// Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
$parameters = [1.0, 0.1];
$sse = $lm->optimize(new Exponential([0.0, 0.5, 1.0, 1.5, 2.0]), $parameters);
printf("LevenbergMarquardt: residual = %s, a = %.6f, b = %.6f (exact: 2, 0.5)\n", fmtG($sse, 3), $parameters[0],
    $parameters[1]);

// An interface method a subclass leaves out throws when the optimizer calls it.
$err = null;
try {
    $origin = [0.0, 0.0];
    $newton->optimize(new Incomplete(), $origin);
} catch (Throwable $e) {
    $err = $e;
}
echo 'missing gradient(): ok=' . ($err === null ? 'True' : 'False') . ', error mentions it: '
    . ($err !== null && str_contains($err->getMessage(), 'gradient') ? 'True' : 'False') . "\n";
