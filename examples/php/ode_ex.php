<?php
// CMD: just TARGET=ode_ex run-php
// ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{OdeOptions, OdeSolution, OdeSolver, RPlot, StepMethod};

/** C#'s "G6" (like C's %g): six significant digits, no padding. */
function fmtG(float $x): string
{
    if ($x == 0.0) {
        return '0';
    }
    [$mant, $exp] = explode('e', sprintf('%.5e', $x));
    $exp = (int) $exp;
    $sign = $mant[0] === '-' ? '-' : '';
    $digits = rtrim(str_replace(['-', '.'], '', $mant), '0');
    if ($exp < -5 || $exp >= 6) {
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

/** @return list<float> */
function lastState(OdeSolution $sol): array
{
    return $sol->stateAt($sol->size() - 1);
}

/**
 * The time vector followed by the requested state components (0-based), one value per step.
 *
 * @return array{list<float>, list<list<float>>}
 */
function solutionSeries(OdeSolution $sol, int ...$components): array
{
    $steps = range(0, $sol->size() - 1);
    $states = array_map(fn(int $i): array => $sol->stateAt($i), $steps);
    return [array_map(fn(int $i): float => $sol->timeAt($i), $steps),
        array_map(fn(int $c): array => array_column($states, $c), $components)];
}

function rk4Solver(float $step): OdeSolver
{
    $options = new OdeOptions();
    $options->method = StepMethod::Rk4;
    $options->stepSize = $step;
    return new OdeSolver($options);
}

// NOTE: like the Lua version this example uses the built-in named systems via solveBuiltin.

echo "=================== Every named built-in system (no director needed) ===================\n";
$solver = new OdeSolver();
$systems = [
    ['exponential_decay', [1.0], [1.0]],
    ['logistic_growth', [1.0, 1.0], [0.5]],
    ['harmonic_oscillator', [1.0], [1.0, 0.0]],
    ['van_der_pol', [1.0], [2.0, 0.0]],
    ['lorenz', [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0]],
];
foreach ($systems as [$name, $parameters, $y0]) {
    $sol = $solver->solveBuiltin($name, $parameters, $y0, 0.0, 1.0);
    printf("%-22s steps=%-6d final state=[%s]\n", $name, $sol->stepsTaken, implode(', ', array_map('fmtG', lastState($sol))));
}

echo "\n=================== Harmonic oscillator (energy conservation) ===================\n";
$sol = rk4Solver(0.01)->solveBuiltin('harmonic_oscillator', [1.0], [1.0, 0.0], 0.0, 20.0);
[$x, $v] = lastState($sol);
printf("x(20) = %.6f (cos(20) = %.6f)\n", $x, cos(20.0));
printf("energy x^2+v^2 = %.6f (should stay near 1.0)\n", $x * $x + $v * $v);
[$ts, $series] = solutionSeries($sol, 0, 1);
$p = RPlot::plot($ts, $series[0], 'l', 'x(t)');
$p->lines($ts, $series[1], 'v(t)');
$p->title('Harmonic Oscillator')->xLabel('t')->yLabel('state');
$p->saveSvg('ode_harmonic_oscillator_php.svg');
echo "wrote ode_harmonic_oscillator_php.svg\n";

echo "\n=================== Lorenz attractor (phase plane) ===================\n";
$sol = rk4Solver(0.005)->solveBuiltin('lorenz', [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0], 0.0, 25.0);
echo "steps_taken = {$sol->stepsTaken}\n";
[, $series] = solutionSeries($sol, 0, 2);
$p = RPlot::plot($series[0], $series[1], 'l', 'trajectory');
$p->title('Lorenz Attractor (x-z phase plane)')->xLabel('x')->yLabel('z');
$p->saveSvg('ode_lorenz_phase_plane_php.svg');
echo "wrote ode_lorenz_phase_plane_php.svg\n";
