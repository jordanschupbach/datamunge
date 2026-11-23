<?php
// CMD: just TARGET=lmm_ex run-php
// Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Lmm};
use function Datamunge\Data\penguins;

/** A double the way C# prints it (shortest round-trip, e.g. 6, 0.1, 1E-05). */
function cs(float $x): string
{
    if (is_nan($x)) {
        return 'NaN';
    }
    if (is_infinite($x)) {
        return $x > 0 ? '∞' : '-∞';
    }
    if ($x == 0.0) {
        return fdiv(1.0, $x) < 0 ? '-0' : '0';
    }
    for ($p = 0; $p < 17; $p++) {
        $s = sprintf("%.{$p}e", $x);
        if ((float) $s === $x) {
            break;
        }
    }
    [$mant, $exp] = explode('e', $s);
    $exp = (int) $exp;
    $sign = $mant[0] === '-' ? '-' : '';
    $digits = rtrim(str_replace(['-', '.'], '', $mant), '0');
    if ($exp < -5 || $exp >= 15) {
        $body = $digits[0] . (strlen($digits) > 1 ? '.' . substr($digits, 1) : '')
            . 'E' . ($exp < 0 ? '-' : '+') . sprintf('%02d', abs($exp));
    } elseif ($exp < 0) {
        $body = '0.' . str_repeat('0', -$exp - 1) . $digits;
    } elseif (strlen($digits) <= $exp + 1) {
        $body = $digits . str_repeat('0', $exp + 1 - strlen($digits));
    } else {
        $body = substr($digits, 0, $exp + 1) . '.' . substr($digits, $exp + 1);
    }
    return $sign . $body;
}

/**
 * .NET's seeded System.Random (Knuth's subtractive generator), so the simulated data -- and so
 * the output -- match the C# example exactly.
 */
final class DotNetRandom
{
    private const MBIG = 2147483647;
    private const MSEED = 161803398;
    /** @var array<int, int> */
    private array $seedArray;
    private int $inext = 0;
    private int $inextp = 21;

    public function __construct(int $seed)
    {
        $this->seedArray = array_fill(0, 56, 0);
        $mj = self::MSEED - abs($seed);
        $this->seedArray[55] = $mj;
        $mk = 1;
        for ($i = 1; $i < 55; $i++) {
            $ii = (21 * $i) % 55;
            $this->seedArray[$ii] = $mk;
            $mk = $mj - $mk;
            if ($mk < 0) {
                $mk += self::MBIG;
            }
            $mj = $this->seedArray[$ii];
        }
        for ($k = 1; $k < 5; $k++) {
            for ($i = 1; $i < 56; $i++) {
                $this->seedArray[$i] -= $this->seedArray[1 + ($i + 30) % 55];
                if ($this->seedArray[$i] < 0) {
                    $this->seedArray[$i] += self::MBIG;
                }
            }
        }
    }

    private function sample(): float
    {
        $this->inext = $this->inext + 1 >= 56 ? 1 : $this->inext + 1;
        $this->inextp = $this->inextp + 1 >= 56 ? 1 : $this->inextp + 1;
        $ret = $this->seedArray[$this->inext] - $this->seedArray[$this->inextp];
        if ($ret === self::MBIG) {
            $ret--;
        }
        if ($ret < 0) {
            $ret += self::MBIG;
        }
        $this->seedArray[$this->inext] = $ret;
        return $ret * (1.0 / self::MBIG);
    }

    /** A double in [0, 1). */
    public function nextDouble(): float
    {
        return $this->sample();
    }

    /** An int in [$min, $max). */
    public function next(int $min, int $max): int
    {
        return (int) ($this->sample() * ($max - $min)) + $min;
    }
}

$rng = new DotNetRandom(2024);
$uniform = fn(): float => 1 - $rng->nextDouble();  // (0, 1]: avoids log(0) in the Box-Muller transform.
$gauss = fn(): float => sqrt(-2 * log($uniform())) * cos(2 * M_PI * $uniform());

echo "=================== Random intercept on a real dataset (penguins) ===================\n";
(new Lmm(penguins(), 'body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)'))->printSummary();

echo "\n=================== Random intercept + slope on a simulated multi-school dataset ===================\n";
// 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
// and the return on an extra hour of study vary by school -- the textbook case for a
// random-intercept-and-slope model.
const N_SCHOOLS = 30;
$schoolIntercept = array_map(fn(): float => $gauss() * 6, range(1, N_SCHOOLS));
$schoolSlope = array_map(fn(): float => $gauss() * 1.2, range(1, N_SCHOOLS));
const TRUE_INTERCEPT = 60.0;
const TRUE_SLOPE = 3.0;

$school = $studyHours = $score = [];
for ($s = 0; $s < N_SCHOOLS; $s++) {
    $nStudents = 15 + $rng->next(0, 21);
    for ($student = 0; $student < $nStudents; $student++) {
        $hours = $rng->nextDouble() * 10;
        $school[] = (float) $s;
        $studyHours[] = $hours;
        $score[] = TRUE_INTERCEPT + $schoolIntercept[$s] + (TRUE_SLOPE + $schoolSlope[$s]) * $hours + $gauss() * 4;
    }
}
$df = new DataFrame();
$df->addNumericColumn('school', $school);
$df->addNumericColumn('study_hours', $studyHours);
$df->addNumericColumn('score', $score);

$model = new Lmm($df, 'score ~ study_hours + (1 + study_hours | school)');
$model->printSummary();

echo "\nTrue generating values: intercept=" . cs(TRUE_INTERCEPT) . ', slope=' . cs(TRUE_SLOPE)
    . ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0\n";

echo "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---\n";
$labels = $model->groupLabels();
for ($idx = 0; $idx < 3; $idx++) {
    $re = $model->randomEffectsForGroup($idx);
    echo "school $labels[$idx]: intercept shift=" . cs($re[0]) . ', slope shift=' . cs($re[1]) . "\n";
}

echo "\n--- Prediction: population-level vs. school-adjusted ---\n";
$population = new DataFrame();
$school0 = new DataFrame();
$population->addNumericColumn('study_hours', [5.0]);
$school0->addNumericColumn('study_hours', [5.0]);
$school0->addNumericColumn('school', [0.0]);
echo '5 study hours, unseen school:      ' . cs($model->predict($population)[0]) . " (fixed effects only)\n";
echo '5 study hours, school 0 (known):    ' . cs($model->predict($school0)[0]) . " (fixed effects + school 0's BLUP)\n";
