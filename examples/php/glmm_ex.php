<?php
// CMD: just TARGET=glmm_ex run-php
// Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{DataFrame, Glmm};
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

$rng = new DotNetRandom(4242);
$uniform = fn(): float => $rng->nextDouble();
$gauss = function () use ($uniform): float {
    [$u1, $u2] = [1 - $uniform(), $uniform()];  // 1 - u keeps log() away from 0
    return sqrt(-2 * log($u1)) * cos(2 * M_PI * $u2);
};

echo "=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================\n";
// Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
// confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
// is not tied to location, so this is a well-behaved fit.
$penguins = penguins();
$isMale = $bodyMass = $island = [];
for ($i = 0; $i < $penguins->nrows(); $i++) {
    if (array_filter(['sex', 'body_mass_g', 'island'], fn(string $c): bool => $penguins->isNull($c, $i)) !== []) {
        continue;
    }
    $isMale[] = $penguins->stringAt('sex', $i) === 'male' ? 1.0 : 0.0;
    $bodyMass[] = $penguins->numericAt('body_mass_g', $i);
    $island[] = $penguins->stringAt('island', $i);
}
$sexDf = new DataFrame();
$sexDf->addNumericColumn('is_male', $isMale);
$sexDf->addNumericColumn('body_mass_g', $bodyMass);
$sexDf->addStringColumn('island', $island);
(new Glmm($sexDf, 'is_male ~ body_mass_g + (1 | island)', 'binomial'))->printSummary();

echo "\n=================== Poisson mixed model on simulated multi-site count data ===================\n";
// 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
// the baseline traffic and the promo's effectiveness vary by store.
const N_STORES = 25;
$storeEffect = array_map(fn(): float => $gauss() * 0.4, range(1, N_STORES));
const TRUE_INTERCEPT = 2.0;
const TRUE_SLOPE = 0.3;

$store = $promo = $visits = [];
for ($s = 0; $s < N_STORES; $s++) {
    $nDays = 15 + $rng->next(0, 11);
    for ($day = 0; $day < $nDays; $day++) {
        $promoIntensity = $uniform() * 3;
        $lam = exp(TRUE_INTERCEPT + $storeEffect[$s] + TRUE_SLOPE * $promoIntensity);
        // Knuth's Poisson sampler.
        [$threshold, $k, $p] = [exp(-$lam), 0, 1.0];
        do {
            $k++;
            $p *= $uniform();
        } while ($p > $threshold);
        $store[] = (float) $s;
        $promo[] = $promoIntensity;
        $visits[] = (float) ($k - 1);
    }
}
$df = new DataFrame();
$df->addNumericColumn('store', $store);
$df->addNumericColumn('promo', $promo);
$df->addNumericColumn('visits', $visits);
$storeModel = new Glmm($df, 'visits ~ promo + (1 | store)', 'poisson');

$storeModel->printSummary();

echo "\nTrue generating values: intercept=" . cs(TRUE_INTERCEPT) . ', slope=' . cs(TRUE_SLOPE)
    . ", random-intercept SD (log scale)=0.4\n";

echo "\n--- BLUPs for a few stores ---\n";
$labels = $storeModel->groupLabels();
for ($idx = 0; $idx < 3; $idx++) {
    echo "store $labels[$idx]: intercept shift=" . cs($storeModel->randomEffectsForGroup($idx)[0]) . "\n";
}

echo "\n--- Prediction: population-level vs. store-adjusted ---\n";
$population = new DataFrame();
$store0 = new DataFrame();
$population->addNumericColumn('promo', [1.5]);
$store0->addNumericColumn('promo', [1.5]);
$store0->addNumericColumn('store', [0.0]);
echo 'promo=1.5, unseen store:   ' . cs($storeModel->predict($population)[0]) . " expected visits (fixed effects only)\n";
echo 'promo=1.5, store 0 (known): ' . cs($storeModel->predict($store0)[0]) . " expected visits (fixed effects + store 0's BLUP)\n";
