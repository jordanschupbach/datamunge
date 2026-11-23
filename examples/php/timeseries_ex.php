<?php
// CMD: just TARGET=timeseries_ex run-php
// Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{Arima, ArimaOptions, ExponentialSmoothing, ExponentialSmoothingOptions, SeasonalType, TrendType};

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

$rng = new DotNetRandom(42);
$seed = function (int $n) use (&$rng): void {
    $rng = new DotNetRandom($n);
};
$uniform = function () use (&$rng): float {
    return 1 - $rng->nextDouble();  // (0, 1]: avoids log(0) in the Box-Muller transform.
};
$gauss = fn(): float => sqrt(-2 * log($uniform())) * cos(2 * M_PI * $uniform());

/** @param list<float> $v */
function joined(array $v): string
{
    return implode(', ', array_map('cs', $v));
}

echo "=================== ARIMA(1,1,1) on a simulated random walk with drift ===================\n";
$seed(42);
$y = [];
[$level, $prevShock] = [100.0, 0.0];
for ($i = 0; $i < 150; $i++) {
    $shock = $gauss();
    $level += 0.3 + 0.4 * $prevShock + $shock;  // integrated ARMA(0,1) increments with drift
    $y[] = $level;
    $prevShock = $shock;
}

$options = new ArimaOptions();
$options->p = 1;
$options->d = 1;
$options->q = 1;
$options->dePopulationSize = 80;
$options->deMaxGenerations = 400;
$model = new Arima($y, $options);

echo 'AR coefficient: ' . cs($model->arCoefficients()[0]) . "\n";
echo 'MA coefficient: ' . cs($model->maCoefficients()[0]) . "\n";
echo 'sigma^2: ' . cs($model->sigma2()) . ', AIC: ' . cs($model->aic()) . ', BIC: ' . cs($model->bic()) . "\n";

[$forecastValues, $stdErrors] = $model->forecastWithIntervals(6);
echo '6-step forecast: ' . joined($forecastValues) . "\n";
echo 'forecast std. errors: ' . joined($stdErrors) . "\n";

echo "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================\n";
$seed(7);
$s = [];
$prev = 0.0;
for ($i = 0; $i < 120; $i++) {
    $prev = 0.5 * $prev + $gauss();
    $s[] = 20 + 0.2 * $i + 5 * sin(2 * M_PI * $i / 12) + $prev;
}

$options2 = new ArimaOptions();
$options2->p = 1;
$options2->seasonalP = 1;
$options2->seasonalD = 1;
$options2->seasonalPeriod = 12;
$options2->dePopulationSize = 100;
$options2->deMaxGenerations = 500;
$model2 = new Arima($s, $options2);
echo 'AR coefficient: ' . cs($model2->arCoefficients()[0]) . ', seasonal AR coefficient: '
    . cs($model2->seasonalArCoefficients()[0]) . "\n";
echo '12-step forecast: ' . joined($model2->forecast(12)) . "\n";

echo "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================\n";
$seasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
$seed(11);
$y2 = array_map(fn(int $i): float => (100 + 2 * $i) * $seasonalShape[$i % 12] + $gauss() * 3, range(0, 47));

$esOptions = new ExponentialSmoothingOptions();
$esOptions->trend = TrendType::Additive;
$esOptions->seasonal = SeasonalType::Multiplicative;
$esOptions->seasonalPeriod = 12;
$esOptions->dePopulationSize = 60;
$esOptions->deMaxGenerations = 300;
$esModel = new ExponentialSmoothing($y2, $esOptions);
echo 'alpha=' . cs($esModel->alpha()) . ' beta=' . cs($esModel->beta()) . ' gamma=' . cs($esModel->gamma()) . "\n";
echo 'sigma^2: ' . cs($esModel->sigma2()) . ', AIC: ' . cs($esModel->aic()) . "\n";
echo '12-month forecast: ' . joined($esModel->forecast(12)) . "\n";

echo "\n=================== Simple exponential smoothing vs. Holt's linear trend ===================\n";
$flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];

$ses = new ExponentialSmoothing($flat, new ExponentialSmoothingOptions());
echo 'SES alpha= ' . cs($ses->alpha()) . "\n";
echo 'SES 5-step forecast: ' . joined($ses->forecast(5)) . "\n";

$holtOptions = new ExponentialSmoothingOptions();
$holtOptions->trend = TrendType::Additive;
$holt = new ExponentialSmoothing($flat, $holtOptions);
echo 'Holt alpha=' . cs($holt->alpha()) . ' beta=' . cs($holt->beta()) . "\n";
echo 'Holt 5-step forecast: ' . joined($holt->forecast(5)) . "\n";
