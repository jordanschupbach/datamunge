<?php
// CMD: just TARGET=rplot_ex run-php
// R base-graphics-style plotting with RPlot / RLayout: plot, hist, barplot, boxplot, pie, qqnorm, layouts.
declare(strict_types=1);

require_once __DIR__ . '/../../src/datamungephp/php/datamunge.php';

use Datamunge\{RLayout, RPlot, Rgb};

// plot(x, y, type = "p") then abline() layered on afterward.
$scatter = RPlot::plot([1.0, 2.0, 3.0, 4.0, 5.0], [2.1, 3.9, 6.2, 7.8, 10.1], 'p', 'observed');
$scatter->abline(0.0, 2.0, new Rgb(220, 38, 38), 1.5);
$scatter->title('plot() + abline()')->xLabel('x')->yLabel('y');
$scatter->saveSvg('php_rplot_scatter.svg');

// hist(): equal-width binning over the data range.
$histogram = RPlot::hist([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 6.0, 6.0, 7.0], 6, 'counts');
$histogram->saveSvg('php_rplot_hist.svg');

// barplot(): categorical positions with x tick labels.
RPlot::barplot([23.0, 41.0, 12.0], ['A', 'B', 'C'])->title('barplot()')->saveSvg('php_rplot_barplot.svg');

// boxplot(): Tukey five-number summary per group.
RPlot::boxplot([[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0], [1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 20.0]],
    ['low variance', 'has outlier'])
    ->title('boxplot()')->saveSvg('php_rplot_boxplot.svg');

// pie(): wedge areas proportional to value, axes hidden automatically.
RPlot::pie([35.0, 25.0, 20.0, 20.0], ['Q1', 'Q2', 'Q3', 'Q4'])->title('pie()')->saveSvg('php_rplot_pie.svg');

// curve() is omitted to stay in step with the Lua example (Lua cannot supply the callback).

// qqnorm() + qqline(): standard-normal Q-Q plot with a fitted reference line.
$residuals = [-2.1, -1.3, -0.8, -0.4, -0.1, 0.2, 0.5, 0.9, 1.4, 2.3];
RPlot::qqnorm($residuals)->qqline($residuals)->saveSvg('php_rplot_qqnorm.svg');

// par(mfrow = c(1, 2))-style multi-panel composition via RLayout.
RLayout::create(1, 2)->add($scatter)->add($histogram)->saveSvg('php_rplot_layout.svg');

echo "Wrote 7 SVGs to php_rplot_*.svg\n";
