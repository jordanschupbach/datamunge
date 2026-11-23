-- CMD: just TARGET=hypothesis_tests_ex run-haskell
-- Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
module Main (main) where

import Control.Monad (filterM, when)
import Data.List (sort)
import Datamunge
import Numeric (floatToDigits)

printResult :: String -> HypothesisTestResult -> IO ()
printResult label r = do
  alternative <- hypothesisTestResultAlternative r
  let alternativeName = case alternative of
        AlternativeLess -> "less"
        AlternativeGreater -> "greater"
        AlternativeTwoSided -> "two.sided"
  statistic <- hypothesisTestResultStatistic r
  parameter1 <- hypothesisTestResultParameter1 r
  parameter2 <- hypothesisTestResultParameter2 r
  pValue <- hypothesisTestResultPValue r
  hasConfInt <- hypothesisTestResultHasConfInt r
  lower <- hypothesisTestResultConfIntLower r
  upper <- hypothesisTestResultConfIntUpper r
  method <- hypothesisTestResultMethod r
  putStr (label ++ ": statistic=" ++ fmt statistic)
  when (parameter1 /= 0) $ putStr (", df1=" ++ fmt parameter1)
  when (parameter2 /= 0) $ putStr (", df2=" ++ fmt parameter2)
  putStr (", p=" ++ fmt pValue ++ " (" ++ alternativeName ++ ")")
  when hasConfInt $ putStr (", CI=[" ++ fmt lower ++ ", " ++ fmt upper ++ "]")
  putStrLn (" -- " ++ method)

columnForSpecies :: DataFrame -> String -> String -> IO [Double]
columnForSpecies df name species = do
  n <- dataFrameNrows df
  rows <- filterM (fmap (== species) . dataFrameStringAt df "Species") [0 .. n - 1]
  mapM (dataFrameNumericAt df name) rows

column :: DataFrame -> String -> IO [Double]
column df name = do
  n <- dataFrameNrows df
  mapM (dataFrameNumericAt df name) [0 .. n - 1]

main :: IO ()
main = do
  iris <- dataIris

  setosaPetal <- columnForSpecies iris "Petal.Length" "setosa"
  versicolorPetal <- columnForSpecies iris "Petal.Length" "versicolor"
  virginicaPetal <- columnForSpecies iris "Petal.Length" "virginica"

  putStrLn "=================== t-tests: petal length, setosa vs. versicolor ==================="
  tTestTwoSample setosaPetal versicolorPetal >>= printResult "Welch two-sample t-test"
  wilcoxonRankSumTest setosaPetal versicolorPetal >>= printResult "Wilcoxon rank-sum test"

  putStrLn "\n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ==================="
  let allPetal = setosaPetal ++ versicolorPetal ++ virginicaPetal
      sizes = map length [setosaPetal, versicolorPetal, virginicaPetal]
  oneWayAnova allPetal sizes >>= printResult "One-way ANOVA"
  kruskalWallisTest allPetal sizes >>= printResult "Kruskal-Wallis"

  putStrLn "\n=================== Correlation: sepal length vs. petal length ==================="
  sepalLength <- column iris "Sepal.Length"
  petalLength <- column iris "Petal.Length"
  pearsonCorrelationTest sepalLength petalLength >>= printResult "Pearson correlation"
  spearmanCorrelationTest sepalLength petalLength >>= printResult "Spearman correlation"

  putStrLn "\n=================== F-test: petal length variance, setosa vs. virginica ==================="
  fTestVariance setosaPetal virginicaPetal >>= printResult "F test"

  putStrLn "\n=================== Normality: is sepal length normally distributed within setosa? ==================="
  setosaSepal <- columnForSpecies iris "Sepal.Length" "setosa"
  shapiroFranciaTest setosaSepal >>= printResult "Shapiro-Francia"
  ksTestOneSampleNormalWithSd setosaSepal 5.006 0.3525 >>= printResult "KS vs. fitted normal"

  putStrLn "\n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ==================="
  let medianAll = sort allPetal !! (length allPetal `div` 2)
      countLong = length . filter (> medianAll)
  do
    let setosaLong = countLong setosaPetal
        setosaShort = length setosaPetal - setosaLong
        versicolorLong = countLong versicolorPetal
        versicolorShort = length versicolorPetal - versicolorLong
    putStrLn ("table: setosa=[" ++ show setosaLong ++ "," ++ show setosaShort ++ "] versicolor=["
                ++ show versicolorLong ++ "," ++ show versicolorShort ++ "]")
    let table = map fromIntegral [setosaLong, setosaShort, versicolorLong, versicolorShort]
    chiSquaredTestIndependence table 2 2 >>= printResult "Chi-squared independence"
    fisherExactTest2x2 setosaLong setosaShort versicolorLong versicolorShort >>= printResult "Fisher's exact test"

  putStrLn "\n=================== Proportion / binomial: fraction of \"long\" petals overall ==================="
  do
    let longCount = countLong allPetal
        n = length allPetal
    proportionTestOneSampleWithP longCount n 0.5 >>= printResult "One-sample proportion test (vs 0.5)"
    binomialTestWithP longCount n 0.5 >>= printResult "Exact binomial test (vs 0.5)"

-- | A Double as C#'s default ToString() prints it: the shortest round-trip digits, in E notation
-- when the decimal exponent is below -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
fmt :: Double -> String
fmt x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | x < 0 || isNegativeZero x = '-' : fmt (negate x)
  | x == 0 = "0"
  | exp10 < -4 || exp10 >= 15 =
      let (d, rest) = splitAt 1 digits
          magnitude = show (abs exp10)
       in d ++ (if null rest then "" else '.' : rest) ++ "E" ++ (if exp10 < 0 then "-" else "+")
            ++ replicate (2 - length magnitude) '0' ++ magnitude
  | e <= 0 = "0." ++ replicate (negate e) '0' ++ digits
  | e >= length digits = digits ++ replicate (e - length digits) '0'
  | otherwise = let (whole, frac) = splitAt e digits in whole ++ "." ++ frac
  where
    (ds, e) = floatToDigits 10 x
    digits = concatMap show ds
    exp10 = e - 1
