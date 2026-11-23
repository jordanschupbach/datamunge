-- CMD: just TARGET=timeseries_ex run-haskell
-- Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
module Main (main) where

import Control.Monad (forM, forM_, replicateM)
import Data.Array.IO (IOUArray, newArray, readArray, writeArray)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Data.List (intercalate)
import Datamunge
import Numeric (floatToDigits)

-- | A standard normal deviate by Box-Muller, from uniforms in (0, 1] (avoiding log(0)).
gauss :: DotNetRandom -> IO Double
gauss rng = do
  u1 <- (1 -) <$> nextDouble rng
  u2 <- (1 -) <$> nextDouble rng
  pure (sqrt (-2 * log u1) * cos (2 * pi * u2))

joined :: [Double] -> String
joined = intercalate ", " . map fmt

main :: IO ()
main = do
  putStrLn "=================== ARIMA(1,1,1) on a simulated random walk with drift ==================="
  rng42 <- newDotNetRandom 42
  shocks <- replicateM 150 (gauss rng42)
  -- integrated ARMA(0,1) increments with drift
  let y = drop 1 (scanl (+) 100.0 (zipWith (\shock prevShock -> 0.3 + 0.4 * prevShock + shock) shocks (0.0 : shocks)))

  options <- newArimaOptions
  arimaOptionsSetP options 1
  arimaOptionsSetD options 1
  arimaOptionsSetQ options 1
  arimaOptionsSetDePopulationSize options 80
  arimaOptionsSetDeMaxGenerations options 400
  model <- newArimaWithOptions y options

  ar <- arimaArCoefficients model
  putStrLn ("AR coefficient: " ++ fmt (ar !! 0))
  ma <- arimaMaCoefficients model
  putStrLn ("MA coefficient: " ++ fmt (ma !! 0))
  sigma2 <- arimaSigma2 model
  aic <- arimaAic model
  bic <- arimaBic model
  putStrLn ("sigma^2: " ++ fmt sigma2 ++ ", AIC: " ++ fmt aic ++ ", BIC: " ++ fmt bic)

  (forecastValues, stdErrors) <- arimaForecastWithIntervals model 6
  putStrLn ("6-step forecast: " ++ joined forecastValues)
  putStrLn ("forecast std. errors: " ++ joined stdErrors)

  putStrLn "\n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ==================="
  rng7 <- newDotNetRandom 7
  noise <- replicateM 120 (gauss rng7)
  let ar1 = drop 1 (scanl (\prev e -> 0.5 * prev + e) 0.0 noise)
      s = [20 + 0.2 * fromIntegral i + 5 * sin (2 * pi * fromIntegral i / 12) + prev | (i, prev) <- zip [0 :: Int ..] ar1]

  options2 <- newArimaOptions
  arimaOptionsSetP options2 1
  arimaOptionsSetSeasonalP options2 1
  arimaOptionsSetSeasonalD options2 1
  arimaOptionsSetSeasonalPeriod options2 12
  arimaOptionsSetDePopulationSize options2 100
  arimaOptionsSetDeMaxGenerations options2 500
  model2 <- newArimaWithOptions s options2
  ar2 <- arimaArCoefficients model2
  seasonalAr <- arimaSeasonalArCoefficients model2
  putStrLn ("AR coefficient: " ++ fmt (ar2 !! 0) ++ ", seasonal AR coefficient: " ++ fmt (seasonalAr !! 0))
  forecast2 <- arimaForecast model2 12
  putStrLn ("12-step forecast: " ++ joined forecast2)

  putStrLn "\n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ==================="
  let seasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6]
  rng11 <- newDotNetRandom 11
  y2 <- forM [0 .. 47 :: Int] $ \i -> do
    e <- gauss rng11
    pure ((100 + 2 * fromIntegral i) * seasonalShape !! (i `mod` 12) + e * 3)

  esOptions <- newExponentialSmoothingOptions
  exponentialSmoothingOptionsSetTrend esOptions TrendTypeAdditive
  exponentialSmoothingOptionsSetSeasonal esOptions SeasonalTypeMultiplicative
  exponentialSmoothingOptionsSetSeasonalPeriod esOptions 12
  exponentialSmoothingOptionsSetDePopulationSize esOptions 60
  exponentialSmoothingOptionsSetDeMaxGenerations esOptions 300
  esModel <- newExponentialSmoothingWithOptions y2 esOptions
  alpha <- exponentialSmoothingAlpha esModel
  beta <- exponentialSmoothingBeta esModel
  gamma <- exponentialSmoothingGamma esModel
  putStrLn ("alpha=" ++ fmt alpha ++ " beta=" ++ fmt beta ++ " gamma=" ++ fmt gamma)
  esSigma2 <- exponentialSmoothingSigma2 esModel
  esAic <- exponentialSmoothingAic esModel
  putStrLn ("sigma^2: " ++ fmt esSigma2 ++ ", AIC: " ++ fmt esAic)
  esForecast <- exponentialSmoothingForecast esModel 12
  putStrLn ("12-month forecast: " ++ joined esForecast)

  putStrLn "\n=================== Simple exponential smoothing vs. Holt's linear trend ==================="
  let flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7]

  ses <- newExponentialSmoothingOptions >>= newExponentialSmoothingWithOptions flat
  sesAlpha <- exponentialSmoothingAlpha ses
  putStrLn ("SES alpha= " ++ fmt sesAlpha)
  sesForecast <- exponentialSmoothingForecast ses 5
  putStrLn ("SES 5-step forecast: " ++ joined sesForecast)

  holtOptions <- newExponentialSmoothingOptions
  exponentialSmoothingOptionsSetTrend holtOptions TrendTypeAdditive
  holt <- newExponentialSmoothingWithOptions flat holtOptions
  holtAlpha <- exponentialSmoothingAlpha holt
  holtBeta <- exponentialSmoothingBeta holt
  putStrLn ("Holt alpha=" ++ fmt holtAlpha ++ " beta=" ++ fmt holtBeta)
  holtForecast <- exponentialSmoothingForecast holt 5
  putStrLn ("Holt 5-step forecast: " ++ joined holtForecast)

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

-- | System.Random(seed) of .NET (the seeded, Knuth subtractive generator), so the simulated data
-- match the C# example's exactly.
data DotNetRandom = DotNetRandom (IOUArray Int Int) (IORef Int) (IORef Int)

newDotNetRandom :: Int -> IO DotNetRandom
newDotNetRandom seed = do
  seeds <- newArray (0, 55) 0
  let mbig = 2147483647
      mj0 = 161803398 - (if seed == -2147483648 then mbig else abs seed)
  writeArray seeds 55 mj0
  let fill i mj mk
        | i >= 55 = pure ()
        | otherwise = do
            let ii = (21 * i) `mod` 55
            writeArray seeds ii mk
            let mk' = mj - mk
            fill (i + 1) mk (if mk' < 0 then mk' + mbig else mk')
  fill 1 mj0 1
  forM_ [1 .. 4 :: Int] $ \_ -> forM_ [1 .. 55] $ \i -> do
    a <- readArray seeds i
    b <- readArray seeds (1 + (i + 30) `mod` 55)
    let v = a - b
    writeArray seeds i (if v < 0 then v + mbig else v)
  DotNetRandom seeds <$> newIORef 0 <*> newIORef 21

sample :: DotNetRandom -> IO Double
sample (DotNetRandom seeds inextRef inextpRef) = do
  let mbig = 2147483647
      bump i = if i + 1 >= 56 then 1 else i + 1
  inext <- bump <$> readIORef inextRef
  inextp <- bump <$> readIORef inextpRef
  writeIORef inextRef inext
  writeIORef inextpRef inextp
  a <- readArray seeds inext
  b <- readArray seeds inextp
  let v0 = a - b
      v1 = if v0 == mbig then v0 - 1 else v0
      v = if v1 < 0 then v1 + mbig else v1
  writeArray seeds inext v
  pure (fromIntegral v * (1.0 / fromIntegral mbig))

-- | Random.NextDouble(): [0, 1).
nextDouble :: DotNetRandom -> IO Double
nextDouble = sample

