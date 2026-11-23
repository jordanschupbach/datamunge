-- CMD: just TARGET=lmm_ex run-haskell
-- Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
module Main (main) where

import Control.Monad (forM, forM_, replicateM)
import Data.Array.IO (IOUArray, newArray, readArray, writeArray)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  rng <- newDotNetRandom 2024
  let uniform = (1 -) <$> nextDouble rng -- (0, 1]: avoids log(0) in the Box-Muller transform.
      gauss = do
        u1 <- uniform
        u2 <- uniform
        pure (sqrt (-2 * log u1) * cos (2 * pi * u2))

  putStrLn "=================== Random intercept on a real dataset (penguins) ==================="
  penguins <- dataPenguins
  newLmm penguins "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)" >>= lmmPrintSummary

  putStrLn "\n=================== Random intercept + slope on a simulated multi-school dataset ==================="
  -- 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
  -- and the return on an extra hour of study vary by school -- the textbook case for a
  -- random-intercept-and-slope model.
  let nSchools = 30
      trueIntercept = 60.0
      trueSlope = 3.0
  schoolIntercept <- replicateM nSchools ((* 6) <$> gauss)
  schoolSlope <- replicateM nSchools ((* 1.2) <$> gauss)

  rowsBySchool <- forM (zip3 [0 :: Int ..] schoolIntercept schoolSlope) $ \(s, intercept, slope) -> do
    nStudents <- (15 +) <$> nextInt rng 0 21
    replicateM nStudents $ do
      hours <- (* 10) <$> nextDouble rng
      noise <- gauss
      pure (fromIntegral s, hours, trueIntercept + intercept + (trueSlope + slope) * hours + noise * 4)
  let (school, studyHours, score) = unzip3 (concat rowsBySchool)
  df <- newDataFrame
  dataFrameAddNumericColumn df "school" school
  dataFrameAddNumericColumn df "study_hours" studyHours
  dataFrameAddNumericColumn df "score" score

  model <- newLmm df "score ~ study_hours + (1 + study_hours | school)"
  lmmPrintSummary model

  putStrLn ("\nTrue generating values: intercept=" ++ fmt trueIntercept ++ ", slope=" ++ fmt trueSlope
              ++ ", random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0")

  putStrLn "\n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---"
  labels <- lmmGroupLabels model
  forM_ [0 .. 2] $ \idx -> do
    re <- lmmRandomEffectsForGroup model idx
    putStrLn ("school " ++ labels !! idx ++ ": intercept shift=" ++ fmt (re !! 0) ++ ", slope shift=" ++ fmt (re !! 1))

  putStrLn "\n--- Prediction: population-level vs. school-adjusted ---"
  population <- newDataFrame
  school0 <- newDataFrame
  dataFrameAddNumericColumn population "study_hours" [5.0]
  dataFrameAddNumericColumn school0 "study_hours" [5.0]
  dataFrameAddNumericColumn school0 "school" [0.0]
  populationPrediction <- (!! 0) <$> lmmPredict model population
  putStrLn ("5 study hours, unseen school:      " ++ fmt populationPrediction ++ " (fixed effects only)")
  schoolPrediction <- (!! 0) <$> lmmPredict model school0
  putStrLn ("5 study hours, school 0 (known):    " ++ fmt schoolPrediction ++ " (fixed effects + school 0's BLUP)")

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

-- | Random.Next(minValue, maxValue): [minValue, maxValue).
nextInt :: DotNetRandom -> Int -> Int -> IO Int
nextInt rng lo hi = (\s -> truncate (s * fromIntegral (hi - lo)) + lo) <$> sample rng
