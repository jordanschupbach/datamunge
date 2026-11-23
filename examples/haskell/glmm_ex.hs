-- CMD: just TARGET=glmm_ex run-haskell
-- Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
module Main (main) where

import Control.Monad (filterM, forM, forM_, replicateM)
import Data.Array.IO (IOUArray, newArray, readArray, writeArray)
import Data.IORef (IORef, newIORef, readIORef, writeIORef)
import Datamunge
import Numeric (floatToDigits)

main :: IO ()
main = do
  rng <- newDotNetRandom 4242
  let uniform = nextDouble rng
      gauss = do
        u1 <- (1 -) <$> uniform -- 1 - u keeps log() away from 0
        u2 <- uniform
        pure (sqrt (-2 * log u1) * cos (2 * pi * u2))

  putStrLn "=================== Binomial (logistic) mixed model on a real dataset (penguins) ==================="
  -- Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
  -- confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
  -- is not tied to location, so this is a well-behaved fit.
  do
    penguins <- dataPenguins
    n <- dataFrameNrows penguins
    complete <- filterM (\i -> not . or <$> mapM (\c -> dataFrameIsNull penguins c i) ["sex", "body_mass_g", "island"]) [0 .. n - 1]
    isMale <- mapM (\i -> (\s -> if s == "male" then 1.0 else 0.0) <$> dataFrameStringAt penguins "sex" i) complete
    bodyMass <- mapM (dataFrameNumericAt penguins "body_mass_g") complete
    island <- mapM (dataFrameStringAt penguins "island") complete
    sexDf <- newDataFrame
    dataFrameAddNumericColumn sexDf "is_male" isMale
    dataFrameAddNumericColumn sexDf "body_mass_g" bodyMass
    dataFrameAddStringColumn sexDf "island" island
    newGlmmWithFamily sexDf "is_male ~ body_mass_g + (1 | island)" "binomial" >>= glmmPrintSummary

  putStrLn "\n=================== Poisson mixed model on simulated multi-site count data ==================="
  -- 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
  -- the baseline traffic and the promo's effectiveness vary by store.
  let nStores = 25
      trueIntercept = 2.0
      trueSlope = 0.3
  storeEffect <- replicateM nStores ((* 0.4) <$> gauss)

  rowsByStore <- forM (zip [0 :: Int ..] storeEffect) $ \(s, effect) -> do
    nDays <- (15 +) <$> nextInt rng 0 11
    replicateM nDays $ do
      promoIntensity <- (* 3) <$> uniform
      let lam = exp (trueIntercept + effect + trueSlope * promoIntensity)
          limit = exp (negate lam)
          -- Knuth's Poisson sampler.
          knuth k p = do
            p' <- (p *) <$> uniform
            if p' > limit then knuth (k + 1) p' else pure (k + 1)
      k <- knuth (0 :: Int) 1.0
      pure (fromIntegral s, promoIntensity, fromIntegral (k - 1))
  let (store, promo, visits) = unzip3 (concat rowsByStore)
  df <- newDataFrame
  dataFrameAddNumericColumn df "store" store
  dataFrameAddNumericColumn df "promo" promo
  dataFrameAddNumericColumn df "visits" visits
  storeModel <- newGlmmWithFamily df "visits ~ promo + (1 | store)" "poisson"

  glmmPrintSummary storeModel

  putStrLn ("\nTrue generating values: intercept=" ++ fmt trueIntercept ++ ", slope=" ++ fmt trueSlope
              ++ ", random-intercept SD (log scale)=0.4")

  putStrLn "\n--- BLUPs for a few stores ---"
  labels <- glmmGroupLabels storeModel
  forM_ [0 .. 2] $ \idx -> do
    re <- glmmRandomEffectsForGroup storeModel idx
    putStrLn ("store " ++ labels !! idx ++ ": intercept shift=" ++ fmt (re !! 0))

  putStrLn "\n--- Prediction: population-level vs. store-adjusted ---"
  population <- newDataFrame
  store0 <- newDataFrame
  dataFrameAddNumericColumn population "promo" [1.5]
  dataFrameAddNumericColumn store0 "promo" [1.5]
  dataFrameAddNumericColumn store0 "store" [0.0]
  populationPrediction <- (!! 0) <$> glmmPredict storeModel population
  putStrLn ("promo=1.5, unseen store:   " ++ fmt populationPrediction ++ " expected visits (fixed effects only)")
  storePrediction <- (!! 0) <$> glmmPredict storeModel store0
  putStrLn ("promo=1.5, store 0 (known): " ++ fmt storePrediction ++ " expected visits (fixed effects + store 0's BLUP)")

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
