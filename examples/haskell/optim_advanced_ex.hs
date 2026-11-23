-- CMD: just TARGET=optim_advanced_ex run-haskell
-- Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
-- AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Haskell.
--
-- Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
-- EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
-- ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
-- implement the objective. From Haskell, extend the interface class with a record of overrides
-- for its virtual methods; the optimizer then calls the Haskell overrides. Coordinates are updated
-- in place through an IORef (the C++ parameter is a non-const std::vector<double>&).
module Main (main) where

import Control.Exception (try)
import Control.Monad (forM_)
import Data.IORef (newIORef, readIORef)
import Data.List (isInfixOf)
import Data.Typeable (Typeable, typeOf)
import Datamunge
import Numeric (floatToDigits)

-- | An object's type name, standing in for C#'s default ToString() of the object.
describe :: Typeable a => a -> String
describe = show . typeOf

-- | Newton's method objective: f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
quadratic :: IO HessianFunction
quadratic =
  extendHessianFunction hessianFunctionOverrides
    { hessianFunctionEvaluateOverride = Just $ \_ c -> pure ((c !! 0 - 1) ^ (2 :: Int) + 10 * (c !! 1 + 2) ^ (2 :: Int))
    , hessianFunctionGradientOverride = Just $ \_ c -> pure [2 * (c !! 0 - 1), 20 * (c !! 1 + 2)]
    , hessianFunctionHessianOverride = Just $ \_ _ -> pure [[2.0, 0.0], [0.0, 20.0]]
    }

-- | Residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5, with their Jacobian.
exponential :: [Double] -> IO ResidualFunction
exponential ts =
  extendResidualFunction residualFunctionOverrides
    { residualFunctionResidualsOverride = Just $ \_ p -> pure [p !! 0 * exp (p !! 1 * t) - 2 * exp (0.5 * t) | t <- ts]
    , residualFunctionJacobianOverride = Just $ \_ p -> pure [[exp (p !! 1 * t), p !! 0 * t * exp (p !! 1 * t)] | t <- ts]
    }

-- | Overrides evaluate() only: gradient() and hessian() are left out.
incomplete :: IO HessianFunction
incomplete = extendHessianFunction hessianFunctionOverrides {hessianFunctionEvaluateOverride = Just $ \_ _ -> pure 0.0}

main :: IO ()
main = do
  putStrLn "=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ==================="
  -- Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
  -- then check the acquisition function favors an unexplored region.
  do
    let xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]
        ys = [(x - 1) * (x - 1) | x <- xs]
    surrogate <- newRbfGaussianProcessSurrogateWithNoise 1.0 1e-6
    rbfGaussianProcessSurrogateFit surrogate [[x] | x <- xs] ys
    let incumbent = min 0.0 (minimum ys) -- best observed value so far
    putStrLn ("incumbent (best observed y): " ++ fmt incumbent)
    forM_ [0.25, 1.0, 2.5] $ \x -> do
      acquisition <- rbfGaussianProcessSurrogateAcquisition surrogate [x] incumbent
      putStrLn ("acquisition(x=" ++ fixed 2 x ++ ") = " ++ fixed 6 acquisition)

  putStrLn "\n=================== Options structs + optimizer construction (all 10 new classes) ==================="
  -- Every new Options struct and optimizer class constructs cleanly, proving
  -- the binding for the whole batch is wired correctly.
  pgOptions <- newProximalGradientOptions
  proximalGradientOptionsSetStepSize pgOptions 0.05
  pg <- newProximalGradientWithOptions pgOptions
  putStrLn ("ProximalGradient: " ++ describe pg)
  fista <- newFistaOptions >>= newFistaWithOptions
  putStrLn ("FISTA: " ++ describe fista)
  lmOptions <- newLevenbergMarquardtOptions
  levenbergMarquardtOptionsSetMaxIterations lmOptions 200
  lm <- newLevenbergMarquardtWithOptions lmOptions
  putStrLn ("LevenbergMarquardt: " ++ describe lm)
  newton <- newNewtonOptions >>= newNewtonWithOptions
  putStrLn ("Newton: " ++ describe newton)
  trn <- newTrustRegionNewtonOptions >>= newTrustRegionNewtonWithOptions
  putStrLn ("TrustRegionNewton: " ++ describe trn)
  al <- newAugmentedLagrangianOptions >>= newAugmentedLagrangianWithOptions
  putStrLn ("AugmentedLagrangian: " ++ describe al)
  sqp <- newSqpOptions >>= newSqpWithOptions
  putStrLn ("SQP: " ++ describe sqp)
  ip <- newInteriorPointOptions >>= newInteriorPointWithOptions
  putStrLn ("InteriorPoint: " ++ describe ip)
  boOptions <- newBayesianOptimizationOptions
  bayesianOptimizationOptionsSetInitialSamples boOptions 8
  bo <- newBayesianOptimizationWithOptions boOptions
  putStrLn ("BayesianOptimization: " ++ describe bo)

  putStrLn "\n=================== Haskell objectives (subclassing the C++ interfaces) ==================="
  -- Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
  do
    start <- newIORef [5.0, 5.0]
    value <- quadratic >>= \objective -> newtonOptimize newton objective start
    [x, y] <- readIORef start
    putStrLn ("Newton: f = " ++ general 6 value ++ " at (" ++ fixed 6 x ++ ", " ++ fixed 6 y ++ ") (exact minimum 0 at (1, -2))")

  -- Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
  do
    parameters <- newIORef [1.0, 0.1]
    sse <- exponential [0.0, 0.5, 1.0, 1.5, 2.0] >>= \objective -> levenbergMarquardtOptimize lm objective parameters
    [a, b] <- readIORef parameters
    putStrLn ("LevenbergMarquardt: residual = " ++ general 3 sse ++ ", a = " ++ fixed 6 a ++ ", b = " ++ fixed 6 b ++ " (exact: 2, 0.5)")

  -- An interface method an extension leaves out throws when the optimizer calls it.
  do
    origin <- newIORef [0.0, 0.0]
    outcome <- try (incomplete >>= \objective -> newtonOptimize newton objective origin)
    let (ok, mentions) = case outcome of
          Right _ -> (True, False)
          Left err -> (False, "gradient" `isInfixOf` errorMessage err)
    putStrLn ("missing gradient(): ok=" ++ show ok ++ ", error mentions it: " ++ show mentions)

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

-- | C#'s ToString("F<n>"): the exact value rounded to n decimals, midpoints away from zero.
fixed :: Int -> Double -> String
fixed n x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | otherwise = sign ++ whole ++ (if n > 0 then '.' : frac else "")
  where
    scaled = floor (abs (toRational x) * 10 ^ n + 1 / 2) :: Integer
    sign = if x < 0 || isNegativeZero x then "-" else ""
    padded = let s = show scaled in replicate (n + 1 - length s) '0' ++ s
    (whole, frac) = splitAt (length padded - n) padded

-- | C#'s ToString("G<p>"): the exact value rounded to p significant digits, trailing zeros
-- dropped, in E notation when the exponent is below -4 or at least p (1E-05, 1.23E+06).
general :: Int -> Double -> String
general p x
  | isNaN x = "NaN"
  | isInfinite x = if x < 0 then "-∞" else "∞"
  | x < 0 || isNegativeZero x = '-' : general p (negate x)
  | x == 0 = "0"
  | exp10 < -4 || exp10 >= p =
      let (d, rest) = splitAt 1 digits
          magnitude = show (abs exp10)
       in d ++ (if null rest then "" else '.' : rest) ++ "E" ++ (if exp10 < 0 then "-" else "+")
            ++ replicate (2 - length magnitude) '0' ++ magnitude
  | exp10 < 0 = "0." ++ replicate (negate exp10 - 1) '0' ++ digits
  | exp10 + 1 >= length digits = digits ++ replicate (exp10 + 1 - length digits) '0'
  | otherwise = let (whole, frac) = splitAt (exp10 + 1) digits in whole ++ "." ++ frac
  where
    e0 = snd (floatToDigits 10 x) - 1
    roundAt e = floor (toRational x * 10 ^^ (p - 1 - e) + 1 / 2) :: Integer
    (q, exp10) = let r = roundAt e0 in if r >= 10 ^ p then (r `div` 10, e0 + 1) else (r, e0)
    digits = reverse (dropWhile (== '0') (reverse (show q)))
