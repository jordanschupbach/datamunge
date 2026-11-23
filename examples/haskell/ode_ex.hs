-- CMD: just TARGET=ode_ex run-haskell
-- ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
module Main (main) where

import Control.Monad (forM_)
import Data.List (intercalate)
import Datamunge
import Numeric (floatToDigits)

-- | Approximates C's %g: six significant digits, no padding.
fmtG :: Double -> String
fmtG = general 6

lastState :: OdeSolution -> IO [Double]
lastState sol = do
  size <- odeSolutionSize sol
  odeSolutionStateAt sol (size - 1)

-- | The time vector followed by the requested state components (0-based), one value per step.
solutionSeries :: OdeSolution -> [Int] -> IO ([Double], [[Double]])
solutionSeries sol components = do
  size <- odeSolutionSize sol
  let steps = [0 .. size - 1]
  states <- mapM (odeSolutionStateAt sol) steps
  times <- mapM (odeSolutionTimeAt sol) steps
  pure (times, [map (!! c) states | c <- components])

rk4Solver :: Double -> IO OdeSolver
rk4Solver step = do
  options <- newOdeOptions
  odeOptionsSetMethod options StepMethodRk4
  odeOptionsSetStepSize options step
  newOdeSolverWithOptions options

-- | C#'s {value,-n}: left-aligned in a field of n characters.
padRight :: Int -> String -> String
padRight n s = s ++ replicate (n - length s) ' '

-- NOTE: like the Lua version this example uses the built-in named systems via solveBuiltin.

main :: IO ()
main = do
  putStrLn "=================== Every named built-in system (no director needed) ==================="
  do
    solver <- newOdeSolver
    let systems =
          [ ("exponential_decay", [1.0], [1.0])
          , ("logistic_growth", [1.0, 1.0], [0.5])
          , ("harmonic_oscillator", [1.0], [1.0, 0.0])
          , ("van_der_pol", [1.0], [2.0, 0.0])
          , ("lorenz", [10.0, 28.0, 8.0 / 3], [1.0, 1.0, 1.0])
          ]
    forM_ systems $ \(name, parameters, y0) -> do
      sol <- odeSolverSolveBuiltin solver name parameters y0 0.0 1.0
      steps <- odeSolutionStepsTaken sol
      final <- lastState sol
      putStrLn (padRight 22 name ++ " steps=" ++ padRight 6 (show steps) ++ " final state=[" ++ intercalate ", " (map fmtG final) ++ "]")

  putStrLn "\n=================== Harmonic oscillator (energy conservation) ==================="
  do
    sol <- rk4Solver 0.01 >>= \solver -> odeSolverSolveBuiltin solver "harmonic_oscillator" [1.0] [1.0, 0.0] 0.0 20.0
    final <- lastState sol
    let (x, v) = (final !! 0, final !! 1)
    putStrLn ("x(20) = " ++ fixed 6 x ++ " (cos(20) = " ++ fixed 6 (cos 20.0) ++ ")")
    putStrLn ("energy x^2+v^2 = " ++ fixed 6 (x * x + v * v) ++ " (should stay near 1.0)")
    (ts, series) <- solutionSeries sol [0, 1]
    p <- rPlotPlotWithLabel ts (series !! 0) "l" "x(t)"
    _ <- rPlotLinesWithLabel p ts (series !! 1) "v(t)"
    _ <- rPlotTitle p "Harmonic Oscillator" >>= (`rPlotXLabel` "t") >>= (`rPlotYLabel` "state")
    plotSaveSvg p "ode_harmonic_oscillator_haskell.svg"
    putStrLn "wrote ode_harmonic_oscillator_haskell.svg"

  putStrLn "\n=================== Lorenz attractor (phase plane) ==================="
  do
    sol <- rk4Solver 0.005 >>= \solver -> odeSolverSolveBuiltin solver "lorenz" [10.0, 28.0, 8.0 / 3] [1.0, 1.0, 1.0] 0.0 25.0
    steps <- odeSolutionStepsTaken sol
    putStrLn ("steps_taken = " ++ show steps)
    (_, series) <- solutionSeries sol [0, 2]
    p <- rPlotPlotWithLabel (series !! 0) (series !! 1) "l" "trajectory"
    _ <- rPlotTitle p "Lorenz Attractor (x-z phase plane)" >>= (`rPlotXLabel` "x") >>= (`rPlotYLabel` "z")
    plotSaveSvg p "ode_lorenz_phase_plane_haskell.svg"
    putStrLn "wrote ode_lorenz_phase_plane_haskell.svg"

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
