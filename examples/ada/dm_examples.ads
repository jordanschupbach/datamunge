--  Helpers shared by the Ada ports of the datamunge examples: C#-compatible number formatting,
--  .NET's seeded System.Random, and the C library's math functions (which .NET calls on Linux).
with Datamunge;
with Interfaces;

package Dm_Examples is

   pragma Linker_Options ("-lm");

   function Fmt (X : Long_Float) return String;
   --  X as C#'s default ToString() prints it: the shortest round-trip digits, in E notation when
   --  the decimal exponent is below -4 or at least 15 ("6", "0.1", "1E-05", "1E+15").

   function F (X : Long_Float; Decimals : Natural) return String;
   --  C#'s fixed-point format {X:F<Decimals>} (printf "%.<Decimals>f").

   function G (X : Long_Float; Precision : Positive) return String;
   --  C#'s general format {X:G<Precision>}: Precision significant digits, trailing zeros removed,
   --  E notation (E-05, E+15) when the exponent is below -4 or at least Precision.

   function Join (V : Datamunge.Long_Float_Vector; Separator : String := ", ") return String;
   --  The elements of V formatted with Fmt, like C#'s string.Join(Separator, V).

   function Img (N : Long_Long_Integer) return String;
   --  N without Ada's leading space.

   function Pad_Left (S : String; Width : Natural) return String;
   function Pad_Right (S : String; Width : Natural) return String;
   --  S padded with spaces to Width characters, like C#'s {S,Width} and {S,-Width}.

   function C_Log (X : Long_Float) return Long_Float
     with Import, Convention => C, External_Name => "log";
   function C_Exp (X : Long_Float) return Long_Float
     with Import, Convention => C, External_Name => "exp";
   function C_Cos (X : Long_Float) return Long_Float
     with Import, Convention => C, External_Name => "cos";
   function C_Sin (X : Long_Float) return Long_Float
     with Import, Convention => C, External_Name => "sin";
   function C_Pow (X, Y : Long_Float) return Long_Float
     with Import, Convention => C, External_Name => "pow";
   function C_Sqrt (X : Long_Float) return Long_Float
     with Import, Convention => C, External_Name => "sqrt";
   --  The C library's functions, which .NET's Math class calls on Linux, so the results are
   --  bitwise the same as the C# example's.

   Pi : constant Long_Float := 3.14159265358979323846264338327950288;
   --  Math.PI.

   procedure Flush;
   --  Flushes Ada's standard output and the C library's stdout, before and after C++ prints.

   type Dot_Net_Random is tagged private;
   --  .NET's System.Random(Seed): the seeded Knuth subtractive generator, so simulated data match
   --  the C# examples exactly.

   function New_Dot_Net_Random (Seed : Interfaces.Integer_32) return Dot_Net_Random;
   function Next_Double (Self : in out Dot_Net_Random) return Long_Float;
   --  Random.NextDouble(): a value in [0, 1).
   function Next (Self : in out Dot_Net_Random; Min_Value, Max_Value : Interfaces.Integer_32)
     return Interfaces.Integer_32;
   --  Random.Next(Min_Value, Max_Value): an integer in [Min_Value, Max_Value).

private

   type Seed_Array is array (0 .. 55) of Interfaces.Integer_32;

   type Dot_Net_Random is tagged record
      Seeds  : Seed_Array := [others => 0];
      Inext  : Integer := 0;
      Inextp : Integer := 21;
   end record;

end Dm_Examples;
