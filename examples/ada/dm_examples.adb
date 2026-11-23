with Ada.Strings.Fixed;
with Ada.Strings.Unbounded;
with Ada.Text_IO;
with Interfaces.C;
with System;

package body Dm_Examples is

   use Interfaces;
   use type Interfaces.C.int;

   function C_Snprintf
     (Buffer : System.Address; Size : C.size_t; Format : C.char_array; X : C.double) return C.int
     with Import, Convention => C_Variadic_3, External_Name => "snprintf";

   function C_Strtod (S : C.char_array; End_Ptr : System.Address) return C.double
     with Import, Convention => C, External_Name => "strtod";

   function C_Fflush (Stream : System.Address) return C.int
     with Import, Convention => C, External_Name => "fflush";

   --  X through a C printf format with one double argument.
   function C_Format (Format : String; X : Long_Float) return String is
      Buffer : C.char_array (0 .. 399) := [others => C.nul];
      N      : constant C.int :=
        C_Snprintf (Buffer'Address, Buffer'Length, C.To_C (Format), C.double (X));
   begin
      if N < 0 then
         return "";
      end if;
      return C.To_Ada (Buffer);
   end C_Format;

   function Img (N : Long_Long_Integer) return String is
     (Ada.Strings.Fixed.Trim (N'Image, Ada.Strings.Left));

   function Join (V : Datamunge.Long_Float_Vector; Separator : String := ", ") return String is
      Result : Ada.Strings.Unbounded.Unbounded_String;
   begin
      for I in V.First_Index .. V.Last_Index loop
         if I > V.First_Index then
            Ada.Strings.Unbounded.Append (Result, Separator);
         end if;
         Ada.Strings.Unbounded.Append (Result, Fmt (V (I)));
      end loop;
      return Ada.Strings.Unbounded.To_String (Result);
   end Join;

   function Pad_Left (S : String; Width : Natural) return String is
     ((if S'Length >= Width then S else [1 .. Width - S'Length => ' '] & S));

   function Pad_Right (S : String; Width : Natural) return String is
     ((if S'Length >= Width then S else S & [1 .. Width - S'Length => ' ']));

   function Is_Nan (X : Long_Float) return Boolean is (X /= X);

   function Is_Inf (X : Long_Float) return Boolean is
     (not Is_Nan (X) and then (X > Long_Float'Last or else X < Long_Float'First));

   --  Formats the significant digits Digits (no trailing zeros) with decimal exponent E as C# does.
   function Layout (Negative : Boolean; Digs : String; E : Integer; Sci : Boolean) return String is
      Sign : constant String := (if Negative then "-" else "");
      Mag  : constant String := Img (Long_Long_Integer (abs E));
   begin
      if Sci then
         return Sign & Digs (Digs'First)
           & (if Digs'Length > 1 then "." & Digs (Digs'First + 1 .. Digs'Last) else "")
           & (if E < 0 then "E-" else "E+") & (if Mag'Length < 2 then "0" & Mag else Mag);
      elsif E < 0 then
         return Sign & "0." & [1 .. -E - 1 => '0'] & Digs;
      elsif Digs'Length <= E + 1 then
         return Sign & Digs & [1 .. E + 1 - Digs'Length => '0'];
      else
         return Sign & Digs (Digs'First .. Digs'First + E) & "."
           & Digs (Digs'First + E + 1 .. Digs'Last);
      end if;
   end Layout;

   --  Splits a "%.Ne" string into its significant digits (trailing zeros removed) and exponent.
   procedure Split (S : String; Digs : out String; Last : out Natural; E : out Integer) is
      P : constant Natural := Ada.Strings.Fixed.Index (S, "e");
   begin
      Last := Digs'First - 1;
      for C of S (S'First .. P - 1) loop
         if C in '0' .. '9' then
            Last := Last + 1;
            Digs (Last) := C;
         end if;
      end loop;
      while Last > Digs'First and then Digs (Last) = '0' loop
         Last := Last - 1;
      end loop;
      E := Integer'Value (S (P + 1 .. S'Last));
   end Split;

   function Special (X : Long_Float) return String is
     ((if Is_Nan (X) then "NaN" elsif X > 0.0 then "∞" else "-∞"));

   function Fmt (X : Long_Float) return String is
   begin
      if Is_Nan (X) or else Is_Inf (X) then
         return Special (X);
      elsif X = 0.0 then
         return (if Long_Float'Copy_Sign (1.0, X) < 0.0 then "-0" else "0");
      end if;
      for Precision in 1 .. 17 loop
         declare
            S : constant String := C_Format ("%." & Img (Long_Long_Integer (Precision - 1)) & "e", X);
         begin
            if Precision = 17 or else Long_Float (C_Strtod (C.To_C (S), System.Null_Address)) = X then
               declare
                  Digs : String (1 .. 20);
                  Last : Natural;
                  E    : Integer;
               begin
                  Split (S, Digs, Last, E);
                  return Layout (X < 0.0, Digs (1 .. Last), E, E < -4 or else E >= 15);
               end;
            end if;
         end;
      end loop;
      return "";
   end Fmt;

   function F (X : Long_Float; Decimals : Natural) return String is
     ((if Is_Nan (X) or else Is_Inf (X) then Special (X)
       else C_Format ("%." & Img (Long_Long_Integer (Decimals)) & "f", X)));

   function G (X : Long_Float; Precision : Positive) return String is
   begin
      if Is_Nan (X) or else Is_Inf (X) then
         return Special (X);
      elsif X = 0.0 then
         return (if Long_Float'Copy_Sign (1.0, X) < 0.0 then "-0" else "0");
      end if;
      declare
         S    : constant String := C_Format ("%." & Img (Long_Long_Integer (Precision - 1)) & "e", X);
         Digs : String (1 .. 40);
         Last : Natural;
         E    : Integer;
      begin
         Split (S, Digs, Last, E);
         return Layout (X < 0.0, Digs (1 .. Last), E, E < -4 or else E >= Precision);
      end;
   end G;

   procedure Flush is
      C_Stdout : System.Address
        with Import, Convention => C, External_Name => "stdout";
      Ignored : C.int;
   begin
      Ada.Text_IO.Flush;
      Ignored := C_Fflush (C_Stdout);
   end Flush;

   M_Big  : constant Integer_32 := Integer_32'Last;
   M_Seed : constant Integer_32 := 161_803_398;

   function New_Dot_Net_Random (Seed : Integer_32) return Dot_Net_Random is
      R      : Dot_Net_Random;
      Mj, Mk : Integer_32;
      Ii     : Integer;
   begin
      Mj := (if Seed = Integer_32'First then M_Seed - M_Big else M_Seed - abs Seed);
      R.Seeds (55) := Mj;
      Mk := 1;
      for I in 1 .. 54 loop
         Ii := (21 * I) mod 55;
         R.Seeds (Ii) := Mk;
         Mk := Mj - Mk;
         if Mk < 0 then
            Mk := Mk + M_Big;
         end if;
         Mj := R.Seeds (Ii);
      end loop;
      for K in 1 .. 4 loop
         for I in 1 .. 55 loop
            R.Seeds (I) := R.Seeds (I) - R.Seeds (1 + (I + 30) mod 55);
            if R.Seeds (I) < 0 then
               R.Seeds (I) := R.Seeds (I) + M_Big;
            end if;
         end loop;
      end loop;
      R.Inext := 0;
      R.Inextp := 21;
      return R;
   end New_Dot_Net_Random;

   function Internal_Sample (Self : in out Dot_Net_Random) return Integer_32 is
      Result : Integer_32;
   begin
      Self.Inext := (if Self.Inext + 1 >= 56 then 1 else Self.Inext + 1);
      Self.Inextp := (if Self.Inextp + 1 >= 56 then 1 else Self.Inextp + 1);
      Result := Self.Seeds (Self.Inext) - Self.Seeds (Self.Inextp);
      if Result = M_Big then
         Result := Result - 1;
      end if;
      if Result < 0 then
         Result := Result + M_Big;
      end if;
      Self.Seeds (Self.Inext) := Result;
      return Result;
   end Internal_Sample;

   function Next_Double (Self : in out Dot_Net_Random) return Long_Float is
      Scale : constant Long_Float := 1.0 / Long_Float (M_Big);
   begin
      return Long_Float (Internal_Sample (Self)) * Scale;
   end Next_Double;

   function Next (Self : in out Dot_Net_Random; Min_Value, Max_Value : Integer_32)
     return Integer_32 is
     (Integer_32 (Long_Float'Truncation
        (Next_Double (Self) * Long_Float (Integer_64 (Max_Value) - Integer_64 (Min_Value))))
      + Min_Value);

end Dm_Examples;
