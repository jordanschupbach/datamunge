--  CMD: just TARGET=timeseries_ex run-ada
--  Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
with Ada.Text_IO; use Ada.Text_IO;
with Interfaces;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Timeseries_Ex is
   Rng : Dot_Net_Random := New_Dot_Net_Random (42);

   procedure Seed (N : Integer) is
   begin
      Rng := New_Dot_Net_Random (Interfaces.Integer_32 (N));
   end Seed;

   --  (0, 1]: avoids log(0) in the Box-Muller transform.
   function Uniform return Long_Float is (1.0 - Rng.Next_Double);

   function Gauss return Long_Float is
      U1 : constant Long_Float := Uniform;
      U2 : constant Long_Float := Uniform;
   begin
      return C_Sqrt (-2.0 * C_Log (U1)) * C_Cos (2.0 * Pi * U2);
   end Gauss;

   Y, S, Y2 : Long_Float_Vector;
   Options   : Arima_Options;
   Model     : Arima;
   Es_Options : Exponential_Smoothing_Options;
   Es_Model  : Exponential_Smoothing;
   Intervals : Long_Float_Vector_Long_Float_Vector_Pair;
   Seasonal_Shape : constant Long_Float_Vector := [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6];
   Flat      : constant Long_Float_Vector := [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7];
begin
   Put_Line ("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================");
   Seed (42);
   declare
      Level      : Long_Float := 100.0;
      Prev_Shock : Long_Float := 0.0;
      Shock      : Long_Float;
   begin
      for I in 0 .. 149 loop
         Shock := Gauss;
         Level := Level + (0.3 + 0.4 * Prev_Shock + Shock);  --  integrated ARMA(0,1) increments with drift
         Y.Append (Level);
         Prev_Shock := Shock;
      end loop;
   end;

   Options := New_Arima_Options;
   Options.Set_P (1);
   Options.Set_D (1);
   Options.Set_Q (1);
   Options.Set_De_Population_Size (80);
   Options.Set_De_Max_Generations (400);
   Model := New_Arima (Y, Options);
   Free (Options);

   Put_Line ("AR coefficient: " & Fmt (Model.Ar_Coefficients.First_Element));
   Put_Line ("MA coefficient: " & Fmt (Model.Ma_Coefficients.First_Element));
   Put_Line ("sigma^2: " & Fmt (Model.Sigma2) & ", AIC: " & Fmt (Model.Aic) & ", BIC: " & Fmt (Model.Bic));

   Intervals := Model.Forecast_With_Intervals (6);
   Put_Line ("6-step forecast: " & Join (Intervals.First));
   Put_Line ("forecast std. errors: " & Join (Intervals.Second));
   Free (Model);

   Put_Line (ASCII.LF & "=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================");
   Seed (7);
   declare
      Prev : Long_Float := 0.0;
   begin
      for I in 0 .. 119 loop
         Prev := 0.5 * Prev + Gauss;
         S.Append (20.0 + 0.2 * Long_Float (I) + 5.0 * C_Sin (2.0 * Pi * Long_Float (I) / 12.0) + Prev);
      end loop;
   end;

   Options := New_Arima_Options;
   Options.Set_P (1);
   Options.Set_Seasonal_P (1);
   Options.Set_Seasonal_D (1);
   Options.Set_Seasonal_Period (12);
   Options.Set_De_Population_Size (100);
   Options.Set_De_Max_Generations (500);
   Model := New_Arima (S, Options);
   Free (Options);
   Put_Line ("AR coefficient: " & Fmt (Model.Ar_Coefficients.First_Element) & ", seasonal AR coefficient: "
             & Fmt (Model.Seasonal_Ar_Coefficients.First_Element));
   Put_Line ("12-step forecast: " & Join (Model.Forecast (12)));
   Free (Model);

   Put_Line (ASCII.LF & "=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================");
   Seed (11);
   for I in 0 .. 47 loop
      Y2.Append ((100.0 + 2.0 * Long_Float (I)) * Seasonal_Shape (I mod 12 + 1) + Gauss * 3.0);
   end loop;

   Es_Options := New_Exponential_Smoothing_Options;
   Es_Options.Set_Trend (Additive);
   Es_Options.Set_Seasonal (Multiplicative);
   Es_Options.Set_Seasonal_Period (12);
   Es_Options.Set_De_Population_Size (60);
   Es_Options.Set_De_Max_Generations (300);
   Es_Model := New_Exponential_Smoothing (Y2, Es_Options);
   Free (Es_Options);
   Put_Line ("alpha=" & Fmt (Es_Model.Alpha) & " beta=" & Fmt (Es_Model.Beta) & " gamma=" & Fmt (Es_Model.Gamma));
   Put_Line ("sigma^2: " & Fmt (Es_Model.Sigma2) & ", AIC: " & Fmt (Es_Model.Aic));
   Put_Line ("12-month forecast: " & Join (Es_Model.Forecast (12)));
   Free (Es_Model);

   Put_Line (ASCII.LF & "=================== Simple exponential smoothing vs. Holt's linear trend ===================");
   Es_Options := New_Exponential_Smoothing_Options;
   Es_Model := New_Exponential_Smoothing (Flat, Es_Options);
   Free (Es_Options);
   Put_Line ("SES alpha= " & Fmt (Es_Model.Alpha));
   Put_Line ("SES 5-step forecast: " & Join (Es_Model.Forecast (5)));
   Free (Es_Model);

   Es_Options := New_Exponential_Smoothing_Options;
   Es_Options.Set_Trend (Additive);
   Es_Model := New_Exponential_Smoothing (Flat, Es_Options);
   Free (Es_Options);
   Put_Line ("Holt alpha=" & Fmt (Es_Model.Alpha) & " beta=" & Fmt (Es_Model.Beta));
   Put_Line ("Holt 5-step forecast: " & Join (Es_Model.Forecast (5)));
   Free (Es_Model);
end Timeseries_Ex;
