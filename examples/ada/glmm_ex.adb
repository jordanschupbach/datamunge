--  CMD: just TARGET=glmm_ex run-ada
--  Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
with Ada.Text_IO; use Ada.Text_IO;
with Interfaces;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Glmm_Ex is
   use type Interfaces.Integer_32;
   Rng : Dot_Net_Random := New_Dot_Net_Random (4242);

   function Uniform return Long_Float is (Rng.Next_Double);

   function Gauss return Long_Float is
      U1 : constant Long_Float := 1.0 - Uniform;  -- 1 - u keeps log() away from 0
      U2 : constant Long_Float := Uniform;
   begin
      return C_Sqrt (-2.0 * C_Log (U1)) * C_Cos (2.0 * Pi * U2);
   end Gauss;

   N_Stores       : constant := 25;
   True_Intercept : constant Long_Float := 2.0;
   True_Slope     : constant Long_Float := 0.3;
   Store_Effect   : array (0 .. N_Stores - 1) of Long_Float;
   Store, Promo, Visits : Long_Float_Vector;
   Df, Population, Store0 : Data_Frame;
   Store_Model    : Glmm;
begin
   Put_Line ("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================");
   --  Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
   --  confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
   --  is not tied to location, so this is a well-behaved fit.
   declare
      Penguins           : Data_Frame := Data.Penguins;
      Sex_Df             : Data_Frame := New_Data_Frame;
      Is_Male, Body_Mass : Long_Float_Vector;
      Island             : String_Vector;
      Sex_Model          : Glmm;
   begin
      for I in 0 .. Penguins.Nrows - 1 loop
         if not (Penguins.Is_Null ("sex", I) or else Penguins.Is_Null ("body_mass_g", I)
                 or else Penguins.Is_Null ("island", I))
         then
            Is_Male.Append ((if Penguins.String_At ("sex", I) = "male" then 1.0 else 0.0));
            Body_Mass.Append (Penguins.Numeric_At ("body_mass_g", I));
            Island.Append (Penguins.String_At ("island", I));
         end if;
      end loop;
      Sex_Df.Add_Numeric_Column ("is_male", Is_Male);
      Sex_Df.Add_Numeric_Column ("body_mass_g", Body_Mass);
      Sex_Df.Add_String_Column ("island", Island);
      Sex_Model := New_Glmm (Sex_Df, "is_male ~ body_mass_g + (1 | island)", "binomial");
      Sex_Model.Print_Summary;
      Free (Sex_Model);
      Free (Sex_Df);
      Free (Penguins);
   end;

   Put_Line (ASCII.LF & "=================== Poisson mixed model on simulated multi-site count data ===================");
   --  25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
   --  the baseline traffic and the promo's effectiveness vary by store.
   for S in Store_Effect'Range loop
      Store_Effect (S) := Gauss * 0.4;
   end loop;

   for S in Store_Effect'Range loop
      declare
         N_Days : constant Interfaces.Integer_32 := 15 + Rng.Next (0, 11);
      begin
         for Day in 1 .. N_Days loop
            declare
               Promo_Intensity : constant Long_Float := Uniform * 3.0;
               Lam             : constant Long_Float :=
                 C_Exp (True_Intercept + Store_Effect (S) + True_Slope * Promo_Intensity);
               --  Knuth's Poisson sampler.
               Threshold       : constant Long_Float := C_Exp (-Lam);
               K               : Natural := 0;
               P               : Long_Float := 1.0;
            begin
               loop
                  K := K + 1;
                  P := P * Uniform;
                  exit when not (P > Threshold);
               end loop;
               Store.Append (Long_Float (S));
               Promo.Append (Promo_Intensity);
               Visits.Append (Long_Float (K - 1));
            end;
         end loop;
      end;
   end loop;
   Df := New_Data_Frame;
   Df.Add_Numeric_Column ("store", Store);
   Df.Add_Numeric_Column ("promo", Promo);
   Df.Add_Numeric_Column ("visits", Visits);
   Store_Model := New_Glmm (Df, "visits ~ promo + (1 | store)", "poisson");

   Store_Model.Print_Summary;

   Put_Line (ASCII.LF & "True generating values: intercept=" & Fmt (True_Intercept) & ", slope=" & Fmt (True_Slope)
             & ", random-intercept SD (log scale)=0.4");

   Put_Line (ASCII.LF & "--- BLUPs for a few stores ---");
   declare
      Labels : constant String_Vector := Store_Model.Group_Labels;
   begin
      for Idx in 0 .. 2 loop
         Put_Line ("store " & Labels (Idx + 1) & ": intercept shift="
                   & Fmt (Store_Model.Random_Effects_For_Group (Long_Long_Integer (Idx)) (1)));
      end loop;
   end;

   Put_Line (ASCII.LF & "--- Prediction: population-level vs. store-adjusted ---");
   Population := New_Data_Frame;
   Store0 := New_Data_Frame;
   Population.Add_Numeric_Column ("promo", [1.5]);
   Store0.Add_Numeric_Column ("promo", [1.5]);
   Store0.Add_Numeric_Column ("store", [0.0]);
   Put_Line ("promo=1.5, unseen store:   " & Fmt (Store_Model.Predict (Population) (1))
             & " expected visits (fixed effects only)");
   Put_Line ("promo=1.5, store 0 (known): " & Fmt (Store_Model.Predict (Store0) (1))
             & " expected visits (fixed effects + store 0's BLUP)");

   Free (Store_Model);
   Free (Store0);
   Free (Population);
   Free (Df);
end Glmm_Ex;
