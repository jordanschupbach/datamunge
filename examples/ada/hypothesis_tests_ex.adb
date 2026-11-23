--  CMD: just TARGET=hypothesis_tests_ex run-ada
--  Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Hypothesis_Tests_Ex is
   use type Long_Float_Vector;

   --  Prints a test result, and frees it (each test function returns a new result object).
   procedure Print_Result (Label : String; Result : Hypothesis_Test_Result) is
      R   : Hypothesis_Test_Result := Result;
      Alt : constant String :=
        (case R.Alternative_2 is
            when Less    => "less",
            when Greater => "greater",
            when others  => "two.sided");
   begin
      Put (Label & ": statistic=" & Fmt (R.Statistic));
      if R.Parameter1 /= 0.0 then
         Put (", df1=" & Fmt (R.Parameter1));
      end if;
      if R.Parameter2 /= 0.0 then
         Put (", df2=" & Fmt (R.Parameter2));
      end if;
      Put (", p=" & Fmt (R.P_Value) & " (" & Alt & ")");
      if R.Has_Conf_Int then
         Put (", CI=[" & Fmt (R.Conf_Int_Lower) & ", " & Fmt (R.Conf_Int_Upper) & "]");
      end if;
      Put_Line (" -- " & R.Method);
      Free (R);
   end Print_Result;

   function Column_For_Species (Df : Data_Frame; Column, Species : String) return Long_Float_Vector is
      Values : Long_Float_Vector;
   begin
      for I in 0 .. Df.Nrows - 1 loop
         if Df.String_At ("Species", I) = Species then
            Values.Append (Df.Numeric_At (Column, I));
         end if;
      end loop;
      return Values;
   end Column_For_Species;

   function Column (Df : Data_Frame; Name : String) return Long_Float_Vector is
      Values : Long_Float_Vector;
   begin
      for I in 0 .. Df.Nrows - 1 loop
         Values.Append (Df.Numeric_At (Name, I));
      end loop;
      return Values;
   end Column;

   package Sorting is new Long_Float_Vectors.Generic_Sorting;

   Iris             : Data_Frame := Data.Iris;
   Setosa_Petal     : constant Long_Float_Vector := Column_For_Species (Iris, "Petal.Length", "setosa");
   Versicolor_Petal : constant Long_Float_Vector := Column_For_Species (Iris, "Petal.Length", "versicolor");
   Virginica_Petal  : constant Long_Float_Vector := Column_For_Species (Iris, "Petal.Length", "virginica");
   All_Petal        : constant Long_Float_Vector := Setosa_Petal & Versicolor_Petal & Virginica_Petal;
   Sizes            : constant Long_Long_Integer_Vector :=
     [Long_Long_Integer (Setosa_Petal.Length), Long_Long_Integer (Versicolor_Petal.Length),
      Long_Long_Integer (Virginica_Petal.Length)];
   Sepal_Length     : constant Long_Float_Vector := Column (Iris, "Sepal.Length");
   Petal_Length     : constant Long_Float_Vector := Column (Iris, "Petal.Length");
   Setosa_Sepal     : constant Long_Float_Vector := Column_For_Species (Iris, "Sepal.Length", "setosa");
   Median_All       : Long_Float;

   function Count_Long (V : Long_Float_Vector) return Long_Long_Integer is
      Count : Long_Long_Integer := 0;
   begin
      for X of V loop
         if X > Median_All then
            Count := Count + 1;
         end if;
      end loop;
      return Count;
   end Count_Long;

begin
   Put_Line ("=================== t-tests: petal length, setosa vs. versicolor ===================");
   Print_Result ("Welch two-sample t-test", T_Test_Two_Sample (Setosa_Petal, Versicolor_Petal));
   Print_Result ("Wilcoxon rank-sum test", Wilcoxon_Rank_Sum_Test (Setosa_Petal, Versicolor_Petal));

   Put_Line (ASCII.LF & "=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================");
   Print_Result ("One-way ANOVA", One_Way_Anova (All_Petal, Sizes));
   Print_Result ("Kruskal-Wallis", Kruskal_Wallis_Test (All_Petal, Sizes));

   Put_Line (ASCII.LF & "=================== Correlation: sepal length vs. petal length ===================");
   Print_Result ("Pearson correlation", Pearson_Correlation_Test (Sepal_Length, Petal_Length));
   Print_Result ("Spearman correlation", Spearman_Correlation_Test (Sepal_Length, Petal_Length));

   Put_Line (ASCII.LF & "=================== F-test: petal length variance, setosa vs. virginica ===================");
   Print_Result ("F test", F_Test_Variance (Setosa_Petal, Virginica_Petal));

   Put_Line (ASCII.LF
             & "=================== Normality: is sepal length normally distributed within setosa? ===================");
   Print_Result ("Shapiro-Francia", Shapiro_Francia_Test (Setosa_Sepal));
   Print_Result ("KS vs. fitted normal", Ks_Test_One_Sample_Normal (Setosa_Sepal, 5.006, 0.3525));

   Put_Line (ASCII.LF & "=================== Chi-squared / Fisher: is petal length ""long"" independent of species?"
             & " ===================");
   declare
      Sorted : Long_Float_Vector := All_Petal;
   begin
      Sorting.Sort (Sorted);
      Median_All := Sorted (Natural (Sorted.Length) / 2 + 1);
   end;
   declare
      Setosa_Long      : constant Long_Long_Integer := Count_Long (Setosa_Petal);
      Setosa_Short     : constant Long_Long_Integer := Long_Long_Integer (Setosa_Petal.Length) - Setosa_Long;
      Versicolor_Long  : constant Long_Long_Integer := Count_Long (Versicolor_Petal);
      Versicolor_Short : constant Long_Long_Integer :=
        Long_Long_Integer (Versicolor_Petal.Length) - Versicolor_Long;
   begin
      Put_Line ("table: setosa=[" & Img (Setosa_Long) & "," & Img (Setosa_Short) & "] versicolor=["
                & Img (Versicolor_Long) & "," & Img (Versicolor_Short) & "]");
      Print_Result ("Chi-squared independence", Chi_Squared_Test_Independence
        ([Long_Float (Setosa_Long), Long_Float (Setosa_Short), Long_Float (Versicolor_Long),
          Long_Float (Versicolor_Short)], 2, 2));
      Print_Result ("Fisher's exact test",
                    Fisher_Exact_Test_2x2 (Setosa_Long, Setosa_Short, Versicolor_Long, Versicolor_Short));
   end;

   Put_Line (ASCII.LF & "=================== Proportion / binomial: fraction of ""long"" petals overall"
             & " ===================");
   declare
      Long_Count : constant Long_Long_Integer := Count_Long (All_Petal);
      N          : constant Long_Long_Integer := Long_Long_Integer (All_Petal.Length);
   begin
      Print_Result ("One-sample proportion test (vs 0.5)", Proportion_Test_One_Sample (Long_Count, N, 0.5));
      Print_Result ("Exact binomial test (vs 0.5)", Binomial_Test (Long_Count, N, 0.5));
   end;

   Free (Iris);
end Hypothesis_Tests_Ex;
