--  CMD: just TARGET=mds_ex run-ada
--  Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Mds_Ex is
   Iris            : Data_Frame := Data.Iris;
   Feature_Columns : constant String_Vector := ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];
   Species         : String_Vector;
   Model, Manhattan_Model : Mds;
   Table           : Data_Frame;
   Plot            : R_Plot;
begin
   for I in 0 .. Iris.Nrows - 1 loop
      Species.Append (Iris.String_At ("Species", I));
   end loop;

   Put_Line ("=================== Classical MDS on iris (euclidean) ===================");
   Model := New_Mds (Iris, Feature_Columns, 2, "euclidean");
   Model.Print_Summary;

   Table := Model.Embedding_Frame;
   Put_Line (ASCII.LF & "Embedding as a DataFrame:" & ASCII.LF & Table.Image);
   Free (Table);

   Plot := Model.Plot_Embedding_Grouped (Species);
   Plot.Save_Svg ("mds_iris_embedding.svg");
   Free (Plot);
   Put_Line (ASCII.LF & "Embedding scatter (colored by species) saved as mds_iris_embedding.svg");

   Put_Line (ASCII.LF & "=================== Classical MDS on iris (manhattan) ===================");
   Manhattan_Model := New_Mds (Iris, Feature_Columns, 2, "manhattan");
   Put_Line ("Goodness of fit: euclidean=" & F (Model.Goodness_Of_Fit, 4)
             & ", manhattan=" & F (Manhattan_Model.Goodness_Of_Fit, 4));

   Free (Manhattan_Model);
   Free (Model);
   Free (Iris);
end Mds_Ex;
