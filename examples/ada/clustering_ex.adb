--  CMD: just TARGET=clustering_ex run-ada
--  Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
with Ada.Text_IO; use Ada.Text_IO;
with Datamunge; use Datamunge;
with Dm_Examples; use Dm_Examples;

procedure Clustering_Ex is
   Iris            : Data_Frame := Data.Iris;
   Species         : String_Vector;
   Feature_Columns : constant String_Vector := ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"];

   --  For each cluster, print its size and its majority species (negative labels are noise).
   procedure Report_Purity (Cluster_Labels : Long_Long_Integer_Vector; N_Clusters : Natural) is
      type Votes is record
         Names  : String_Vector;   --  the species in first-seen order
         Counts : Integer_Vector;
      end record;
      All_Votes : array (0 .. N_Clusters - 1) of Votes;
      N         : constant Natural := Natural'Min (Natural (Cluster_Labels.Length), Natural (Species.Length));
   begin
      for I in 1 .. N loop
         if Cluster_Labels.Element (I) >= 0 then
            declare
               V : Votes renames All_Votes (Natural (Cluster_Labels.Element (I)));
               K : constant Natural := V.Names.Find_Index (Species (I));
            begin
               if K = String_Vectors.No_Index then
                  V.Names.Append (Species (I));
                  V.Counts.Append (1);
               else
                  V.Counts (K) := V.Counts (K) + 1;
               end if;
            end;
         end if;
      end loop;
      for C in All_Votes'Range loop
         declare
            V     : Votes renames All_Votes (C);
            Total : Natural := 0;
            Best  : Natural := 0;
         begin
            for J in 1 .. Natural (V.Counts.Length) loop
               Total := Total + V.Counts (J);
               if Best = 0 or else V.Counts (J) > V.Counts (Best) then
                  Best := J;
               end if;
            end loop;
            if Total > 0 then
               Put_Line ("  cluster " & Img (Long_Long_Integer (C)) & ": " & Img (Long_Long_Integer (Total))
                         & " points, majority " & V.Names (Best) & " (" & Img (Long_Long_Integer (V.Counts.Element (Best)))
                         & "/" & Img (Long_Long_Integer (Total)) & ")");
            end if;
         end;
      end loop;
   end Report_Purity;

   function To_Long (Labels : Integer_Vector) return Long_Long_Integer_Vector is
      Result : Long_Long_Integer_Vector;
   begin
      for L of Labels loop
         Result.Append (Long_Long_Integer (L));
      end loop;
      return Result;
   end To_Long;

   K_Model    : K_Means;
   Model      : Agglomerative_Clustering;
   Db_Model   : Dbscan;
begin
   for I in 0 .. Iris.Nrows - 1 loop
      Species.Append (Iris.String_At ("Species", I));
   end loop;

   Put_Line ("=================== K-means (k=3) on iris ===================");
   K_Model := New_K_Means (Iris, Feature_Columns, 3);
   Put_Line ("Inertia: " & Fmt (K_Model.Inertia) & ", iterations (best run): " & Img (K_Model.Iterations_Used));
   Report_Purity (K_Model.Labels, 3);
   Free (K_Model);

   Put_Line (ASCII.LF & "=================== Agglomerative clustering on iris ===================");
   for Linkage of String_Vector'(["ward", "complete", "average"]) loop
      Model := New_Agglomerative_Clustering (Iris, Feature_Columns, 3, Linkage);
      Put_Line ("--- linkage=" & Linkage & " ---");
      Report_Purity (Model.Labels, 3);
      Free (Model);
   end loop;

   Put_Line (ASCII.LF & "--- Re-cutting the ward dendrogram at k=2 without refitting ---");
   Model := New_Agglomerative_Clustering (Iris, Feature_Columns, 3, "ward");
   Report_Purity (Model.Cut (2), 2);
   Free (Model);

   Put_Line (ASCII.LF & "=================== DBSCAN on iris ===================");
   Db_Model := New_Dbscan (Iris, Feature_Columns, 0.6, 5);
   Put_Line ("Clusters found: " & Img (Db_Model.N_Clusters) & ", noise points: " & Img (Db_Model.N_Noise)
             & " (of " & Img (Db_Model.Observations) & ")");
   Report_Purity (To_Long (Db_Model.Labels), Natural (Db_Model.N_Clusters));
   Free (Db_Model);

   Free (Iris);
end Clustering_Ex;
