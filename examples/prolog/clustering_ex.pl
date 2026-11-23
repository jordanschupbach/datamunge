% CMD: just TARGET=clustering_ex run-prolog
% Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
:- use_module(library(datamunge)).
:- initialization(main, main).

% For each cluster, print its size and its majority species (negative labels are noise).
% Ties go to the species seen first in the cluster, as C#'s MaxBy over a Dictionary does.
report_purity(Species, ClusterLabels, NClusters) :-
    Last is NClusters - 1,
    forall(between(0, Last, C),
           ( findall(Sp, (nth0(I, ClusterLabels, C), nth0(I, Species, Sp)), Members),
             length(Members, Total),
             (   Total =:= 0
             ->  true
             ;   list_to_set(Members, Seen),
                 findall(Count-Sp, (member(Sp, Seen), occurrences(Members, Sp, Count)), Votes),
                 best_vote(Votes, Best-BestSpecies),
                 format("  cluster ~w: ~w points, majority ~w (~w/~w)~n", [C, Total, BestSpecies, Best, Total])
             ) )).

occurrences(List, X, Count) :- aggregate_all(count, member(X, List), Count).

best_vote([V|Vs], Best) :- foldl([C-S, C0-S0, B]>>(C > C0 -> B = C-S ; B = C0-S0), Vs, V, Best).

main :-
    data_iris(Iris),
    data_frame_nrows(Iris, N), Last is N - 1,
    findall(S, (between(0, Last, I), data_frame_string_at(Iris, "Species", I, S)), Species),
    FeatureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],

    format("=================== K-means (k=3) on iris ===================~n"),
    k_means_new(Iris, FeatureColumns, 3, KMeans),
    k_means_inertia(KMeans, Inertia), cs(Inertia, InertiaText),
    k_means_iterations_used(KMeans, Iterations),
    format("Inertia: ~w, iterations (best run): ~w~n", [InertiaText, Iterations]),
    k_means_labels(KMeans, KLabels),
    report_purity(Species, KLabels, 3),

    format("~n=================== Agglomerative clustering on iris ===================~n"),
    forall(member(Linkage, ["ward", "complete", "average"]),
           ( agglomerative_clustering_new(Iris, FeatureColumns, 3, Linkage, Model),
             format("--- linkage=~w ---~n", [Linkage]),
             agglomerative_clustering_labels(Model, Labels),
             report_purity(Species, Labels, 3) )),

    format("~n--- Re-cutting the ward dendrogram at k=2 without refitting ---~n"),
    agglomerative_clustering_new(Iris, FeatureColumns, 3, "ward", WardModel),
    agglomerative_clustering_cut(WardModel, 2, WardLabels),
    report_purity(Species, WardLabels, 2),

    format("~n=================== DBSCAN on iris ===================~n"),
    dbscan_new(Iris, FeatureColumns, 0.6, 5, Dbscan),
    dbscan_n_clusters(Dbscan, NClusters), dbscan_n_noise(Dbscan, NNoise), dbscan_observations(Dbscan, Obs),
    format("Clusters found: ~w, noise points: ~w (of ~w)~n", [NClusters, NNoise, Obs]),
    dbscan_labels(Dbscan, DLabels),
    report_purity(Species, DLabels, NClusters).

% cs(+Number, -String): Number as C#'s default ToString() prints it: integers as they are,
% doubles in their shortest round-trip digits, in E notation when the decimal exponent is below
% -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
cs(X, S) :- integer(X), !, number_string(X, S).
cs(X, S) :- float_class(X, nan), !, S = "NaN".
cs(X, S) :- float_class(X, infinite), !, ( X < 0 -> S = "-∞" ; S = "∞" ).
cs(X, S) :-
    format(codes(C0), "~w", [X]),
    (   C0 = [0'-|C1] -> Sign = "-" ; C1 = C0, Sign = "" ),
    (   append(M, [0'e|E], C1) -> number_codes(Exp, E) ; M = C1, Exp = 0 ),
    (   append(IP, [0'.|FP], M) -> true ; IP = M, FP = [] ),
    append(IP, FP, D0),
    length(IP, L0),
    P0 is L0 + Exp,
    strip_zeros(D0, P0, D1, Point),
    reverse(D1, R1), strip_trailing(R1, R2), reverse(R2, Digits),
    cs_digits(Digits, Point, Sign, S).

strip_zeros([0'0|T], P0, D, P) :- !, P1 is P0 - 1, strip_zeros(T, P1, D, P).
strip_zeros(D, P, D, P).
strip_trailing([0'0|T], R) :- !, strip_trailing(T, R).
strip_trailing(R, R).

cs_digits([], _, Sign, S) :- !, string_concat(Sign, "0", S).
cs_digits([D|Rest], Point, Sign, S) :-
    Exp10 is Point - 1,
    (   ( Exp10 < -4 ; Exp10 >= 15 )
    ->  ( Rest == [] -> Frac = "" ; string_codes(R, Rest), string_concat(".", R, Frac) ),
        ( Exp10 < 0 -> ES = "-" ; ES = "+" ),
        A is abs(Exp10),
        format(string(S), "~w~c~wE~w~|~`0t~d~2+", [Sign, D, Frac, ES, A])
    ;   Point =< 0
    ->  Z is -Point, length(Zs, Z), maplist(=(0'0), Zs),
        format(string(S), "~w0.~s~s", [Sign, Zs, [D|Rest]])
    ;   length([D|Rest], N), Point >= N
    ->  Z is Point - N, length(Zs, Z), maplist(=(0'0), Zs),
        format(string(S), "~w~s~s", [Sign, [D|Rest], Zs])
    ;   length(Int, Point), append(Int, Frac, [D|Rest]),
        format(string(S), "~w~s.~s", [Sign, Int, Frac])
    ).
