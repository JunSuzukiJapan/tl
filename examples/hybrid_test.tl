let data = [10.0, 20.0, 30.0, 5.0];

index(0).
index(1).
index(2).
index(3).

small_idx(i) :- index(i), lt(data[i], 15.0)

?- small_idx(0)
?- small_idx(1)
?- small_idx(2)
?- small_idx(3)
?- small_idx(X)
