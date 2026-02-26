// SKIP: pure Datalog (?- queries without fn main) causes heap corruption — separate issue
edge(1, 2).
edge(2, 3).
edge(3, 4).
path(x, y) :- edge(x, y);
path(x, z) :- edge(x, y), path(y, z);
?- path(1, 2);
?- path(1, 3);
?- path(1, 4);
