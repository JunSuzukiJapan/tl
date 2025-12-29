
relation edge(src: i64, dst: i64);
relation path(src: i64, dst: i64);

path(x, y) :- edge(x, y);
path(x, z) :- edge(x, y), path(y, z);

?- path(1, 2);
