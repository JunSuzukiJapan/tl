fn main() {
    println("Running logic syntax test");
}

relation father(String, String);

father("tarou", "ichirou").
father("tarou", "jirou").

?- father("tarou", "ichirou")
