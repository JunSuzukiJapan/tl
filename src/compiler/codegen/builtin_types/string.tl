
impl String {
    extern fn len(self) -> i64;
    extern fn concat(self, other: String) -> String;
    extern fn char_at(self, index: i64) -> Char;
    extern fn print(self) -> ();
    extern fn to_i64(self) -> i64;
    extern fn display(self) -> ();
    extern fn contains(self, sub: String) -> bool;
    extern fn from_int(val: i64) -> String;
}
