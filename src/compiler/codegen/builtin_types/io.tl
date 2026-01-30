
struct File { handle: i64 }
struct Path { handle: i64 }

impl File {
    extern fn read_string() -> String;
    extern fn write_string(s: String) -> ();
    extern fn read_to_end() -> Vec<u8>;
    extern fn write(data: Vec<u8>) -> ();
    extern fn close() -> ();
    extern fn free() -> ();
}

impl Path {
    extern fn join(self, other: String) -> Path;
    extern fn exists(self) -> bool;
    extern fn is_dir(self) -> bool;
    extern fn is_file(self) -> bool;
    extern fn to_string(self) -> String;
    extern fn free(self) -> ();
}
