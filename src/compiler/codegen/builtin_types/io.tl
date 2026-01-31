
struct File { handle: i64 }
struct Path { handle: i64 }
struct Env { dummy: i64 }
struct Http { dummy: i64 }

impl File {
    extern fn open(path: String, mode: String) -> File;
    extern fn exists(path: String) -> bool;
    extern fn read(path: String) -> String;
    extern fn write(path: String, content: String) -> bool;
    extern fn download(url: String, path: String) -> bool;
    extern fn read_binary(path: String) -> Vec<u8>;
    
    extern fn read_string(self) -> String;
    extern fn write_string(self, content: String) -> ();
    extern fn close(self) -> ();
    extern fn free(self) -> ();
}

impl Path {
    extern fn new(path: String) -> Path;
    extern fn exists(self) -> bool;
    extern fn join(self, other: String) -> Path;
    extern fn is_dir(self) -> bool;
    extern fn is_file(self) -> bool;
    extern fn to_string(self) -> String;
    extern fn free(self) -> ();
}

impl Env {
    extern fn get(key: String) -> String;
}

impl Http {
    extern fn get(url: String) -> String;
}

extern fn read_line(prompt: String) -> String;
