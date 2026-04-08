struct MyError {
    msg: String,
}

impl Error for MyError {
    fn source(self) -> Option<dyn Error> {
        let none: Option<dyn Error> = Option::None();
        none
    }
}

struct WrappedError {
    inner: MyError,
}

impl Error for WrappedError {
    fn source(self) -> Option<dyn Error> {
        // Return inner wrapped in Option
        let some: Option<dyn Error> = Option::Some(self.inner as dyn Error);
        some
    }
}

fn main() {
    let err = MyError { msg: "Something went wrong" };
    let src = err.source();
    
    if src.is_none() {
        print("source is none");
    } else {
        print("source is some");
    }

    let wrap = WrappedError { inner: MyError { msg: "inner" } };
    let src2 = wrap.source();
    if src2.is_some() {
        print("wrapped source is some");
    } else {
        print("wrapped source is none");
    }
}
