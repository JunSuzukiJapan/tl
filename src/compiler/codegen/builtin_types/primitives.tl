
struct F32 { dummy: i64 }
struct F64 { dummy: i64 }
struct I64 { dummy: i64 }
struct I32 { dummy: i64 }
struct Bool { dummy: i64 }

impl F32 {
    extern fn abs(self) -> f32;
    extern fn sqrt(self) -> f32;
    extern fn exp(self) -> f32;
    extern fn ln(self) -> f32;
    extern fn sin(self) -> f32;
    extern fn cos(self) -> f32;
    extern fn tan(self) -> f32;
    extern fn powf(self, n: f32) -> f32;
    extern fn pow(self, n: f32) -> f32;
    extern fn powi(self, n: i32) -> f32;
    extern fn max(self, other: f32) -> f32;
    extern fn min(self, other: f32) -> f32;
    extern fn clamp(self, min: f32, max: f32) -> f32;
    extern fn ceil(self) -> f32;
    extern fn floor(self) -> f32;
    extern fn round(self) -> f32;
    extern fn signum(self) -> f32;
}

impl F64 {
    extern fn abs(self) -> f64;
    extern fn sqrt(self) -> f64;
    extern fn exp(self) -> f64;
    extern fn ln(self) -> f64;
    extern fn sin(self) -> f64;
    extern fn cos(self) -> f64;
    extern fn tan(self) -> f64;
    extern fn powf(self, n: f64) -> f64;
    extern fn pow(self, n: f64) -> f64;
    extern fn powi(self, n: i32) -> f64;
}

impl I64 {
    extern fn abs(self) -> i64;
    extern fn signum(self) -> i64;
    extern fn pow(self, exp: i64) -> i64;
}

impl I32 {
    extern fn abs(self) -> i32;
    extern fn signum(self) -> i32;
}
