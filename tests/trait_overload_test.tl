struct Complex {
    real: f64,
    imag: f64,
}

impl Display for Complex {
    fn to_string(self) -> String {
        return format("{} + {}i", self.real, self.imag);
    }
}

impl Add<Complex> for Complex {
    fn add(self, rhs: Complex) -> Complex {
        return Complex {
            real: self.real + rhs.real,
            imag: self.imag + rhs.imag,
        };
    }
}

impl Sub<Complex> for Complex {
    fn sub(self, rhs: Complex) -> Complex {
        return Complex {
            real: self.real - rhs.real,
            imag: self.imag - rhs.imag,
        };
    }
}

impl Eq<Complex> for Complex {
    fn eq(self, rhs: Complex) -> bool {
        return self.real == rhs.real && self.imag == rhs.imag;
    }
}

fn main() {
    let c1 = Complex { real: 1.5, imag: 2.0 };
    let c2 = Complex { real: 3.5, imag: 1.0 };

    let sum = c1 + c2;
    let diff = c1 - c2;
    
    print("c1 = ");
    println(c1);
    
    print("c2 = ");
    println(c2);
    
    print("Sum (c1 + c2) = ");
    println(sum);
    
    print("Diff (c1 - c2) = ");
    println(diff);
    
    let is_eq = c1 == c2;
    println(format("c1 == c2: {}", is_eq));
    
    let is_same = c1 == c1;
    println(format("c1 == c1: {}", is_same));
}
