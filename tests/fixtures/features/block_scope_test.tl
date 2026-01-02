fn main() {
    let x = 10;
    print(x); // 10

    {
        let x = 20;
        print(x); // 20
        let y = x + 5;
        print(y); // 25
    }

    print(x); // 10 (Original x is preserved)

    {
        print(x); // 10 (Access outer x)
        let z = 30;
        print(z); // 30
    }
    
    // Test returned value from block
    let res = {
        let a = 5;
        let b = 6;
        a * b // 30
    };
    print(res); // 30
    
    // Test nested shadowing
    {
        let x = 100;
        {
             let x = 200;
             print(x); // 200
        }
        print(x); // 100
    }
}
