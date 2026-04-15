fn main() {
    let f1 = F32::nan();
    let f2 = F32::infinity();
    let f3 = F64::max_value();
    let i1 = I64::min_value();

    if f1.is_nan() { }
    if f2.is_inf() { }
}
