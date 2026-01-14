fn main() {
    println("Hello refactor!");
    let t = Tensor::randn([2, 2], false);
    println(t);
    let s = t.sum();
    println(s);
}
