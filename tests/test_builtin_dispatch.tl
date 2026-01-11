fn main() {
    println("Hello refactor!");
    let t = Tensor::randn([2, 2]);
    println(t);
    let s = sum(t);
    println(s);
}
