// ロードマップ フェーズ3: 形状操作 網羅テスト (chunk/split 追加)
fn main() {
    // --- 3.1 expand / broadcast_to ---
    let t1: Tensor<f32, 2> = [[1.0, 2.0, 3.0]];
    let expanded = t1.expand([3, 3]);
    print("expand([1,3] -> [3,3]):");
    print(expanded);

    let bc = t1.broadcast_to([3, 3]);
    print("broadcast_to([1,3] -> [3,3]):");
    print(bc);

    // --- 3.2 reshape ---
    let t3 = Tensor::full([2, 3], 2.0);
    let reshaped = t3.reshape([6]);
    print("reshape([2,3] -> [6]):");
    print(reshaped);

    // --- 3.3 stack ---
    let a = Tensor::full([3], 1.0);
    let b = Tensor::full([3], 2.0);
    let stacked = Tensor::stack(a, b, 0);
    print("stack([1,1,1],[2,2,2],0):");
    print(stacked);

    // --- 3.4 narrow ---
    let mut nd = Vec::new();
    nd.push(10.0);
    nd.push(20.0);
    nd.push(30.0);
    nd.push(40.0);
    nd.push(50.0);
    nd.push(60.0);
    let mut ns = Vec::new();
    ns.push(6);
    let t4 = Tensor::from_vec(nd, ns);
    let nar = t4.narrow(0, 1, 3);
    print("narrow(0, 1, 3):");
    print(nar);

    // --- 3.5 transpose ---
    let t5 = Tensor::full([2, 3], 5.0);
    let tp = t5.transpose(0, 1);
    print("transpose([2,3], 0, 1):");
    print(tp);

    // --- 3.6 contiguous ---
    let t6 = Tensor::full([3, 2], 3.0);
    let c = t6.contiguous();
    print("contiguous:");
    print(c);

    // --- 3.7 chunk ---
    let c0 = t4.chunk(3, 0, 0);
    print("chunk(3,0,0):");
    print(c0);
    let c1 = t4.chunk(3, 0, 1);
    print("chunk(3,0,1):");
    print(c1);
    let c2 = t4.chunk(3, 0, 2);
    print("chunk(3,0,2):");
    print(c2);

    // --- 3.8 split ---
    let s0 = t4.split(2, 0, 0);
    print("split(2,0,0):");
    print(s0);
    let s1 = t4.split(2, 0, 1);
    print("split(2,0,1):");
    print(s1);
    let s2 = t4.split(2, 0, 2);
    print("split(2,0,2):");
    print(s2);
}
