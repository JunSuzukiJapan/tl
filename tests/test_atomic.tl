fn main() {
    let a = AtomicI64::new(10);
    let val = a.load();
    if val == 10 {
        tl_print_string("AtomicI64 ロード成功！\n");
    } else {
        tl_print_string("AtomicI64 ロードエラー！\n");
    }

    a.store(20);
    if a.load() == 20 {
        tl_print_string("AtomicI64 ストア成功！\n");
    }

    let prev = a.add(5);
    if prev == 20 && a.load() == 25 {
        tl_print_string("AtomicI64 加算成功！\n");
    }

    let success = a.compare_exchange(25, 30);
    if success && a.load() == 30 {
        tl_print_string("AtomicI64 cmpxchg 成功！\n");
    }

    // AtomicI32 Test
    let a32 = AtomicI32::new(100 as i32);
    if a32.load() == 100 as i32 {
        tl_print_string("AtomicI32 ロード成功！\n");
    }
    a32.add(50 as i32);
    if a32.load() == 150 as i32 {
        tl_print_string("AtomicI32 加算成功！\n");
    }
}
