fn main() {
    let ch = Channel::new(10);
    ch.send(42);
    
    let v1 = ch.recv();
    if v1 == 42 {
        tl_print_string("受信成功！\n");
    } else {
        tl_print_string("受信エラー！\n");
    }

    let v2_opt = ch.try_recv();
    match v2_opt {
        Some(v) => {
            tl_print_string("予期しないデータを受信しました！\n");
        },
        None => {
            tl_print_string("正常にデータなし（Nothing）を確認しました！\n");
        }
    }
}
