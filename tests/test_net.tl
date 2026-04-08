fn main() {
    let port = "127.0.0.1:8090";
    print("Testing network module...");
    
    let listener_res = TcpListener::bind(port);
    if listener_res.is_ok() {
        let listener = listener_res.unwrap();
        print("Bound to " + port);
    } else {
        print("Bind failed or port in use");
    }
}
