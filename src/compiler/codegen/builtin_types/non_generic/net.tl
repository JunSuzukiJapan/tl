extern fn tl_net_listener_bind(addr: String) -> i64;
extern fn tl_net_listener_accept(listener_id: i64) -> i64;

extern fn tl_net_stream_connect(addr: String) -> i64;
extern fn tl_net_stream_write(stream_id: i64, buf: Vec<u8>) -> i64;
extern fn tl_net_stream_read(stream_id: i64, max_len: i64) -> Vec<u8>;

struct TcpListener {
    id: i64,
}

impl TcpListener {
    fn bind(addr: String) -> Result<TcpListener, String> {
        let id = tl_net_listener_bind(addr);
        if id > 0 {
            let res: Result<TcpListener, String> = Result::Ok(TcpListener { id: id });
            res
        } else {
            let res: Result<TcpListener, String> = Result::Err("Failed to bind");
            res
        }
    }

    fn accept(self) -> Result<TcpStream, String> {
        let id = tl_net_listener_accept(self.id);
        if id > 0 {
            let res: Result<TcpStream, String> = Result::Ok(TcpStream { id: id });
            res
        } else {
            let res: Result<TcpStream, String> = Result::Err("Failed to accept");
            res
        }
    }
}

struct TcpStream {
    id: i64,
}

impl TcpStream {
    fn connect(addr: String) -> Result<TcpStream, String> {
        let id = tl_net_stream_connect(addr);
        if id > 0 {
            let res: Result<TcpStream, String> = Result::Ok(TcpStream { id: id });
            res
        } else {
            let res: Result<TcpStream, String> = Result::Err("Failed to connect");
            res
        }
    }

    fn write(self, buf: Vec<u8>) -> Result<i64, String> {
        let n = tl_net_stream_write(self.id, buf);
        if n >= 0 {
            let res: Result<i64, String> = Result::Ok(n);
            res
        } else {
            let res: Result<i64, String> = Result::Err("Failed to write");
            res
        }
    }

    fn read(self, max_len: i64) -> Result<Vec<u8>, String> {
        let v = tl_net_stream_read(self.id, max_len);
        let res: Result<Vec<u8>, String> = Result::Ok(v);
        res
    }
}
