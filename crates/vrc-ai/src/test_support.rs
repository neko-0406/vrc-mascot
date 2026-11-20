//! Local `tokio::net::TcpListener` mock HTTP/1.1 server for unit and integration tests.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio::task::JoinHandle;

#[derive(Debug, Clone)]
pub(crate) struct RecordedRequest {
    pub method: String,
    pub target: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

#[derive(Debug, Clone)]
pub(crate) struct MockResponse {
    pub status: u16,
    pub content_type: String,
    pub body: Vec<u8>,
}

impl MockResponse {
    pub fn json(status: u16, body: impl Into<String>) -> Self {
        Self {
            status,
            content_type: "application/json; charset=utf-8".to_owned(),
            body: body.into().into_bytes(),
        }
    }

    pub fn bytes(status: u16, content_type: impl Into<String>, body: Vec<u8>) -> Self {
        Self {
            status,
            content_type: content_type.into(),
            body,
        }
    }
}

pub(crate) struct MockHttpServer {
    addr: SocketAddr,
    requests: Arc<Mutex<Vec<RecordedRequest>>>,
    handle: JoinHandle<()>,
}

impl MockHttpServer {
    pub async fn start<F>(handler: F) -> Self
    where
        F: Fn(&RecordedRequest) -> MockResponse + Send + Sync + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind local mock listener");
        let addr = listener.local_addr().expect("mock listener local addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let requests_bg = Arc::clone(&requests);
        let handler = Arc::new(handler);

        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let requests_conn = Arc::clone(&requests_bg);
                let handler_conn = Arc::clone(&handler);
                tokio::spawn(async move {
                    let Some(req) = read_http_request(&mut stream).await else {
                        return;
                    };
                    let resp = handler_conn(&req);
                    requests_conn.lock().expect("lock requests").push(req);

                    let reason = match resp.status {
                        200 => "OK",
                        400 => "Bad Request",
                        401 => "Unauthorized",
                        404 => "Not Found",
                        500 => "Internal Server Error",
                        _ => "Status",
                    };
                    let header = format!(
                        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        resp.status,
                        reason,
                        resp.content_type,
                        resp.body.len()
                    );
                    let _ = stream.write_all(header.as_bytes()).await;
                    let _ = stream.write_all(&resp.body).await;
                    let _ = stream.shutdown().await;
                });
            }
        });

        Self {
            addr,
            requests,
            handle,
        }
    }

    pub fn base_url(&self) -> String {
        format!("http://{}", self.addr)
    }

    pub fn requests(&self) -> Vec<RecordedRequest> {
        self.requests.lock().expect("lock requests").clone()
    }
}

impl Drop for MockHttpServer {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Option<RecordedRequest> {
    let mut buf = Vec::with_capacity(2048);
    let mut chunk = [0_u8; 1024];

    let header_end = loop {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            return None;
        }
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = find_header_end(&buf) {
            break pos;
        }
    };

    let header_text = String::from_utf8_lossy(&buf[..header_end]);
    let mut lines = header_text.split("\r\n");
    let request_line = lines.next()?;
    let mut parts = request_line.split_whitespace();
    let method = parts.next()?.to_owned();
    let target = parts.next()?.to_owned();

    let mut headers = HashMap::new();
    let mut content_length = 0_usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().to_owned();
            if key == "content-length" {
                content_length = val.parse::<usize>().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }

    let body_start = header_end + 4;
    while buf.len() < body_start + content_length {
        let n = stream.read(&mut chunk).await.ok()?;
        if n == 0 {
            break;
        }
        buf.extend_from_slice(&chunk[..n]);
    }
    let body_end = (body_start + content_length).min(buf.len());
    let body = String::from_utf8_lossy(&buf[body_start..body_end]).into_owned();

    Some(RecordedRequest {
        method,
        target,
        headers,
        body,
    })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n")
}
