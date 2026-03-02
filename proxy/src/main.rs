use std::net::SocketAddr;
use std::sync::Arc;
use bytes::Bytes;
use http_body_util::Full;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode, body::Incoming};
use hyper_util::rt::TokioIo;
use tokio::net::TcpListener;

use obscura_config::Config;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // OBS-1c: Load configuration
    let config = Config::load().expect("Failed to load obscura.toml");
    let addr: SocketAddr = format!("{}:{}", config.server.host, config.server.port).parse()?;
    
    let listener = TcpListener::bind(addr).await?;
    println!("Obscura Proxy online at http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);

        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(handle_request))
                .await
            {
                eprintln!("Error handling connection: {:?}", err);
            }
        });
    }
}

async fn handle_request(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::http::Error> {
    match req.uri().path() {
        // OBS-1b: Health check endpoint
        "/health" => {
            let body = serde_json::json!({ "status": "ok" }).to_string();
            Ok(Response::builder()
                .status(StatusCode::OK)
                .header("Content-Type", "application/json")
                .body(Full::new(Bytes::from(body)))?)
        }
        _ => Ok(Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Full::new(Bytes::from("Not Found")))?)
    }
}
