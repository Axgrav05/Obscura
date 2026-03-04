use core::convert::Infallible;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use tokio::net::TcpListener;
use config::Config;
use inference::ModelEnvironment;

async fn handle_request(
    req: Request<hyper::body::Incoming>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if req.uri().path() == "/health" {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(r#"{"status": "ok"}"#)))
            .unwrap();
        return Ok(response);
    }

    let response = Response::builder()
        .status(StatusCode::OK)
        .header("Content-Type", "application/json")
        .body(Full::new(Bytes::from(
            r#"{"message": "Hello from Obscura Proxy"}"#,
        )))
        .unwrap();
    Ok(response)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let _config = Config::load_from_file("obscura.toml").unwrap_or_else(|e| {
        tracing::warn!("Failed to load obscura.toml config, using defaults: {}", e);
        Config::default()
    });

    let _model_env = ModelEnvironment::load()
        .expect("Failed to load model environment, crashing on startup (Fail-Closed policy)");

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(handle_request))
                .await
            {
                tracing::error!("Error serving connection: {:?}", err);
            }
        });
    }
}
