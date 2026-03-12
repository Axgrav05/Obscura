use config::Config;
use core::convert::Infallible;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use inference::{ModelEnvironment, NerModel};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

#[allow(clippy::collapsible_if)]
async fn handle_request(
    mut req: Request<hyper::body::Incoming>,
    model: Arc<NerModel>,
    config: Arc<Config>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if req.uri().path() == "/health" {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(r#"{"status": "ok"}"#)))
            .unwrap();
        return Ok(response);
    }

    let upstream_url = config.app.upstream_url.clone();
    
    // OBS-7e: Parse X-Obscura-Skip-Redaction (SECURE REFINEMENT)
    let mut skipped_entities: Vec<String> = config.app.disabled_entities.clone();
    
    let client_is_authorized = req.headers().get("X-Api-Key")
        .and_then(|h| h.to_str().ok())
        .map(|s| s == config.app.api_key)
        .unwrap_or(false);

    if let Some(skip_header) = req.headers().get("X-Obscura-Skip-Redaction") {
        if client_is_authorized {
            if let Ok(skip_str) = skip_header.to_str() {
                let overrides: Vec<String> = skip_str.split(',').map(|s| s.trim().to_string()).collect();
                skipped_entities.extend(overrides);
            }
        } else {
            // PR FEEDBACK: Prevent unauthenticated skip
            return Ok(Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .body(Full::new(Bytes::from("Unauthorized: Cannot skip redaction without valid API key")))
                .unwrap());
        }
    }
    
    // Build upstream URI
    let path_and_query = req.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or("/");
    let uri_string = format!("{}{}", upstream_url, path_and_query);
    let uri = uri_string.parse::<hyper::Uri>().expect("Invalid URI");
    *req.uri_mut() = uri;

    let client = reqwest::Client::new();
    let mut req_builder = client.request(req.method().clone(), uri_string.clone());
    
    // Forward headers
    for (key, value) in req.headers().iter() {
        if key != hyper::header::HOST {
            req_builder = req_builder.header(key, value);
        }
    }

    // REDACTION LOGIC (PR FEEDBACK: Implementing actual redaction)
    let body_bytes = http_body_util::BodyExt::collect(req.into_body()).await.unwrap().to_bytes();
    let body_str = String::from_utf8_lossy(&body_bytes);
    
    // We only redact if not explicitly skipped
    let final_body = if skipped_entities.contains(&"*".to_string()) {
        body_bytes
    } else {
        let bert_spans = model.predict(&body_str).unwrap_or_default();
        let (redacted_str, _mapping) = inference::redact::redact(&body_str, bert_spans, &skipped_entities).unwrap();
        Bytes::from(redacted_str)
    };

    req_builder = req_builder.body(final_body);

    match req_builder.send().await {
        Ok(upstream_resp) => {
            let mut response = Response::builder().status(upstream_resp.status());
            
            for (key, value) in upstream_resp.headers().iter() {
                response = response.header(key, value);
            }
            
            let bytes = upstream_resp.bytes().await.unwrap();
            Ok(response.body(Full::new(bytes)).unwrap())
        }
        Err(e) => {
            tracing::error!("Upstream request failed: {}", e);
            Ok(Response::builder()
                .status(StatusCode::BAD_GATEWAY)
                .body(Full::new(Bytes::from("Bad Gateway")))
                .unwrap())
        }
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    tracing::info!("Loading config...");
    let config = Arc::new(Config::load_from_file("obscura.toml").unwrap_or_else(|e| {
        tracing::warn!("Failed to load obscura.toml config, using defaults: {}", e);
        Config::default()
    }));

    tracing::info!("Loading model environment...");
    let model_env = ModelEnvironment::load()
        .expect("Failed to load model environment, crashing on startup (Fail-Closed policy)");
    
    tracing::info!("Initializing NerModel (this may take a moment)...");
    let model = Arc::new(NerModel::load(&model_env).expect("Failed to load NerModel"));
    tracing::info!("NerModel loaded successfully.");

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let model_clone = Arc::clone(&model);
        let config_clone = Arc::clone(&config);
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(io, service_fn(move |req| {
                    handle_request(req, Arc::clone(&model_clone), Arc::clone(&config_clone))
                }))
                .await
            {
                tracing::error!("Error serving connection: {:?}", err);
            }
        });
    }
}
