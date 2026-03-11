use config::Config;
use core::convert::Infallible;
use http_body_util::Full;
use hyper::body::Bytes;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use inference::redact::redact;
use inference::redact::rehydrate;
use inference::{ModelEnvironment, NerModel};
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;

async fn handle_request(
    mut req: Request<hyper::body::Incoming>,
    model: Arc<NerModel>,
) -> Result<Response<Full<Bytes>>, Infallible> {
    if req.uri().path() == "/health" {
        let response = Response::builder()
            .status(StatusCode::OK)
            .header("Content-Type", "application/json")
            .body(Full::new(Bytes::from(r#"{"status": "ok"}"#)))
            .unwrap();
        return Ok(response);
    }

    let config = Config::load_from_file("obscura.toml").unwrap_or_default();
    let upstream_url = config.upstream_url.clone();

    // Parse X-Obscura-Skip-Redaction header
    let mut skipped_entities: Vec<String> = config.disabled_entities.clone();
    if let Some(skip_header) = req.headers().get("X-Obscura-Skip-Redaction") {
        if let Ok(skip_str) = skip_header.to_str() {
            let overrides: Vec<String> =
                skip_str.split(',').map(|s| s.trim().to_string()).collect();
            skipped_entities.extend(overrides);
        }
    }

    // Build upstream URI
    let uri_string = format!(
        "{}{}",
        upstream_url,
        req.uri()
            .path_and_query()
            .map(|pq| pq.as_str())
            .unwrap_or("")
    );

    let client = reqwest::Client::new();
    let mut req_builder = client.request(req.method().clone(), uri_string.clone());

    // Forward headers (drop Host)
    for (key, value) in req.headers().iter() {
        if key != hyper::header::HOST {
            req_builder = req_builder.header(key, value);
        }
    }

    // Collect body bytes
    let body_bytes = http_body_util::BodyExt::collect(req.into_body())
        .await
        .unwrap()
        .to_bytes();

    // --- OBS-14c: Redact PII from request body before forwarding ---
    let (redacted_body, mapping) = match redact_body(&body_bytes, &model, &skipped_entities) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!("Redaction failed, blocking request (fail-closed): {}", e);
            return Ok(Response::builder()
                .status(StatusCode::INTERNAL_SERVER_ERROR)
                .body(Full::new(Bytes::from("Redaction error")))
                .unwrap());
        }
    };

    req_builder = req_builder.body(redacted_body.clone());

    match req_builder.send().await {
        Ok(upstream_resp) => {
            let mut response = Response::builder().status(upstream_resp.status());

            for (key, value) in upstream_resp.headers().iter() {
                response = response.header(key, value);
            }

            let resp_bytes = upstream_resp.bytes().await.unwrap();

            // --- OBS-14d: Rehydrate PII tokens in LLM response ---
            let final_bytes = rehydrate_body(resp_bytes, &mapping);

            Ok(response.body(Full::new(final_bytes)).unwrap())
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

/// Extract the `messages` content from an OpenAI-style JSON body,
/// run NER + redaction, and return the redacted bytes + mapping.
fn redact_body(
    body: &Bytes,
    model: &NerModel,
    skipped_entities: &[String],
) -> anyhow::Result<(Bytes, inference::mapping::MappingDictionary)> {
    let mut json: serde_json::Value = serde_json::from_slice(body)?;

    if let Some(messages) = json.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for message in messages.iter_mut() {
            if let Some(content) = message.get_mut("content").and_then(|c| c.as_str()) {
                let content_str = content.to_string();
                let spans = model.predict(&content_str)?;
                let (redacted, mapping) = redact(&content_str, spans, skipped_entities)?;
                *message.get_mut("content").unwrap() = serde_json::Value::String(redacted);
                // Return after first content field for now (non-streaming, single message)
                return Ok((Bytes::from(serde_json::to_vec(&json)?), mapping));
            }
        }
    }

    // Non-chat body: pass through unchanged with empty mapping
    Ok((
        body.clone(),
        inference::mapping::MappingDictionary::new(),
    ))
}

/// OBS-14d: Find-replace mapping tokens in the LLM response with original values.
fn rehydrate_body(
    bytes: Bytes,
    mapping: &inference::mapping::MappingDictionary,
) -> Bytes {
    if mapping.mappings.is_empty() {
        return bytes;
    }

    let Ok(text) = std::str::from_utf8(&bytes) else {
        tracing::warn!("Response body is not valid UTF-8, skipping rehydration");
        return bytes;
    };

    Bytes::from(rehydrate(text, mapping).into_bytes())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let config = Config::load_from_file("obscura.toml").unwrap_or_else(|e| {
        tracing::warn!("Failed to load obscura.toml, using defaults: {}", e);
        Config::default()
    });

    let model_env = ModelEnvironment::load()
        .expect("Failed to load model environment (fail-closed)");

    let model = Arc::new(
        NerModel::load(&model_env).expect("Failed to load NER model (fail-closed)"),
    );

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    let listener = TcpListener::bind(addr).await?;
    tracing::info!("Listening on http://{}", addr);

    loop {
        let (stream, _) = listener.accept().await?;
        let io = TokioIo::new(stream);
        let model = Arc::clone(&model);
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req| handle_request(req, Arc::clone(&model))),
                )
                .await
            {
                tracing::error!("Error serving connection: {:?}", err);
            }
        });
    }
}
