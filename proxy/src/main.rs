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
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use serde_json;

async fn handle_request(
    req: Request<hyper::body::Incoming>,
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

/// Remap a local per-message token (e.g. `[PERSON_1]`) to a globally unique
/// token using `counters` shared across all messages in the request.
///
/// Without this, two messages that each produce `[PERSON_1]` for different
/// people would collide in the merged mapping — last write would win.
fn bump_token(local_token: &str, counters: &mut HashMap<String, usize>) -> String {
    let inner = local_token.trim_start_matches('[').trim_end_matches(']');
    if let Some(idx) = inner.rfind('_') {
        let label = &inner[..idx];
        let count = counters.entry(label.to_string()).or_insert(0);
        *count += 1;
        format!("[{}_{}]", label, count)
    } else {
        local_token.to_string() // fallback: pass through unchanged
    }
}

/// Extract the `messages` content from an OpenAI-style JSON body,
/// run NER + redaction on ALL messages, and return the redacted bytes + mapping.
///
/// Tokens are namespaced globally across all messages via `bump_token` to
/// prevent cross-message token collisions in the merged MappingDictionary.
fn redact_body(
    body: &Bytes,
    model: &NerModel,
    skipped_entities: &[String],
) -> anyhow::Result<(Bytes, inference::mapping::MappingDictionary)> {
    let mut json: serde_json::Value = serde_json::from_slice(body)?;
    let mut combined_mapping = inference::mapping::MappingDictionary::new();
    let mut global_counters: HashMap<String, usize> = HashMap::new();

    if let Some(messages) = json.get_mut("messages").and_then(|m| m.as_array_mut()) {
        for message in messages.iter_mut() {
            if let Some(content_val) = message.get_mut("content") {
                if let Some(content) = content_val.as_str() {
                    let content_str = content.to_string();
                    let spans = model.predict(&content_str)?;
                    let (redacted, msg_mapping) =
                        redact(&content_str, spans, skipped_entities)?;

                    // Remap local tokens to globally unique tokens so that
                    // [PERSON_1] from message 0 and [PERSON_1] from message 1
                    // become [PERSON_1] and [PERSON_2] in the merged mapping.
                    let mut remapped = redacted;
                    for (local_token, original) in msg_mapping.mappings {
                        let global_token = bump_token(&local_token, &mut global_counters);
                        remapped = remapped.replace(&local_token, &global_token);
                        combined_mapping.insert(global_token, original);
                    }

                    *content_val = serde_json::Value::String(remapped);
                }
            }
        }
        return Ok((Bytes::from(serde_json::to_vec(&json)?), combined_mapping));
    }

    // Unknown schema — fail closed: block rather than forward PII unredacted.
    // Covers non-chat formats like {"prompt": "..."} that don't have a "messages" array.
    anyhow::bail!("Unrecognised request schema: no 'messages' array found")
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

    // Load config once at startup — passed as Arc<Config> to each request handler.
    let config = Arc::new(Config::load_from_file("obscura.toml").unwrap_or_else(|e| {
        tracing::warn!("Failed to load obscura.toml, using defaults: {}", e);
        Config::default()
    }));

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
        let config = Arc::clone(&config);
        tokio::task::spawn(async move {
            if let Err(err) = http1::Builder::new()
                .serve_connection(
                    io,
                    service_fn(move |req| {
                        handle_request(req, Arc::clone(&model), Arc::clone(&config))
                    }),
                )
                .await
            {
                tracing::error!("Error serving connection: {:?}", err);
            }
        });
    }
}
