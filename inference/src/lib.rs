use std::env;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Context;
use ort::session::Session;
use tokenizers::Tokenizer;

pub mod mapping;
pub mod ner;
pub mod redact;

#[derive(Debug)]
pub struct ModelEnvironment {
    pub model_path: PathBuf,
    pub tokenizer_path: PathBuf,
}

impl ModelEnvironment {
    pub fn load() -> anyhow::Result<Self> {
        let model_path = PathBuf::from(
            env::var("NER_MODEL_PATH")
                .map_err(|_| anyhow::anyhow!("NER_MODEL_PATH environment variable is missing"))?,
        );
        let tokenizer_path = PathBuf::from(
            env::var("NER_TOKENIZER_PATH")
                .map_err(|_| anyhow::anyhow!("NER_TOKENIZER_PATH environment variable is missing"))?,
        );

        if !model_path.exists() {
            anyhow::bail!(
                "NER_MODEL_PATH points to non-existent file: {}",
                model_path.display()
            );
        }
        if !tokenizer_path.exists() {
            anyhow::bail!(
                "NER_TOKENIZER_PATH points to non-existent file: {}",
                tokenizer_path.display()
            );
        }

        Ok(Self {
            model_path,
            tokenizer_path,
        })
    }
}

pub struct NerModel {
    /// ort v2: Session::run() requires &mut self.
    /// Wrapped in Mutex for interior mutability so NerModel is shareable via Arc.
    pub session: Mutex<Session>,
    pub tokenizer: Tokenizer,
}

impl NerModel {
    pub fn load(env: &ModelEnvironment) -> anyhow::Result<Self> {
        // ort v2 API: no Environment — Session::builder() is the entry point.
        let session = Session::builder()
            .context("Failed to create ORT session builder")?
            .commit_from_file(&env.model_path)
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(&env.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
        })
    }
}
