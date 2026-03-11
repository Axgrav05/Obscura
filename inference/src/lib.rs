use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use ort::{Environment, ExecutionProvider, Session, SessionBuilder};
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
    pub session: Session,
    pub tokenizer: Tokenizer,
}

impl NerModel {
    pub fn load(env: &ModelEnvironment) -> anyhow::Result<Self> {
        let ort_env = Arc::new(
            Environment::builder()
                .with_name("obscura_ner")
                .with_execution_providers([ExecutionProvider::CPU(Default::default())])
                .build()
                .context("Failed to build ORT environment")?,
        );

        let session = SessionBuilder::new(&ort_env)
            .context("Failed to create ORT session builder")?
            .with_model_from_file(&env.model_path)
            .context("Failed to load ONNX model")?;

        let tokenizer = Tokenizer::from_file(&env.tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self { session, tokenizer })
    }
}
