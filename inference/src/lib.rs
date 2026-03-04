use std::env;

#[derive(Debug)]
pub struct ModelEnvironment {
    pub model_path: String,
    pub tokenizer_path: String,
}

impl ModelEnvironment {
    pub fn load() -> anyhow::Result<Self> {
        let model_path = env::var("NER_MODEL_PATH")
            .map_err(|_| anyhow::anyhow!("NER_MODEL_PATH environment variable is missing"))?;
        let tokenizer_path = env::var("NER_TOKENIZER_PATH")
            .map_err(|_| anyhow::anyhow!("NER_TOKENIZER_PATH environment variable is missing"))?;

        if !std::path::Path::new(&model_path).exists() {
            anyhow::bail!("NER_MODEL_PATH points to non-existent file/directory: {}", model_path);
        }
        if !std::path::Path::new(&tokenizer_path).exists() {
            anyhow::bail!("NER_TOKENIZER_PATH points to non-existent file/directory: {}", tokenizer_path);
        }

        Ok(Self {
            model_path,
            tokenizer_path,
        })
    }
}
