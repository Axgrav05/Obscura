use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Config {
    #[serde(default)]
    pub app: AppConfig,
}

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub upstream_url: String,
    #[serde(default)]
    pub disabled_entities: Vec<String>,
    #[serde(default)]
    pub api_key: String,
}

impl Config {
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let mut config: Config = toml::from_str(&content)?;

        // Override API key from environment if present
        if let Ok(env_key) = std::env::var("OBSCURA_API_KEY") {
            config.app.api_key = env_key;
        }

        Ok(config)
    }
}
