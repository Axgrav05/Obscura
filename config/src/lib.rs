use serde::{Deserialize, Serialize};
use std::fs;

#[derive(Debug, Deserialize, Serialize, Default)]
pub struct Config {
    #[serde(default)]
    pub disabled_entities: Vec<String>,
}

impl Config {
    pub fn load_from_file(path: &str) -> anyhow::Result<Self> {
        let content = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        Ok(config)
    }
}
