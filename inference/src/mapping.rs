use std::collections::HashMap;

#[derive(Debug, Default)]
pub struct MappingDictionary {
    // Dynamic string keys (e.g., "[DOB_1]", "[PERSON_2]") mapping to original PII text
    pub mappings: HashMap<String, String>,
}

impl MappingDictionary {
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    pub fn insert(&mut self, token: String, original_text: String) {
        self.mappings.insert(token, original_text);
    }
    
    pub fn get(&self, token: &str) -> Option<&String> {
        self.mappings.get(token)
    }
}
