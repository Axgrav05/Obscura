use crate::providers::adapter::ProviderAdapter;
use serde_json::Value;

fn extract_content_text(content: &Value) -> Option<String> {
    // Case 1: content is a plain string
    if let Some(s) = content.as_str() {
        return Some(s.to_string());
    }

    // Case 2: content is an array of parts, e.g. [{"type":"text","text":"hi"}]
    if let Some(parts) = content.as_array() {
        let mut out = String::new();

        for part in parts {
            let part_type = part.get("type").and_then(|t| t.as_str());
            if part_type == Some("text")
                && let Some(t) = part.get("text").and_then(|t| t.as_str())
            {
                out.push_str(t);
            }
        }

        if !out.is_empty() {
            return Some(out);
        }
    }

    None
}

pub struct OpenAIAdapter;

impl ProviderAdapter for OpenAIAdapter {
    fn extract_request_text(&self, body: &str) -> Option<String> {
        let v: Value = serde_json::from_str(body).ok()?;
        let messages = v.get("messages")?.as_array()?;

        let user_messages: Vec<String> = messages
            .iter()
            .filter_map(|m| {
                if m.get("role")?.as_str()? == "user" {
                    extract_content_text(m.get("content")?)
                } else {
                    None
                }
            })
            .collect();

        Some(user_messages.join("\n"))
    }

    fn extract_response_text(&self, body: &str) -> Option<String> {
        let v: Value = serde_json::from_str(body).ok()?;
        let choices = v.get("choices")?.as_array()?;

        let content_val = choices.first()?.get("message")?.get("content")?;

        extract_content_text(content_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    //This simulates “middleware logic” using the trait.
    fn run_pipeline(
        adapter: &dyn ProviderAdapter,
        req_body: &str,
        resp_body: &str,
    ) -> (Option<String>, Option<String>) {
        let req_text = adapter.extract_request_text(req_body);
        let resp_text = adapter.extract_response_text(resp_body);
        (req_text, resp_text)
    }

    #[test]
    fn extract_request_text_single_user_message() {
        let adapter = OpenAIAdapter;

        let sample = r#"
        {
            "model": "gpt-4o-mini",
            "messages": [
                { "role": "user", "content": "Hello world" }
            ]
        }
        "#;

        let result = adapter.extract_request_text(sample);
        assert_eq!(result.as_deref(), Some("Hello world"));
    }

    #[test]
    fn extract_request_text_multiple_user_messages() {
        let adapter = OpenAIAdapter;

        let sample = r#"
        {
            "messages": [
                { "role": "system", "content": "You are helpful." },
                { "role": "user", "content": "First message" },
                { "role": "assistant", "content": "Ok." },
                { "role": "user", "content": "Second message" }
            ]
        }
        "#;

        let result = adapter.extract_request_text(sample);
        assert_eq!(result.as_deref(), Some("First message\nSecond message"));
    }

    #[test]
    fn extract_response_text_basic() {
        let adapter = OpenAIAdapter;

        let sample = r#"
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Hi there!"
                    }
                }
            ]
        }
        "#;

        let result = adapter.extract_response_text(sample);
        assert_eq!(result.as_deref(), Some("Hi there!"));
    }

    #[test]
    fn extract_response_text_missing_choices_returns_none() {
        let adapter = OpenAIAdapter;

        let sample = r#"{ "id": "nope" }"#;

        let result = adapter.extract_response_text(sample);
        assert!(result.is_none());
    }

    #[test]
    fn extract_request_text_content_parts_array() {
        let adapter = OpenAIAdapter;

        let sample = r#"
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "Hello " },
                        { "type": "text", "text": "world" }
                    ]
                }
            ]
        }
        "#;

        let result = adapter.extract_request_text(sample);
        assert_eq!(result.as_deref(), Some("Hello world"));
    }

    #[test]
    fn extract_response_text_content_parts_array() {
        let adapter = OpenAIAdapter;

        let sample = r#"
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": [
                            { "type": "text", "text": "Part A " },
                            { "type": "text", "text": "Part B" }
                        ]
                    }
                }
            ]
        }
        "#;

        let result = adapter.extract_response_text(sample);
        assert_eq!(result.as_deref(), Some("Part A Part B"));
    }

    // integration-style test that runs both methods in sequence, simulating how they might be used together in a middleware pipeline.
    #[test]
    fn integration_pipeline_through_trait_object() {
        let adapter: Box<dyn ProviderAdapter> = Box::new(OpenAIAdapter);

        let req = r#"
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        { "type": "text", "text": "My name is " },
                        { "type": "text", "text": "Josef." }
                    ]
                }
            ]
        }
        "#;

        let resp = r#"
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "Nice to meet you, Josef."
                    }
                }
            ]
        }
        "#;

        let (req_text, resp_text) = run_pipeline(adapter.as_ref(), req, resp);

        assert_eq!(req_text.as_deref(), Some("My name is Josef."));
        assert_eq!(resp_text.as_deref(), Some("Nice to meet you, Josef."));
    }
}
