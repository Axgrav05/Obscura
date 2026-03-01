pub trait ProviderAdapter {
    fn extract_request_text(&self, body: &str) -> Option<String>;
    fn extract_response_text(&self, body: &str) -> Option<String>;
}
