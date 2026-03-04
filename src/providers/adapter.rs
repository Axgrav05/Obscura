pub trait ProviderAdapter {
    fn extract_request_text(&self, body: &str) -> Option<String>;
    fn extract_response_text(&self, body: &str) -> Option<String>;

    // Method for streaming responses
    fn extract_response_delta_text(&self, _body: &str) -> Option<String> {
        None
    }
}
