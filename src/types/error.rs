use std::fmt::Formatter;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexApiError {
    pub code: i32,
    pub message: String,
    pub status: String,
    pub details: Option<Vec<serde_json::Value>>,
}

impl core::fmt::Display for VertexApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Vertex API Error {} - {}", self.code, self.message)?;
        Ok(())
    }
}

impl std::error::Error for VertexApiError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeminiApiError {
    pub error: VertexApiError,
}

impl core::fmt::Display for GeminiApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        write!(
            f,
            "Gemini API Error {} - {}",
            self.error.code, self.error.message
        )
    }
}

impl std::error::Error for GeminiApiError {}
