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
        write!(f, "Gemini API Error {} - {}", self.error.code, self.error.message)
    }
}

impl std::error::Error for GeminiApiError {}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub description: String,
    pub url: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "@type")]
pub enum ErrorType {
    #[serde(rename = "type.googleapis.com/google.rpc.ErrorInfo")]
    ErrorInfo { metadata: ErrorInfoMetadata },

    #[serde(rename = "type.googleapis.com/google.rpc.Help")]
    Help { links: Vec<Link> },

    #[serde(rename = "type.googleapis.com/google.rpc.BadRequest")]
    BadRequest {
        #[serde(rename = "fieldViolations")]
        field_violations: Vec<FieldViolation>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorInfoMetadata {
    pub service: String,
    pub consumer: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldViolation {
    pub field: String,
    pub description: String,
}
