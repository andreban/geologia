use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::VertexApiError;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingRequest {
    pub instances: Vec<TextEmbeddingRequestInstance>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingRequestInstance {
    pub content: String,
    pub task_type: String,
    pub title: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextEmbeddingResponse {
    Ok(TextEmbeddingResponseOk),
    Error { error: VertexApiError },
}

impl TextEmbeddingResponse {
    pub fn into_result(self) -> Result<TextEmbeddingResponseOk> {
        self.into()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingResponseOk {
    pub predictions: Vec<TextEmbeddingPrediction>,
}

impl From<TextEmbeddingResponse> for Result<TextEmbeddingResponseOk> {
    fn from(value: TextEmbeddingResponse) -> Self {
        match value {
            TextEmbeddingResponse::Ok(ok) => Ok(ok),
            TextEmbeddingResponse::Error { error } => Err(Error::VertexError(error)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingPrediction {
    pub embeddings: TextEmbeddingResult,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingResult {
    pub statistics: TextEmbeddingStatistics,
    pub values: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingStatistics {
    pub truncated: bool,
    pub token_count: u32,
}
