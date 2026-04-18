// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::prelude::VertexApiError;

/// Request body for the text embeddings `predict` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingRequest {
    /// The list of text instances to embed.
    pub instances: Vec<TextEmbeddingRequestInstance>,
}

/// A single text instance to embed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingRequestInstance {
    /// The text content to generate an embedding for.
    pub content: String,
    /// The task type for the embedding (e.g. `"RETRIEVAL_DOCUMENT"`, `"RETRIEVAL_QUERY"`).
    pub task_type: String,
    /// An optional title for the content (used with retrieval task types).
    pub title: Option<String>,
}

/// The raw response from the text embeddings endpoint, which may be a success or an error.
///
/// Use [`into_result`](TextEmbeddingResponse::into_result) to convert into a standard `Result`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextEmbeddingResponse {
    Ok(TextEmbeddingResponseOk),
    Error { error: VertexApiError },
}

impl TextEmbeddingResponse {
    /// Converts this response into a `Result`, mapping the error variant to [`crate::error::Error`].
    pub fn into_result(self) -> Result<TextEmbeddingResponseOk> {
        self.into()
    }
}

/// A successful response from the text embeddings endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingResponseOk {
    /// The embedding predictions, one per input instance.
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

/// A single embedding prediction.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingPrediction {
    /// The embedding result containing the vector and statistics.
    pub embeddings: TextEmbeddingResult,
}

/// The embedding vector and associated statistics.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingResult {
    /// Statistics about the embedding computation.
    pub statistics: TextEmbeddingStatistics,
    /// The embedding vector.
    pub values: Vec<f64>,
}

/// Statistics about a text embedding computation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TextEmbeddingStatistics {
    /// Whether the input was truncated to fit the model's context window.
    pub truncated: bool,
    /// The number of tokens in the input.
    pub token_count: u32,
}
