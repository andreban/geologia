// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Formatter;

use serde::{Deserialize, Serialize};

/// A structured error returned by the Vertex AI / Gemini API.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexApiError {
    /// The HTTP status code.
    pub code: i32,
    /// A human-readable error message.
    pub message: String,
    /// The gRPC status string (e.g. `"INVALID_ARGUMENT"`).
    pub status: String,
    /// Optional additional error details.
    pub details: Option<Vec<serde_json::Value>>,
}

impl core::fmt::Display for VertexApiError {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), std::fmt::Error> {
        writeln!(f, "Vertex API Error {} - {}", self.code, self.message)?;
        Ok(())
    }
}

impl std::error::Error for VertexApiError {}

/// A wrapper around [`VertexApiError`] matching the Gemini API error response format.
///
/// The Gemini API nests the error details inside an `error` field.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeminiApiError {
    /// The inner error details.
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
