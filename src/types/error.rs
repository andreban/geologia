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

/// The top-level error envelope returned by the Gemini API, wrapping a [`VertexApiError`]
/// under an `error` key (e.g. `{ "error": { "code": 400, "status": "INVALID_ARGUMENT", .. } }`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VertexApiErrorResponse {
    /// The wrapped error payload.
    pub error: VertexApiError,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gemini_error_envelope() {
        let body = r#"{
            "error": {
                "code": 400,
                "message": "Invalid value at 'contents[0].parts[0]'",
                "status": "INVALID_ARGUMENT",
                "details": [{"@type": "type.googleapis.com/google.rpc.BadRequest"}]
            }
        }"#;

        let parsed: VertexApiErrorResponse =
            serde_json::from_str(body).expect("should parse error envelope");
        assert_eq!(parsed.error.code, 400);
        assert_eq!(parsed.error.status, "INVALID_ARGUMENT");
        assert_eq!(
            parsed.error.message,
            "Invalid value at 'contents[0].parts[0]'"
        );
        assert_eq!(parsed.error.details.unwrap().len(), 1);
    }
}
