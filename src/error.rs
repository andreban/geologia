// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Google Gemini client.

use std::fmt::Display;
use std::time::Duration;
use tokio_util::codec::LinesCodecError;

use crate::types;

/// A type alias for `Result<T, error::Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur when using the Gemini client.
#[derive(Debug)]
pub enum Error {
    /// An environment variable required for configuration was missing or invalid.
    Env(std::env::VarError),
    /// An HTTP transport error from the underlying `reqwest` client.
    HttpClient(reqwest::Error),
    /// A JSON serialization or deserialization error.
    Serde(serde_json::Error),
    /// A structured error returned by the Vertex AI API.
    VertexError(types::VertexApiError),
    /// An HTTP error status whose response body could not be parsed as a structured
    /// Gemini error. Carries the status and the raw response body for diagnosis.
    ApiError {
        /// The HTTP status code returned by the API.
        status: reqwest::StatusCode,
        /// The raw response body.
        body: String,
    },
    /// A 2xx response from the API violated an expected protocol contract
    /// (for example, a resumable upload initiation response missing the
    /// `x-goog-upload-url` header).
    UnexpectedResponse {
        /// Description of the protocol or response invariant violation.
        message: String,
    },
    /// A request or builder argument violated a client-side validation constraint.
    InvalidArgument {
        /// Description of the invalid argument or missing required field.
        message: String,
    },
    /// An error occurred while reading or decoding the SSE event stream.
    EventSourceError(LinesCodecError),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            Error::Env(e) => write!(f, "Environment variable error: {e}"),
            Error::HttpClient(e) => write!(f, "HTTP Client error: {e}"),
            Error::Serde(e) => write!(f, "Serde error: {e}"),
            Error::VertexError(e) => {
                write!(f, "Vertex error: {e}")
            }
            Error::ApiError { status, body } => {
                write!(f, "HTTP error {status}: {body}")
            }
            Error::UnexpectedResponse { message } => {
                write!(f, "Unexpected API response: {message}")
            }
            Error::InvalidArgument { message } => {
                write!(f, "Invalid argument: {message}")
            }
            Error::EventSourceError(e) => {
                write!(f, "EventSource Error: {e}")
            }
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Env(e) => Some(e),
            Error::HttpClient(e) => Some(e),
            Error::Serde(e) => Some(e),
            Error::VertexError(e) => Some(e),
            Error::ApiError { .. }
            | Error::UnexpectedResponse { .. }
            | Error::InvalidArgument { .. } => None,
            Error::EventSourceError(e) => Some(e),
        }
    }
}

impl Error {
    /// Returns the recommended retry delay if this error is a [`Error::VertexError`]
    /// containing a `RetryInfo` entry.
    pub fn retry_delay(&self) -> Option<Duration> {
        match self {
            Error::VertexError(e) => e.retry_delay(),
            _ => None,
        }
    }
}

impl From<reqwest::Error> for Error {
    fn from(e: reqwest::Error) -> Self {
        Error::HttpClient(e)
    }
}

impl From<std::env::VarError> for Error {
    fn from(e: std::env::VarError) -> Self {
        Error::Env(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Serde(e)
    }
}

impl From<types::VertexApiError> for Error {
    fn from(e: types::VertexApiError) -> Self {
        Error::VertexError(e)
    }
}

impl From<LinesCodecError> for Error {
    fn from(e: LinesCodecError) -> Self {
        Error::EventSourceError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_retry_delay_delegates_to_vertex_error() {
        let vertex_err = types::VertexApiError {
            code: 429,
            message: "Quota exceeded".into(),
            status: "RESOURCE_EXHAUSTED".into(),
            details: Some(vec![serde_json::json!({
                "@type": "type.googleapis.com/google.rpc.RetryInfo",
                "retryDelay": "15s"
            })]),
        };
        let err = Error::VertexError(vertex_err);
        assert_eq!(err.retry_delay(), Some(Duration::from_secs(15)));

        let other_err = Error::EventSourceError(LinesCodecError::MaxLineLengthExceeded);
        assert_eq!(other_err.retry_delay(), None);
    }

    #[test]
    fn unexpected_response_formats_message_and_has_no_source() {
        let err = Error::UnexpectedResponse {
            message: "missing x-goog-upload-url header".into(),
        };
        assert_eq!(
            err.to_string(),
            "Unexpected API response: missing x-goog-upload-url header"
        );
        assert!(std::error::Error::source(&err).is_none());
        assert_eq!(err.retry_delay(), None);
    }
}
