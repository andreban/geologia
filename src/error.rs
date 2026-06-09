// Copyright 2026 Andre Cipriani Bandarra
// SPDX-License-Identifier: Apache-2.0

//! Error types for the Google Gemini client.

use std::fmt::Display;
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
    /// The API response contained no candidate completions.
    NoCandidatesError,
    /// An error occurred while decoding the SSE event stream.
    EventSourceError(LinesCodecError),
    /// The SSE event stream closed unexpectedly.
    EventSourceClosedError,
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
            Error::NoCandidatesError => {
                write!(f, "No candidates returned for the prompt")
            }
            Error::EventSourceError(e) => {
                write!(f, "EventSource Error: {e}")
            }
            Error::EventSourceClosedError => {
                write!(f, "EventSource closed error")
            }
        }
    }
}

impl std::error::Error for Error {}

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
