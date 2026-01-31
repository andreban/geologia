use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

use super::Content;

/// Request body for the `countTokens` endpoint.
///
/// Use [`CountTokensRequest::builder`] for ergonomic construction.
///
/// See <https://ai.google.dev/api/tokens#method:-models.counttokens>.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CountTokensRequest {
    /// The content to count tokens for.
    pub contents: Content,
}

impl CountTokensRequest {
    /// Returns a new [`CountTokensRequestBuilder`].
    pub fn builder() -> CountTokensRequestBuilder {
        CountTokensRequestBuilder::default()
    }
}

/// Builder for [`CountTokensRequest`].
#[derive(Debug, Default)]
pub struct CountTokensRequestBuilder {
    contents: Content,
}

impl CountTokensRequestBuilder {
    /// Creates a builder pre-populated with a single text prompt.
    pub fn from_prompt(prompt: &str) -> Self {
        CountTokensRequestBuilder {
            contents: Content {
                parts: Some(vec![super::Part::from_text(prompt.to_string())]),
                ..Default::default()
            },
        }
    }

    /// Consumes the builder and returns the constructed [`CountTokensRequest`].
    pub fn build(self) -> CountTokensRequest {
        CountTokensRequest {
            contents: self.contents,
        }
    }
}

/// The raw response from the `countTokens` endpoint, which may be a success or an error.
///
/// Use [`into_result`](CountTokensResponse::into_result) to convert into a standard `Result`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CountTokensResponse {
    Ok(CountTokensResponseResult),
    Error { error: super::VertexApiError },
}

impl CountTokensResponse {
    /// Converts this response into a `Result`, mapping the error variant to [`crate::error::Error`].
    pub fn into_result(self) -> Result<CountTokensResponseResult> {
        match self {
            CountTokensResponse::Ok(result) => Ok(result),
            CountTokensResponse::Error { error } => Err(Error::VertexError(error)),
        }
    }
}

/// A successful response from the `countTokens` endpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CountTokensResponseResult {
    /// The total number of tokens in the input.
    pub total_tokens: i32,
    /// The total number of billable characters in the input.
    pub total_billable_characters: u32,
}
